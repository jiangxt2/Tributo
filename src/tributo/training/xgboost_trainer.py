"""XGBoost distributed training entry point, based on Ray Train XGBoostTrainer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from pydantic import ConfigDict, Field

from tributo._common.config import StrictConfigModel
from tributo.data.base import S3Config
from tributo.integrations.broker import CancellationChecker
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    DataLoadingMode,
    ProblemType,
    ResourceHints,
)
from tributo.training.base import BaseTrainer
from tributo.training.registry import register
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data
    from ray.train.xgboost import XGBoostTrainer

logger = logging.getLogger(__name__)


# ── Pydantic config models ──


def _gain_as_float(gain: float | list[float]) -> float:
    """Normalize an XGBoost gain value (scalar or per-class list) to float."""
    if isinstance(gain, float):
        return gain
    return gain[0] if gain else 0.0


class XGBoostDataConfig(StrictConfigModel):
    """XGBoost data configuration with training semantics.

    Canonical path uses ``source`` (a ``SourceConfig``).  Legacy flat fields
    (``type``, ``path``, ``uri``, …) are still accepted for backward
    compatibility and will be routed through the legacy adapter.
    """

    source: Any = Field(default=None, description="Canonical SourceConfig dict")
    label_col: str = Field(default="label", description="Label column name")
    feature_columns: list[str] = Field(
        default_factory=list, description="Feature columns for training"
    )
    feature_id_map: dict[str, str] = Field(default_factory=dict)
    # ── legacy flat fields ──
    type: str = Field(default="csv", description="[legacy] data source type")  # noqa: A003
    path: str | None = Field(default=None, description="[legacy] local path")
    uri: str | None = Field(default=None, description="[legacy] S3 URI")
    format: str = Field(default="parquet", description="[legacy] data format")  # noqa: A003
    s3: S3Config = Field(default_factory=S3Config, description="[legacy] S3 config")
    ch_host: str = ""
    ch_port: int = 9000
    ch_database: str = ""
    ch_user: str = "default"
    ch_password: str = ""
    ch_sql: str = ""
    ch_sql_params: dict[str, str] = Field(default_factory=dict)


class ModelConfig(StrictConfigModel):
    """XGBoost model parameters — extra fields allowed for native params."""

    model_config = ConfigDict(extra="allow")

    objective: str = Field(default="binary:logistic", description="Objective function")
    num_class: int | None = Field(
        default=None,
        ge=2,
        description="Number of multi-class categories, only needed for multi:softprob/multi:softmax",
    )


class TrainingParams(StrictConfigModel):
    """Training hyperparameters."""

    num_rounds: int = Field(default=100, ge=1, description="Number of boosting rounds")
    early_stopping_rounds: Optional[int] = Field(default=None, ge=1)
    max_rows_per_worker: Optional[int] = Field(default=None, ge=1)
    val_size: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="Validation set proportion"
    )
    test_size: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Test set proportion"
    )
    seed: int = Field(default=42, description="Random seed")


class RayConfig(StrictConfigModel):
    """Ray cluster configuration."""

    num_workers: int = Field(default=4, ge=1)
    use_gpu: bool = Field(default=False)
    storage_path: Optional[str] = None
    max_failures: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of automatic retries on worker failure. 0=no retry, -1=infinite retries. "
            "Note: checkpoint resumption is not currently supported; retries restart training from scratch."
        ),
    )


class OutputConfig(StrictConfigModel):
    """Output configuration."""

    onnx_path: Optional[str] = None
    onnx_opset: int = Field(default=12, ge=1)
    metrics_path: Optional[str] = None


class XGBoostTrainingConfig(StrictConfigModel):
    """Complete configuration for XGBoost distributed training."""

    data: XGBoostDataConfig = Field(default_factory=XGBoostDataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingParams = Field(default_factory=TrainingParams)
    ray: RayConfig = Field(default_factory=RayConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ── XGBoost BaseTrainer implementation ──


class XGBoostTrainerImpl(BaseTrainer):
    """XGBoost distributed trainer, the first implementation of BaseTrainer.

    Encapsulates XGBoostTrainingConfig validation, data preprocessing, distributed training and ONNX export.
    """

    def __init__(
        self,
        datasets: dict[str, ray.data.Dataset],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        *,
        _validated_config: XGBoostTrainingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(datasets, config, run_config, **kwargs)
        self._train_config = _validated_config or XGBoostTrainingConfig.model_validate(
            config
        )

    def setup(self) -> None:
        """Data preprocessing: filter invalid labels, split train/val/test."""
        from tributo.training.xgboost_evaluator import (
            filter_invalid_labels,
            split_dataset,
        )

        ds = self.datasets.get("train")
        if ds is None:
            raise ValueError("datasets must contain 'train' key")

        label_col = self._train_config.data.label_col
        ds = filter_invalid_labels(ds, label_col=label_col)

        feature_columns = self._train_config.data.feature_columns
        if feature_columns:
            ds = ds.select_columns(feature_columns + [label_col])

        train_ds, val_ds, test_ds = split_dataset(
            ds,
            val_size=self._train_config.training.val_size,
            test_size=self._train_config.training.test_size,
            seed=self._train_config.training.seed,
        )

        self.datasets = {"train": train_ds}
        if val_ds is not None:
            self.datasets["val"] = val_ds
        if test_ds is not None:
            self.datasets["test"] = test_ds

    def training_loop(self) -> Any:
        """Call build_trainer + trainer.fit(), returns Result."""
        cfg = self._train_config
        label_col = cfg.data.label_col

        # Build XGBoost parameters
        xgb_params = {
            k: v
            for k, v in cfg.model.model_dump(exclude={"objective"}).items()
            if not k.startswith("_")
        }
        xgb_params["objective"] = cfg.model.objective

        train_loop_config: dict[str, Any] = {
            "label_col": label_col,
            "xgb_params": xgb_params,
            "num_rounds": cfg.training.num_rounds,
            "max_rows_per_worker": cfg.training.max_rows_per_worker,
        }
        val_ds = self.datasets.get("val")
        test_ds = self.datasets.get("test")
        if val_ds is not None and cfg.training.early_stopping_rounds:
            train_loop_config["early_stopping_rounds"] = (
                cfg.training.early_stopping_rounds
            )

        trainer = build_trainer(
            ray_dataset=self.datasets["train"],
            train_config=train_loop_config,
            val_dataset=val_ds,
            test_dataset=test_ds,
            num_workers=cfg.ray.num_workers,
            use_gpu=cfg.ray.use_gpu,
            storage_path=cfg.ray.storage_path,
            max_failures=cfg.ray.max_failures,
        )

        logger.info("Starting XGBoost training...")
        result = trainer.fit()
        metrics = result.metrics or {}
        logger.info(
            "Training done: n_features=%s, %s",
            metrics.get("n_features", "?"),
            {
                k: f"{float(v):.4f}"
                for k, v in metrics.items()
                if k != "n_features" and not isinstance(v, list)
            },
        )
        return result

    def export_model(self, checkpoint: Any, output_path: str) -> None:
        """Export ONNX to local or S3, save metrics. Results written to self._summary.

        Args:
            checkpoint: Ray Train Result object returned by training_loop().
            output_path: ONNX export path, takes precedence over onnx_path in config.
        """
        from tributo.training.xgboost_evaluator import compute_metrics_summary
        from tributo.training.xgboost_exporter import export_onnx, save_metrics

        result = checkpoint
        cfg = self._train_config
        metrics = result.metrics or {}
        s3_cfg_dict = cfg.data.s3.model_dump()

        # Export ONNX (failure does not interrupt training flow)
        onnx_path = output_path or cfg.output.onnx_path
        onnx_hash = ""
        onnx_size = 0
        if onnx_path:
            try:
                n_features = int(metrics.get("n_features", 0))
                if result.checkpoint is None:
                    raise RuntimeError(
                        "Training result has no checkpoint, cannot export ONNX"
                    )
                onnx_path, onnx_hash, onnx_size = export_onnx(
                    checkpoint=result.checkpoint,
                    onnx_path=onnx_path,
                    n_features=n_features,
                    onnx_opset=cfg.output.onnx_opset,
                    s3_cfg=s3_cfg_dict,
                )
            except Exception:
                logger.exception("ONNX export failed, continuing without ONNX")
                onnx_path = None

        # Save metrics
        feature_columns = [
            c for c in self.datasets["train"].schema().names if c != cfg.data.label_col
        ]
        feature_id_map = cfg.data.model_dump().get("feature_id_map", {})
        summary = {
            "onnx_path": onnx_path,
            "onnx_hash": onnx_hash,
            "onnx_size": onnx_size,
            "feature_columns": feature_columns,
            "feature_id_map": feature_id_map,
            "metrics": compute_metrics_summary(metrics),
        }

        metrics_path = cfg.output.metrics_path
        if metrics_path:
            save_metrics(summary, metrics_path, s3_cfg=s3_cfg_dict)

        # Write to self._summary for run() return
        self._summary.update(summary)


# ── Training loop (Ray worker level) ──


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """XGBoost training main loop executed by each Ray worker.

    Data flow:
    1. Pull the current worker's data partition from the Ray Dataset shard;
    2. Build QuantileDMatrix incrementally via iter_batches (no need to hold all data);
    3. Distributed training with Rabit Allreduce for gradient histogram synchronization across workers;
    4. Rank 0 exports ONNX and reports the path via ray.train.report.

    Config keys:
        label_col (str): Label column name, default "label".
        xgb_params (dict): XGBoost training parameters.
        num_rounds (int): Number of boosting rounds, default 100.
        max_rows_per_worker (int | None): Maximum rows per worker, safety valve.
    """
    import ray.train
    import xgboost

    label_col = config.get("label_col", "label")
    max_rows = config.get("max_rows_per_worker")
    _feature_names: dict[str, list[str]] = {}

    # Cancel signal — inject CancellationChecker via config when using a broker.
    # TODO(v1.1): _tributo_cancel_key and _tributo_cancel_checker are dead code
    # until a broker implementation (e.g. tributo-broker-redis) populates them.
    _cancel_key: str | None = config.get("_tributo_cancel_key")
    _cancel_checker: CancellationChecker | None = config.get("_tributo_cancel_checker")

    class _CancelCallback(xgboost.callback.TrainingCallback):
        """Check cancellation signal after each iteration (broker protocol)."""

        def after_iteration(
            self, model: xgboost.Booster, epoch: int, evts_log: dict
        ) -> bool:
            if _cancel_key is None or _cancel_checker is None:
                return False
            try:
                return _cancel_checker.is_cancelled(_cancel_key)
            except Exception:
                logger.warning(
                    "Cancellation check failed for job %s",
                    _cancel_key,
                    exc_info=True,
                )
                return False  # transient error → don't cancel

    def _make_quantile_dmatrix(
        dataset_key: str,
        ref: xgboost.QuantileDMatrix | None = None,
    ) -> xgboost.QuantileDMatrix | None:
        """Build QuantileDMatrix streamingly from a Ray Dataset shard."""
        try:
            shard = ray.train.get_dataset_shard(dataset_key)
        except KeyError:
            return None
        if shard is None:
            return None

        batches = []
        rows_seen = 0
        for batch in shard.iter_batches(batch_format="pyarrow", batch_size=4096):
            if max_rows is not None and rows_seen >= max_rows:
                break
            if max_rows is not None:
                remaining = max_rows - rows_seen
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            batches.append(batch)
            rows_seen += batch.num_rows

        import pyarrow as pa

        table = pa.concat_tables(batches)
        del batches

        label = table.column(label_col).to_numpy()
        features = table.drop(label_col)

        # Capture feature column names for later importance extraction
        _feature_names[dataset_key] = list(features.schema.names)

        del table

        return xgboost.QuantileDMatrix(features, label=label, ref=ref)

    dtrain = _make_quantile_dmatrix("train")
    if dtrain is None:
        raise RuntimeError("Failed to build training QuantileDMatrix")
    n_features = dtrain.num_col()

    evals = [(dtrain, "train")]
    dval = _make_quantile_dmatrix("val", ref=dtrain)
    if dval is not None:
        evals.append((dval, "val"))

    evals_result: dict[str, Any] = {}
    booster = xgboost.train(
        config.get("xgb_params", {}),
        dtrain,
        num_boost_round=int(config.get("num_rounds", 100)),
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=config.get("early_stopping_rounds"),
        callbacks=[_CancelCallback()],
    )

    world_rank = ray.train.get_context().get_world_rank()

    # Build test DMatrix on *all* ranks so Ray Train doesn't hang
    # get_dataset_shard splits data across workers; every rank must
    # consume its partition even if only rank 0 does sklearn evaluation.
    dtest = _make_quantile_dmatrix("test", ref=dtrain)
    test_labels: list[Any] = []
    if dtest is not None:
        test_shard = ray.train.get_dataset_shard("test")
        rows_seen = 0
        for batch in test_shard.iter_batches(batch_format="pyarrow", batch_size=4096):
            if max_rows is not None and rows_seen >= max_rows:
                break
            if max_rows is not None:
                remaining = max_rows - rows_seen
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            test_labels.append(batch.column(label_col).to_pylist())
            rows_seen += batch.num_rows

    # Row counts per split (all ranks participate)
    row_info = {}
    for key in ("train", "val", "test"):
        try:
            shard = ray.train.get_dataset_shard(key)
            total = 0
            for batch in shard.iter_batches(batch_format="pyarrow", batch_size=4096):
                total += batch.num_rows
            row_info[f"row_count_{key}"] = total
        except Exception:
            pass  # split not available; will be filled by rank 0 below

    if world_rank == 0:
        from ray.train.xgboost import XGBoostCheckpoint

        checkpoint = XGBoostCheckpoint.from_model(booster)

        report_metrics: dict[str, Any] = {"n_features": n_features}
        for eval_name, eval_scores in evals_result.items():
            for metric_name, values in eval_scores.items():
                key = f"{eval_name}-{metric_name}"
                report_metrics[key] = values[-1]
                report_metrics[f"{key}_history"] = values

        # Worker-side sklearn evaluation
        if dtest is not None and test_labels:
            try:
                import numpy as np
                from sklearn.metrics import (
                    average_precision_score,
                    confusion_matrix,
                    f1_score,
                    precision_score,
                    recall_score,
                    roc_auc_score,
                    roc_curve,
                )

                y_true = np.array([v for chunk in test_labels for v in chunk])
                xgb_params = config.get("xgb_params", {})
                is_multiclass = xgb_params.get("objective", "").startswith("multi:")

                if is_multiclass:
                    # Multi-class evaluation
                    y_prob = booster.predict(dtest)  # (N, K)
                    y_pred = y_prob.argmax(axis=1)

                    auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
                    f1 = float(
                        f1_score(y_true, y_pred, average="macro", zero_division=0)
                    )
                    prec = float(
                        precision_score(
                            y_true, y_pred, average="macro", zero_division=0
                        )
                    )
                    rec = float(
                        recall_score(y_true, y_pred, average="macro", zero_division=0)
                    )
                    cm = confusion_matrix(y_true, y_pred).tolist()
                    # Collect class labels for the MultiClassConfusionMatrix
                    # Protocol §3.4.2 expects labels as strings.
                    labels = sorted(str(v) for v in np.unique(y_true).tolist())

                    report_metrics.update(
                        {
                            "eval_auc": auc,
                            "eval_f1_macro": f1,
                            "eval_precision_macro": prec,
                            "eval_recall_macro": rec,
                            "eval_test_rows": int(len(y_true)),
                            "eval_cm": cm,
                            "eval_cm_labels": labels,
                        }
                    )
                else:
                    # Binary evaluation (original)
                    y_prob = booster.predict(dtest)  # (N,)
                    y_pred = (y_prob >= 0.5).astype(int)

                    auc = float(roc_auc_score(y_true, y_prob))
                    f1 = float(f1_score(y_true, y_pred, zero_division=0))
                    prec = float(precision_score(y_true, y_pred, zero_division=0))
                    rec = float(recall_score(y_true, y_pred, zero_division=0))
                    ap = float(average_precision_score(y_true, y_prob))

                    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
                    step = max(1, len(fpr_arr) // 100)
                    fpr_ds = fpr_arr[::step].tolist()
                    tpr_ds = tpr_arr[::step].tolist()

                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tn = fp = fn = tp = 0

                    thresholds = [
                        round(t, 2) for t in np.linspace(0.05, 0.95, 19).tolist()
                    ]
                    thr_prec, thr_rec, thr_f1, thr_pp = [], [], [], []
                    for thr in thresholds:
                        yp = (y_prob >= thr).astype(int)
                        thr_prec.append(
                            round(
                                float(precision_score(y_true, yp, zero_division=0)), 4
                            )
                        )
                        thr_rec.append(
                            round(float(recall_score(y_true, yp, zero_division=0)), 4)
                        )
                        thr_f1.append(
                            round(float(f1_score(y_true, yp, zero_division=0)), 4)
                        )
                        thr_pp.append(int(yp.sum()))

                    report_metrics.update(
                        {
                            "eval_auc": auc,
                            "eval_f1": f1,
                            "eval_precision": prec,
                            "eval_recall": rec,
                            "eval_avg_precision": ap,
                            "eval_test_rows": int(len(y_true)),
                            "eval_cm_tp": int(tp),
                            "eval_cm_fp": int(fp),
                            "eval_cm_fn": int(fn),
                            "eval_cm_tn": int(tn),
                            "eval_roc_fpr": fpr_ds,
                            "eval_roc_tpr": tpr_ds,
                            "eval_thr_thresholds": thresholds,
                            "eval_thr_precision": thr_prec,
                            "eval_thr_recall": thr_rec,
                            "eval_thr_f1": thr_f1,
                            "eval_thr_predicted_positive": thr_pp,
                        }
                    )
            except Exception:
                import logging as _log

                _log.getLogger(__name__).exception(
                    "sklearn evaluation failed — training result unaffected"
                )

        # Feature importance (XGBoost gain)
        try:
            score = booster.get_score(importance_type="gain")
            # score keys are the DMatrix column names (e.g. "feature_0")
            rank = 1
            for feat, gain in sorted(
                score.items(), key=lambda x: -_gain_as_float(x[1])
            ):
                report_metrics.setdefault("feat_imp_rank", []).append(rank)
                report_metrics.setdefault("feat_imp_name", []).append(feat)
                report_metrics.setdefault("feat_imp_score", []).append(
                    round(_gain_as_float(gain), 6)
                )
                rank += 1
        except Exception:
            import logging as _log

            _log.getLogger(__name__).exception("feature importance extraction failed")

        report_metrics.update(row_info)

        ray.train.report(report_metrics, checkpoint=checkpoint)
    else:
        ray.train.report({})


# ── Trainer construction ──


@PublicAPI(stability="beta")
def build_trainer(
    ray_dataset: "ray.data.Dataset",
    train_config: dict[str, Any],
    *,
    val_dataset: "ray.data.Dataset | None" = None,
    test_dataset: "ray.data.Dataset | None" = None,
    num_workers: int = 4,
    use_gpu: bool = False,
    storage_path: str | None = None,
    max_failures: int = 0,
) -> "XGBoostTrainer":
    """Build a Ray XGBoostTrainer instance.

    Args:
        ray_dataset: Training dataset.
        train_config: Config dictionary passed to ``train_loop_per_worker``.
        val_dataset: Validation dataset, enables early stopping when provided.
        test_dataset: Test dataset, enables sklearn evaluation on the worker side.
        num_workers: Number of training workers.
        use_gpu: Whether to use GPU.
        storage_path: Ray Train persistent storage path.
        max_failures: Number of automatic retries on worker failure. 0=no retry, -1=infinite retries.
            Checkpoint resumption is not currently supported; retries restart training from scratch.

    Returns:
        An unstarted XGBoostTrainer; call ``.fit()`` to begin training.
    """
    import os
    import tempfile

    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    datasets: dict[str, "ray.data.Dataset"] = {"train": ray_dataset}
    if val_dataset is not None:
        datasets["val"] = val_dataset
    if test_dataset is not None:
        datasets["test"] = test_dataset

    if storage_path is not None:
        storage = storage_path
    else:
        # Auto-detect: prefer Docker shared volume, then /workspace, then local temp
        if os.path.isdir("/app/.ray_results"):
            storage = "/app/.ray_results"
        elif os.path.isdir("/workspace"):
            storage = "/workspace/ray_results"
        else:
            storage = os.path.join(tempfile.gettempdir(), "ray_results")

    return XGBoostTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            placement_strategy="SPREAD",
        ),
        datasets=datasets,  # type: ignore[arg-type]
        run_config=RunConfig(
            name="tributo-xgboost",
            storage_path=storage,
            # resilience-only: retries restart from scratch, not from checkpoint
            failure_config=FailureConfig(max_failures=max_failures),
        ),
    )


# ── Orchestration entry points ──


def run_training_with_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run XGBoost distributed training from a YAML config dictionary.

    YAML structure::

        data:
          type: csv | s3 | clickhouse
          path: /data/train.parquet       # csv type
          uri: s3://bucket/data.parquet   # s3 type
          format: csv | parquet
          s3: {region, access_key_id, secret_access_key, endpoint}
          label_col: label

        model:
          objective: binary:logistic      # binary classification
          # objective: multi:softprob     # multi-class (num_class required)
          # num_class: 3
          max_depth: 6
          eta: 0.3
          subsample: 1.0
          colsample_bytree: 1.0

        training:
          num_rounds: 100
          early_stopping_rounds: 10
          max_rows_per_worker: null
          val_size: 0.2
          seed: 42

        ray:
          num_workers: 4
          use_gpu: false

        output:
          onnx_path: model.onnx
          onnx_opset: 12
          metrics_path: metrics.json

    Returns:
        ``{"onnx_path": ..., "metrics": ..., "feature_columns": [...]}``
    """
    from tributo.training.data_loader import load_ray_dataset_from_config

    # Pydantic validation
    cfg = XGBoostTrainingConfig.model_validate(config)

    # Load data
    logger.info("Loading data (type=%s)...", cfg.data.type)
    ds = load_ray_dataset_from_config(cfg.data.model_dump())

    # Execute training using XGBoostTrainerImpl
    trainer = XGBoostTrainerImpl(
        datasets={"train": ds},
        config=config,
        _validated_config=cfg,
    )
    output_path = cfg.output.onnx_path or ""
    return trainer.run(output_path=output_path)


# Built-in registration

_trainer_spec = AlgorithmSpec(
    name="xgboost",
    trainer_cls=XGBoostTrainerImpl,
    problem_types=(
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTI_CLASS_CLASSIFICATION,
        ProblemType.REGRESSION,
    ),
    data_modality=("tabular",),
    extras_group="training",
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    resource_hints=ResourceHints(gpu_required=False),
    config_model=XGBoostTrainingConfig,
)
register(_trainer_spec)

# Exported for entry_points discovery (see tributo.plugin)
trainer_spec = _trainer_spec


@PublicAPI(stability="beta")
def run_training_from_json(config_path: str) -> dict[str, Any]:
    """Run XGBoost distributed training from a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Summary dictionary.
    """
    import json
    from pathlib import Path

    path = Path(config_path)
    if path.suffix in {".yaml", ".yml"}:
        raise ValueError(
            "YAML training config is no longer supported; please use JSON."
        )
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return run_training_with_config(loaded)
