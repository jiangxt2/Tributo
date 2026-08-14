"""XGBoost distributed training entry point, based on Ray Train XGBoostTrainer."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import AliasChoices, ConfigDict, Field, model_validator

from tributo._common.config import StrictConfigModel
from tributo.data.base import S3Config
from tributo.exceptions import JobConfigurationError
from tributo.explainability.contracts import ExplainabilityConfig
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    XGBOOST_DESCRIPTOR,
    build_legacy_spec,
)
from tributo.integrations.broker import CancellationChecker
from tributo.training.base import BaseTrainer
from tributo.training.checkpoint import ResumeConfig
from tributo.training.resource import (
    DEFAULT_BATCH_SIZE,
    BoundedCollector,
    ResourceBudget,
    estimate_row_bytes_from_schema,
    preflight_check,
)
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data
    from ray.train.xgboost import XGBoostTrainer

# XGBoost params reserved by Tributo: silently passing these as native
# training parameters would change the execution path (e.g. external-memory
# / data_iter) without going through the materialization-budget contract.
_RESERVED_XGB_PARAMS = frozenset({"external_memory", "data_iter"})


def _populate_xgb_eval_metrics(
    metrics: dict[str, Any],
    evals_result: dict[str, dict[str, list[Any]]],
) -> None:
    """Add current and history keys using one stable evaluation schema."""
    for eval_name, eval_scores in evals_result.items():
        for metric_name, values in eval_scores.items():
            if values:
                key = f"{eval_name}-{metric_name}"
                metrics[key] = values[-1]
                metrics[f"{key}_history"] = list(values)


def _merge_xgb_eval_results(
    previous: dict[str, dict[str, list[Any]]],
    current: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, list[Any]]]:
    """Merge resumed and current XGBoost evaluation histories."""
    merged = {
        eval_name: {
            metric_name: list(values) for metric_name, values in eval_scores.items()
        }
        for eval_name, eval_scores in previous.items()
    }
    for eval_name, eval_scores in current.items():
        target = merged.setdefault(eval_name, {})
        for metric_name, values in eval_scores.items():
            prior_values = target.get(metric_name, [])
            if values[: len(prior_values)] == prior_values:
                target[metric_name] = list(values)
            else:
                target[metric_name] = list(prior_values) + list(values)
    return merged


logger = logging.getLogger(__name__)


@contextmanager
def _managed_resume_checkpoint(
    checkpoint_data: tuple[Any, str, Path],
) -> Iterator[tuple[Any, str]]:
    """Keep a temporary resume checkpoint alive only through its report call."""
    checkpoint, checkpoint_id, checkpoint_dir = checkpoint_data
    try:
        yield checkpoint, checkpoint_id
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


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
    """XGBoost model parameters — extra fields allowed for native params.

    Reserved keys (``external_memory``, ``data_iter``) are rejected: they
    would silently switch the training path away from the budgeted
    ``QuantileDMatrix`` route.
    """

    model_config = ConfigDict(extra="allow")

    objective: str = Field(default="binary:logistic", description="Objective function")
    num_class: int | None = Field(
        default=None,
        ge=2,
        description="Number of multi-class categories, only needed for multi:softprob/multi:softmax",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_reserved_params(cls, data: Any) -> Any:
        """Structured fail-fast for reserved XGBoost parameters."""
        if isinstance(data, dict):
            for key in _RESERVED_XGB_PARAMS:
                if key in data:
                    raise ValueError(
                        f"XGBoost parameter {key!r} is reserved by Tributo: "
                        "external-memory is not supported on the default "
                        "QuantileDMatrix path; remove it from the model config."
                    )
        return data


class TrainingParams(StrictConfigModel):
    """Training hyperparameters."""

    num_rounds: int = Field(default=100, ge=1, description="Number of boosting rounds")
    early_stopping_rounds: Optional[int] = Field(default=None, ge=1)
    max_rows_per_worker: Optional[int] = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "max_rows_per_worker", "max_input_rows_per_worker"
        ),
        description=(
            "Row-count guard per worker: exceeding it fails fast, "
            "training data is never silently truncated.  "
            "Alias: max_input_rows_per_worker."
        ),
    )
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
            "When resume.enabled is true, retries restore the latest retained checkpoint."
        ),
    )
    resume: ResumeConfig = Field(default_factory=ResumeConfig)


class OutputConfig(StrictConfigModel):
    """Output configuration."""

    bundle_uri: Optional[str] = None
    onnx_path: Optional[str] = None
    onnx_opset: int = Field(default=12, ge=1)
    metrics_path: Optional[str] = None
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)

    @model_validator(mode="after")
    def validate_destination_contract(self) -> OutputConfig:
        legacy_fields = {"onnx_path", "onnx_opset", "metrics_path"}
        configured_legacy = sorted(legacy_fields & self.model_fields_set)
        if self.bundle_uri is not None and configured_legacy:
            raise ValueError(
                "output.bundle_uri cannot be combined with legacy output fields: "
                + ", ".join(configured_legacy)
            )
        if self.explainability.enabled and self.bundle_uri is None:
            raise ValueError("output.explainability.enabled requires output.bundle_uri")
        return self


class XGBoostTrainingConfig(StrictConfigModel):
    """Complete configuration for XGBoost distributed training."""

    data: XGBoostDataConfig = Field(default_factory=XGBoostDataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingParams = Field(default_factory=TrainingParams)
    ray: RayConfig = Field(default_factory=RayConfig)
    resource: ResourceBudget = Field(
        default_factory=ResourceBudget,
        description="Single-worker materialization budget",
    )
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
        from tributo.training.checkpoint import load_initial_checkpoint

        cfg = self._train_config
        label_col = cfg.data.label_col

        if cfg.ray.resume.effective_enabled and cfg.ray.num_workers != 1:
            raise JobConfigurationError(
                "T4-A resume currently supports num_workers=1 only; "
                "multi-worker checkpoint coordination is deferred to T4-D."
            )

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
            "resource": cfg.resource.model_dump(),
            "resume": cfg.ray.resume.model_dump(),
        }
        val_ds = self.datasets.get("val")
        test_ds = self.datasets.get("test")
        if val_ds is not None and cfg.training.early_stopping_rounds:
            train_loop_config["early_stopping_rounds"] = (
                cfg.training.early_stopping_rounds
            )

        storage_path = self.run_config.get("storage_path")
        if storage_path is None:
            storage_path = cfg.ray.storage_path
        elif not isinstance(storage_path, str) or not storage_path:
            raise JobConfigurationError(
                "run_config 'storage_path' must be a non-empty string or None"
            )
        run_name = self.run_config.get("name")
        if run_name is None:
            run_name = "tributo-xgboost"
        elif not isinstance(run_name, str) or not run_name:
            raise JobConfigurationError(
                "run_config 'name' must be a non-empty string or None"
            )

        trainer = _build_trainer(
            ray_dataset=self.datasets["train"],
            train_config=train_loop_config,
            val_dataset=val_ds,
            test_dataset=test_ds,
            num_workers=cfg.ray.num_workers,
            use_gpu=cfg.ray.use_gpu,
            storage_path=storage_path,
            max_failures=cfg.ray.max_failures,
            resume_from_checkpoint=load_initial_checkpoint(
                cfg.ray.resume.checkpoint_path
            ),
            run_name=run_name,
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

        # Export ONNX.  Once a path is configured the model is a required
        # artifact: a failed export must fail the run (ADR-001), never
        # produce a "succeeded" run without the model.
        onnx_path = output_path or cfg.output.onnx_path
        onnx_hash = ""
        onnx_size = 0
        if onnx_path:
            n_features = int(metrics.get("n_features", 0))
            if result.checkpoint is None:
                raise RuntimeError(
                    "Training result has no checkpoint, cannot export ONNX"
                )
            try:
                onnx_path, onnx_hash, onnx_size = export_onnx(
                    checkpoint=result.checkpoint,
                    onnx_path=onnx_path,
                    n_features=n_features,
                    onnx_opset=cfg.output.onnx_opset,
                    s3_cfg=s3_cfg_dict,
                )
            except Exception:
                logger.exception(
                    "ONNX export failed for required artifact: %s", onnx_path
                )
                raise

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

    @staticmethod
    def _get_trainer_type() -> str:
        """Return the explicit Bundle source-provider identity."""
        return "xgboost"

    @staticmethod
    def _default_bundle_targets() -> tuple[Any, ...]:
        from tributo.exporting.models import ExportTarget

        return (
            ExportTarget(name="onnx-model", format="onnx", options={"opset": 12}),
            ExportTarget(name="native", format="ubj"),
        )

    @staticmethod
    def _default_bundle_roles() -> dict[str, str]:
        return {"inference": "onnx-model"}


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
        max_rows_per_worker (int | None): Row-count guard — exceeding
            it fails fast; training data is never silently truncated.
        resource (dict): Materialization budget; defaults always
            apply when absent.
    """
    import ray.train
    import xgboost

    from tributo.training.checkpoint import (
        checkpoint_directory,
        read_resume_manifest,
        write_resume_manifest,
    )

    label_col = config.get("label_col", "label")
    resume_cfg = ResumeConfig.model_validate(config.get("resume") or {})
    resume_enabled = resume_cfg.effective_enabled
    checkpoint_interval = resume_cfg.checkpoint_interval
    num_rounds = int(config.get("num_rounds", 100))
    if resume_enabled and ray.train.get_context().get_world_size() != 1:
        raise JobConfigurationError(
            "T4-A resume currently supports num_workers=1 only; "
            "multi-worker checkpoint coordination is deferred to T4-D."
        )
    max_rows = config.get("max_rows_per_worker")
    _feature_names: dict[str, list[str]] = {}
    _rows_seen: dict[str, int] = {}

    # One worker materialization budget shared by all splits —
    # train/val/test matrices and the test label lists can all be alive at
    # the same time.  Over-budget inputs fail before the unbounded
    # concat_tables; rows are never truncated.
    budget = ResourceBudget.model_validate(config.get("resource") or {})
    worker_rank = ray.train.get_context().get_world_rank()
    collector = BoundedCollector(
        budget,
        algorithm="xgboost",
        split="train",
        worker_rank=worker_rank,
        max_rows=max_rows,
    )

    # Test labels collected on the *first* pass over the test shard (inside
    # _make_quantile_dmatrix) so the shard is never iterated twice — a second
    # pass would double-count bytes and rows against the shared budget.
    test_labels: list[Any] = []

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
        *,
        collect_labels: bool = False,
    ) -> xgboost.QuantileDMatrix | None:
        """Build QuantileDMatrix streamingly from a Ray Dataset shard.

        Collects the shard under the shared materialization budget.  If
        ``max_rows`` / ``max_input_rows_per_worker`` is configured and the
        shard exceeds it, collection fails fast — the shard is never
        silently truncated.  With ``collect_labels`` the label column is
        also captured during this single pass (used for the test split, so
        the shard is never iterated twice).
        """
        try:
            shard = ray.train.get_dataset_shard(dataset_key)
        except KeyError:
            return None
        if shard is None:
            return None

        # Phase-1 preflight: schema-level estimate (cheap, no row count
        # required); the in-flight collector remains the hard guarantee.
        row_bytes = None
        schema = getattr(shard, "schema", None)
        if callable(schema):
            try:
                row_bytes = estimate_row_bytes_from_schema(schema())
            except Exception:
                row_bytes = None
        preflight_check(
            rows=None,
            row_bytes=row_bytes,
            budget=budget,
            algorithm="xgboost",
            split=dataset_key,
            worker_rank=worker_rank,
        )

        batches = []
        split_rows = 0
        for batch in shard.iter_batches(
            batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
        ):
            collector.add(batch, split=dataset_key)
            batches.append(batch)
            split_rows += batch.num_rows
            if collect_labels:
                # Rank-0 evaluation labels: compact numpy arrays (not Python
                # pylists — those can exceed the Arrow buffer severalfold),
                # accounted against the shared budget.
                label_array = batch.column(label_col).to_numpy()
                collector.add_bytes(label_array.nbytes, split=dataset_key)
                test_labels.append(label_array)
        _rows_seen[dataset_key] = split_rows

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

    start_round = 0
    initial_booster: xgboost.Booster | None = None
    resume_evals_result: dict[str, Any] = {}
    resume_checkpoint = ray.train.get_checkpoint() if resume_enabled else None
    if resume_checkpoint is not None:
        with checkpoint_directory(resume_checkpoint) as checkpoint_dir:
            envelope = read_resume_manifest(
                checkpoint_dir,
                expected_trainer_type="xgboost",
                expected_resume_id=resume_cfg.resume_id,
            )
            initial_booster = xgboost.Booster()
            initial_booster.load_model(str(checkpoint_dir / "model.json"))
            training_state = json.loads(
                (checkpoint_dir / "training_state.json").read_text()
            )
        resume_evals_result = training_state.get("evals_result") or {}
        checkpoint_params = training_state.get("xgb_params")
        if checkpoint_params is not None and checkpoint_params != config.get(
            "xgb_params", {}
        ):
            raise ValueError(
                "Resume checkpoint XGBoost parameters do not match the current run"
            )
        start_round = int(
            training_state.get("completed_rounds", envelope.completed_step)
        )
        logger.info(
            "Resuming XGBoost training from %s at round %d",
            envelope.resume_id,
            start_round,
        )

    def _best_iteration(model: xgboost.Booster) -> int | None:
        try:
            return int(model.best_iteration)
        except (AttributeError, TypeError, ValueError):
            return None

    def _best_score(model: xgboost.Booster) -> float | None:
        try:
            return float(model.best_score)
        except (AttributeError, TypeError, ValueError):
            return None

    def _write_resume_checkpoint(
        model: xgboost.Booster,
        completed_rounds: int,
        evals_result: dict[str, dict[str, list[Any]]] | None = None,
    ) -> tuple[Any, str, Path]:
        """Persist a framework-neutral resume envelope for one XGBoost round."""
        from ray.train import Checkpoint

        checkpoint_dir = Path(tempfile.mkdtemp(prefix="xgb_resume_ckpt_"))
        try:
            model.save_model(str(checkpoint_dir / "model.json"))
            training_state = {
                "completed_rounds": completed_rounds,
                "best_iteration": _best_iteration(model),
                "best_score": _best_score(model),
                "xgb_params": config.get("xgb_params", {}),
                "evals_result": evals_result or {},
            }
            (checkpoint_dir / "training_state.json").write_text(
                json.dumps(training_state, ensure_ascii=False, default=float)
            )
            envelope = write_resume_manifest(
                checkpoint_dir,
                resume_id=resume_cfg.resume_id,
                trainer_type="xgboost",
                completed_step=completed_rounds,
                framework="xgboost",
                framework_version=xgboost.__version__,
                payload_files=("model.json", "training_state.json"),
                payload_metadata={
                    "model": "model.json",
                    "completed_rounds": completed_rounds,
                    "best_iteration": training_state["best_iteration"],
                    "best_score": training_state["best_score"],
                    "xgb_params": training_state["xgb_params"],
                    "preprocessing": "QuantileDMatrix",
                },
            )
            return (
                Checkpoint.from_directory(str(checkpoint_dir)),
                envelope.resume_id,
                checkpoint_dir,
            )
        except BaseException:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            raise

    def _report_resume_checkpoint(
        model: xgboost.Booster,
        completed_rounds: int,
        evals_log: dict[str, dict[str, list[Any]]],
    ) -> None:
        """Report one synchronized periodic checkpoint across all workers."""
        full_evals_log = _merge_xgb_eval_results(resume_evals_result, evals_log)
        metrics: dict[str, Any] = {"epoch": completed_rounds}
        _populate_xgb_eval_metrics(metrics, full_evals_log)
        if worker_rank == 0:
            with _managed_resume_checkpoint(
                _write_resume_checkpoint(model, completed_rounds, full_evals_log)
            ) as (checkpoint, checkpoint_id):
                metrics["resume_id"] = checkpoint_id
                ray.train.report(metrics, checkpoint=checkpoint)
        else:
            ray.train.report(metrics)

    class _ResumeCheckpointCallback(xgboost.callback.TrainingCallback):
        """Persist resumable state at the configured boosting interval."""

        def after_iteration(
            self,
            model: xgboost.Booster,
            epoch: int,
            evals_log: dict[str, dict[str, list[Any]]],
        ) -> bool:
            if not resume_enabled:
                return False
            completed_rounds = start_round + epoch + 1
            if (
                completed_rounds % checkpoint_interval != 0
                and completed_rounds != num_rounds
            ):
                return False
            _report_resume_checkpoint(model, completed_rounds, evals_log)
            return False

    evals_result: dict[str, dict[str, list[Any]]] = {}
    if initial_booster is not None and start_round >= num_rounds:
        booster = initial_booster
        evals_result = _merge_xgb_eval_results(resume_evals_result, {})
    else:
        current_evals_result: dict[str, dict[str, list[Any]]] = {}
        booster = xgboost.train(
            config.get("xgb_params", {}),
            dtrain,
            num_boost_round=num_rounds - start_round,
            evals=evals,
            evals_result=current_evals_result,
            early_stopping_rounds=config.get("early_stopping_rounds"),
            callbacks=[_CancelCallback(), _ResumeCheckpointCallback()],
            xgb_model=initial_booster,
        )
        evals_result = _merge_xgb_eval_results(
            resume_evals_result, current_evals_result
        )

    world_rank = ray.train.get_context().get_world_rank()

    # Build test DMatrix on *all* ranks so Ray Train doesn't hang
    # get_dataset_shard splits data across workers; every rank must
    # consume its partition even if only rank 0 does sklearn evaluation.
    # Labels are captured during this single pass, on rank 0 only, as
    # compact numpy arrays accounted against the shared budget —
    # the shard is never iterated twice, so test bytes/rows are counted
    # once per worker.
    dtest = _make_quantile_dmatrix("test", ref=dtrain, collect_labels=world_rank == 0)

    # Row counts per split, collected once during DMatrix construction
    # (no second full shard iteration)
    row_info = {f"row_count_{key}": rows for key, rows in _rows_seen.items()}

    if world_rank == 0:
        from ray.train.xgboost import XGBoostCheckpoint

        report_metrics: dict[str, Any] = {"n_features": n_features}
        _populate_xgb_eval_metrics(report_metrics, evals_result)

        # Worker-side sklearn evaluation
        if dtest is not None and test_labels:
            # test_labels are rank-0 numpy arrays (see collect_labels).
            # concat 输出副本与源数组并存 → 评估峰值 = 2 × labels 总量；
            # 源数组已在收集阶段逐批记账（add_bytes），此处补记 concat
            # 副本（1×）。检查发生在 concat 之前——超限则 fail-fast，
            # 不分配副本。记账独立于 try 块：评估库（sklearn）缺失时
            # 预算契约仍然生效。
            concat_copy_bytes = sum(a.nbytes for a in test_labels)
            collector.add_bytes(concat_copy_bytes, split="test")

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

                y_true = np.concatenate(test_labels)
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
        if resume_enabled:
            completed_rounds = booster.num_boosted_rounds()
            with _managed_resume_checkpoint(
                _write_resume_checkpoint(booster, completed_rounds, evals_result)
            ) as (checkpoint, checkpoint_id):
                report_metrics["resume_id"] = checkpoint_id
                ray.train.report(report_metrics, checkpoint=checkpoint)
        else:
            checkpoint = XGBoostCheckpoint.from_model(booster)
            ray.train.report(report_metrics, checkpoint=checkpoint)
    else:
        ray.train.report({})


# ── Trainer construction ──


def _build_trainer(
    ray_dataset: "ray.data.Dataset",
    train_config: dict[str, Any],
    *,
    val_dataset: "ray.data.Dataset | None" = None,
    test_dataset: "ray.data.Dataset | None" = None,
    num_workers: int = 4,
    use_gpu: bool = False,
    storage_path: str | None = None,
    max_failures: int = 0,
    resume_from_checkpoint: Any | None = None,
    run_name: str,
) -> "XGBoostTrainer":
    """Build a Ray XGBoostTrainer with an explicit internal run identity.

    Args:
        ray_dataset: Training dataset.
        train_config: Config dictionary passed to ``train_loop_per_worker``.
        val_dataset: Validation dataset, enables early stopping when provided.
        test_dataset: Test dataset, enables sklearn evaluation on the worker side.
        num_workers: Number of training workers.
        use_gpu: Whether to use GPU.
        storage_path: Ray Train persistent storage path.
        max_failures: Number of automatic retries on worker failure. 0=no retry, -1=infinite retries.
            When resume is enabled, retries restore the latest retained checkpoint.
        resume_from_checkpoint: Optional initial Ray checkpoint to restore.

    Returns:
        An unstarted XGBoostTrainer; call ``.fit()`` to begin training.

    Raises:
        JobConfigurationError: ``train_config`` contains a reserved XGBoost
            parameter (``external_memory``/``data_iter``) — the raw dict
            entry point enforces the same contract as ``ModelConfig``
            (the raw dict entry point enforces the same contract as ``ModelConfig``).
    """
    # The raw train_config entry point bypasses ModelConfig's
    # reserved-key rejection — enforce the same fail-fast contract here
    # (external_memory/data_iter would silently switch the execution path
    # away from the budgeted QuantileDMatrix route).
    reserved = _RESERVED_XGB_PARAMS & set(train_config.get("xgb_params", {}))
    if reserved:
        raise JobConfigurationError(
            f"XGBoost parameter {sorted(reserved)[0]!r} is reserved by Tributo: "
            "external-memory is not supported on the default "
            "QuantileDMatrix path; remove it from train_config['xgb_params']."
        )

    import os
    import tempfile

    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    from tributo.training.checkpoint import checkpoint_config

    resume_config = ResumeConfig.model_validate(train_config.get("resume") or {})
    if resume_from_checkpoint is not None and not resume_config.effective_enabled:
        resume_config = ResumeConfig(enabled=True)
        train_config = {**train_config, "resume": resume_config.model_dump()}
    if resume_config.effective_enabled and num_workers != 1:
        raise JobConfigurationError(
            "T4-A resume currently supports num_workers=1 only; "
            "multi-worker checkpoint coordination is deferred to T4-D."
        )

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
            name=run_name,
            storage_path=storage,
            failure_config=FailureConfig(max_failures=max_failures),
            checkpoint_config=checkpoint_config(resume_config),
        ),
        resume_from_checkpoint=resume_from_checkpoint,
    )


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
    resume_from_checkpoint: Any | None = None,
) -> "XGBoostTrainer":
    """Build a Ray XGBoostTrainer instance with its stable public defaults.

    Args:
        ray_dataset: Training dataset.
        train_config: Config dictionary passed to ``train_loop_per_worker``.
        val_dataset: Validation dataset, enabling early stopping when provided.
        test_dataset: Test dataset, enabling worker-side evaluation when provided.
        num_workers: Number of training workers.
        use_gpu: Whether to use GPU.
        storage_path: Ray Train persistent storage path.
        max_failures: Number of automatic retries on worker failure.
        resume_from_checkpoint: Optional initial Ray checkpoint to restore.

    Returns:
        An unstarted XGBoostTrainer; call ``.fit()`` to begin training.
    """
    return _build_trainer(
        ray_dataset=ray_dataset,
        train_config=train_config,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=num_workers,
        use_gpu=use_gpu,
        storage_path=storage_path,
        max_failures=max_failures,
        resume_from_checkpoint=resume_from_checkpoint,
        run_name="tributo-xgboost",
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
          bundle_uri: s3://bucket/model-bundles

    Returns:
        A Bundle-backed training result containing ``bundle_uri`` and
        ``execution_id``.
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
    if cfg.output.bundle_uri:
        from tributo.exporting.models import BundleOutputConfig

        bundle_config = BundleOutputConfig(
            bundle_uri=cfg.output.bundle_uri,
            explainability=cfg.output.explainability,
        )
        return trainer.run(
            output_path=cfg.output.bundle_uri,
            bundle_config=bundle_config,
        )
    if cfg.output.onnx_path:
        import warnings

        warnings.warn(
            "output.onnx_path selects the deprecated legacy export path; "
            "configure output.bundle_uri for Bundle publication",
            DeprecationWarning,
            stacklevel=2,
        )
        return trainer.run(output_path=cfg.output.onnx_path, legacy_export=True)
    return trainer.run()


# Built-in registration

_trainer_spec = build_legacy_spec(
    XGBOOST_DESCRIPTOR,
    trainer_cls=XGBoostTrainerImpl,
    config_model=XGBoostTrainingConfig,
)

# Exported for the explicit Beta compatibility API.
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
