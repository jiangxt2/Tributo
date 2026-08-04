"""PU Learning distributed trainer.

Based on Ray Train TorchTrainer, focused on Positive-Unlabeled Learning scenarios.
Supports nnPU/uPU risk estimators + automatic class prior estimation + PU-specific metrics.

Differences from DNNTrainer:
- Enforces PU loss (nnPU or uPU)
- Built-in class prior estimation (label_frequency / histogram_match / em)
- Outputs PU-specific metrics during training
- Streamlined configuration for PU use cases
- Data is loaded inside workers; the driver never holds the dataset

Usage::

    from tributo.training.pu_trainer import run_pu_training_with_config

    result = run_pu_training_with_config({
        "data": {"type": "s3", "uri": "s3://bucket/data.parquet", ...},
        "features": [...],
        "pu": {"loss_type": "nnpu", "class_prior_method": "label_frequency"},
        "training": {"epochs": 20},
        "ray": {"num_workers": 1},
        "output": {"onnx_path": "/tmp/model_output"},
    })
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import Field

from tributo._common.config import StrictConfigModel
from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    DataLoadingMode,
    ProblemType,
    ResourceHints,
)
from tributo.training.base import BaseTrainer
from tributo.training.checkpoint import ResumeConfig
from tributo.training.dnn_trainer import (
    DNNModelConfig,
    DNNOutputConfig,
    DNNTrainingParams,
    FeatureItemConfig,
    build_export_checkpoint_config,
    build_features_from_config,
)
from tributo.training.registry import register
from tributo.training.resource import (
    DEFAULT_BATCH_SIZE,
    ResourceBudget,
    collect_bounded,
    estimate_row_bytes_from_schema,
    preflight_check,
)
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)


# ── PU-specific config models ──


class PUConfig(StrictConfigModel):
    """PU Learning configuration."""

    loss_type: Literal["nnpu", "upu"] = Field(
        default="nnpu",
        description="PU loss type: nnpu (non-negative constraint) | upu (unbiased estimate)",
    )
    class_prior: Optional[float] = Field(
        default=None,
        description="Positive class prior (π_p); auto-estimated when None",
    )
    class_prior_method: Literal["label_frequency", "histogram_match", "em"] = Field(
        default="label_frequency",
        description="Class prior estimation method",
    )
    beta: float = Field(
        default=0.0, description="nnPU non-negative constraint threshold"
    )
    gamma: float = Field(default=1.0, description="Negative risk scaling factor")


class PURayConfig(StrictConfigModel):
    """Ray cluster configuration (PU-specific, defaults to num_workers=1).

    PU training is single-worker only: every worker loads the
    *full* dataset itself, so ``num_workers > 1`` multiplies the
    materialization footprint.  Configuration with ``num_workers > 1`` is
    rejected at construction and re-checked inside the worker.  For
    distributed training use DNNTrainer + loss.type=nnpu.
    """

    num_workers: int = Field(
        default=1, ge=1, description="Number of workers (PU is single-worker only)"
    )
    use_gpu: bool = Field(default=False)
    storage_path: Optional[str] = None
    max_failures: int = Field(
        default=0,
        ge=-1,
        description=(
            "Max automatic retries on worker failure. "
            "0 = no retry, -1 = unlimited. "
            "When resume.enabled is true, retries restore the latest retained checkpoint."
        ),
    )
    resume: ResumeConfig = Field(default_factory=ResumeConfig)


class PUTrainingConfig(StrictConfigModel):
    """Complete configuration for PU distributed training."""

    data: Any = Field(default=None, description="Data source configuration")
    features: list[FeatureItemConfig] = Field(
        default_factory=list,
        description="Feature column configuration list",
    )
    model: DNNModelConfig = Field(default_factory=DNNModelConfig)
    pu: PUConfig = Field(default_factory=PUConfig)
    training: DNNTrainingParams = Field(default_factory=DNNTrainingParams)
    ray: PURayConfig = Field(default_factory=PURayConfig)
    resource: ResourceBudget = Field(
        default_factory=ResourceBudget,
        description="Single-worker materialization budget",
    )
    output: DNNOutputConfig = Field(default_factory=DNNOutputConfig)
    label_col: str = Field(default="label", description="Label column name")


# ── PU training loop (worker-level) ──


def pu_train_loop_per_worker(config: dict[str, Any]) -> None:
    """PU training main loop executed on each Ray worker.

    Data is loaded inside the worker (via ray.data); the driver never holds
    the dataset.  Reuses the DNN model architecture and enforces a PU loss.

    Config keys:
        data (dict): Canonical data config with ``source`` key (a ``SourceConfig`` dict).
        features (list): Feature column configs.
        label_col (str): Label column name, default ``"label"``.
        model (dict): DNN model config.
        pu (dict): PU loss config (loss_type, class_prior, class_prior_method, beta, gamma).
        training (dict): Training hyperparameters.
    """
    import json
    import logging
    import tempfile
    from pathlib import Path

    import numpy as np
    import ray.train
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from tributo.training.checkpoint import (
        ResumeConfig,
        capture_rng_state,
        checkpoint_directory,
        read_resume_manifest,
        restore_rng_state,
        write_resume_manifest,
    )
    from tributo.training.data_loader import load_ray_dataset_from_source
    from tributo.training.features.column_types import (
        features_from_dicts,
    )
    from tributo.training.features.dataset import IdentityDataset
    from tributo.training.features.transformer import FeatureTransformer
    from tributo.training.losses.pu_loss import PULoss
    from tributo.training.models.dnn import DNNModel
    from tributo.training.priors import estimate_class_prior

    logger = logging.getLogger(__name__)

    # Parse config
    features = features_from_dicts(config["features"])
    label_col = config.get("label_col", "label")
    model_cfg = config.get("model", {})
    pu_cfg = config.get("pu", {})
    training_cfg = config.get("training", {})
    resume_cfg = ResumeConfig.model_validate(config.get("resume") or {})
    resume_enabled = resume_cfg.effective_enabled
    checkpoint_interval = resume_cfg.checkpoint_interval
    resume_checkpoint = ray.train.get_checkpoint() if resume_enabled else None
    resume_transformer = None
    if resume_checkpoint is not None:
        with checkpoint_directory(resume_checkpoint) as checkpoint_dir:
            read_resume_manifest(
                checkpoint_dir,
                expected_trainer_type="pu",
                expected_resume_id=resume_cfg.resume_id,
            )
            resume_transformer = FeatureTransformer.load(
                checkpoint_dir / "preprocessor.json"
            )
        if resume_transformer.features != features:
            raise ValueError(
                "Resume checkpoint preprocessing features do not match the current run"
            )
    data_cfg = config.get("data")

    # ── Data loading (inside worker, bypasses driver) ──
    if data_cfg is None:
        raise ValueError("data config is required for PU training")

    source = data_cfg.get("source")
    if source is None:
        raise ValueError("data.source is required for PU training")

    # PU is single-worker only.  Every worker loads the full
    # dataset itself, so a second worker would duplicate the entire
    # materialization footprint.  Re-checked here (before any data is
    # read) even though construction already rejects num_workers > 1.
    world_size = ray.train.get_context().get_world_size()
    if world_size > 1:
        raise JobConfigurationError(
            "PU training supports num_workers=1 only; got "
            f"world_size={world_size}. Use DNNTrainer with loss.type=nnpu "
            "for distributed PU training."
        )

    ds = load_ray_dataset_from_source(source)

    # Convert to pandas under the worker materialization budget.
    # Rows are never silently truncated: an over-budget input fails here,
    # before the unbounded concat could run.  prefetch_batches=0 keeps the
    # prefetched batch out of the accounting blind spot.
    import pandas as pd

    budget = ResourceBudget.model_validate(config.get("resource") or {})
    worker_rank = ray.train.get_context().get_world_rank()
    preflight_check(
        rows=None,
        row_bytes=estimate_row_bytes_from_schema(ds.schema()),
        budget=budget,
        algorithm="pu",
        split="train",
        worker_rank=worker_rank,
    )
    batches, summary = collect_bounded(
        ds.iter_batches(
            batch_size=DEFAULT_BATCH_SIZE,
            batch_format="pandas",
            prefetch_batches=0,
        ),
        budget,
        algorithm="pu",
        split="train",
        worker_rank=worker_rank,
    )
    logger.info(
        "PU worker materialization: rows=%d payload=%dB peak=%dB",
        summary.rows_seen,
        summary.payload_bytes,
        summary.estimated_peak_bytes,
    )
    df = pd.concat(batches, ignore_index=True)
    del batches  # release the input list — the concat copy is the peak

    # Prepare data
    feature_names = [f.name for f in features]
    train_data = {name: df[name].values for name in feature_names}
    train_labels = df[label_col].values.astype(np.float32)

    # Train/val split
    val_size = training_cfg.get("val_size", 0.2)
    seed = training_cfg.get("seed", 42)
    n = len(train_labels)
    n_val = int(n * val_size)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    # Fit preprocessor
    transformer = resume_transformer or FeatureTransformer(features)
    train_data_split = {name: train_data[name][train_idx] for name in feature_names}
    if resume_transformer is None:
        train_processed = transformer.fit_transform(train_data_split)
    else:
        train_processed = transformer.transform(train_data_split)
    train_labels_split = train_labels[train_idx]

    val_processed = None
    val_labels_split = None
    if n_val > 0:
        val_processed = transformer.transform(
            {name: train_data[name][val_idx] for name in feature_names}
        )
        val_labels_split = train_labels[val_idx]

    # Build Dataset and DataLoader
    train_dataset = IdentityDataset(train_processed, train_labels_split, features)
    val_dataset = (
        IdentityDataset(val_processed, val_labels_split, features)
        if val_processed
        else None
    )

    batch_size = training_cfg.get("batch_size", 256)
    train_loader = DataLoader(
        train_dataset.to_torch_dataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_dataset.to_torch_dataset(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        if val_dataset
        else None
    )

    # Create model
    model = DNNModel(features, **model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Configure PU loss
    class_prior = pu_cfg.get("class_prior")
    if class_prior is None:
        prior_method = pu_cfg.get("class_prior_method", "label_frequency")
        class_prior = estimate_class_prior(
            int(train_labels_split.sum()),
            len(train_labels_split),
            method=prior_method,
        )
        logger.info(
            "Estimated class_prior: %.4f (method=%s)", class_prior, prior_method
        )

    loss_type = pu_cfg.get("loss_type", "nnpu")
    criterion = PULoss(
        class_prior=class_prior,
        beta=pu_cfg.get("beta", 0.0),
        gamma=pu_cfg.get("gamma", 1.0),
        loss_type=loss_type,
    )
    logger.info("PU Loss: %s, class_prior: %.4f", loss_type, class_prior)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 0.001),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if resume_checkpoint is not None:
        with checkpoint_directory(resume_checkpoint) as checkpoint_dir:
            envelope = read_resume_manifest(
                checkpoint_dir,
                expected_trainer_type="pu",
                expected_resume_id=resume_cfg.resume_id,
            )
            model_state = torch.load(
                checkpoint_dir / "model.pt", map_location="cpu", weights_only=True
            )
            optimizer_state = torch.load(
                checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True
            )
            rng_state = json.loads((checkpoint_dir / "rng_state.json").read_text())
            training_state = json.loads(
                (checkpoint_dir / "training_state.json").read_text()
            )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        restore_rng_state(rng_state)
        start_epoch = envelope.completed_step
        best_val_loss = float(training_state.get("best_val_loss", float("inf")))
        patience_counter = int(training_state.get("patience_counter", 0))
        logger.info(
            "Resuming PU training from %s at epoch %d",
            envelope.resume_id,
            start_epoch,
        )

    # ── Training loop ──
    epochs = training_cfg.get("epochs", 10)
    patience = training_cfg.get("early_stopping_patience")

    for epoch in range(start_epoch, epochs):
        stop_after_report = False
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            train_loss += loss.item() * labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                    labels = batch["label"].to(device)
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_loss += loss.item() * labels.size(0)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_loss /= val_total
            val_acc = val_correct / val_total

            # Early stopping
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        stop_after_report = True

        # Report metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "class_prior": class_prior,
        }
        if val_loader is not None:
            metrics["val_loss"] = val_loss
            metrics["val_acc"] = val_acc

        should_report = (
            not resume_enabled
            or (epoch + 1) % checkpoint_interval == 0
            or stop_after_report
            or epoch + 1 == epochs
        )
        if not should_report:
            continue

        # Save checkpoint (rank 0)
        world_rank = ray.train.get_context().get_world_rank()
        if world_rank == 0:
            from ray.train import Checkpoint

            checkpoint_dir = Path(tempfile.mkdtemp(prefix="pu_ckpt_"))

            torch.save(model.state_dict(), checkpoint_dir / "model.pt")

            preprocessor_state = {
                "features": [f.__dict__ for f in features],
                "label_encoders": {
                    k: {
                        str(kk): (
                            int(vv) if isinstance(vv, (np.integer, np.int64)) else vv
                        )
                        for kk, vv in v.items()
                    }
                    for k, v in transformer.label_encoders.items()
                },
                "norm_params": transformer.norm_params,
            }
            (checkpoint_dir / "preprocessor.json").write_text(
                json.dumps(preprocessor_state, ensure_ascii=False, default=str)
            )

            model_config = build_export_checkpoint_config(
                [f.__dict__ for f in features],
                model_cfg,
                trainer_type="pu",
                task_type="pu_classification",
                framework_version=torch.__version__,
                extra_metadata={
                    "pu": {**pu_cfg, "class_prior": class_prior},
                },
            )
            (checkpoint_dir / "model_config.json").write_text(
                json.dumps(model_config, ensure_ascii=False, default=str)
            )

            if resume_enabled:
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                (checkpoint_dir / "rng_state.json").write_text(
                    json.dumps(capture_rng_state(), ensure_ascii=False)
                )
                (checkpoint_dir / "training_state.json").write_text(
                    json.dumps(
                        {
                            "best_val_loss": best_val_loss,
                            "patience_counter": patience_counter,
                        },
                        ensure_ascii=False,
                    )
                )
                envelope = write_resume_manifest(
                    checkpoint_dir,
                    resume_id=resume_cfg.resume_id,
                    trainer_type="pu",
                    completed_step=epoch + 1,
                    framework="pytorch",
                    framework_version=torch.__version__,
                    payload_files=(
                        "model.pt",
                        "model_config.json",
                        "optimizer.pt",
                        "preprocessor.json",
                        "rng_state.json",
                        "training_state.json",
                    ),
                    payload_metadata={
                        "model": "model.pt",
                        "optimizer": "optimizer.pt",
                        "preprocessing": "preprocessor.json",
                        "rng": "rng_state.json",
                        "early_stopping": "training_state.json",
                    },
                )
                metrics["resume_id"] = envelope.resume_id

            checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
            ray.train.report(metrics, checkpoint=checkpoint)
        else:
            ray.train.report(metrics)

        if stop_after_report:
            break

    logger.info("PU training completed for worker (class_prior=%.4f)", class_prior)


# ── PUTrainer BaseTrainer implementation ──


@PublicAPI(stability="beta")
class PUTrainerImpl(BaseTrainer):
    """PU Learning distributed trainer.

    Focused on Positive-Unlabeled Learning scenarios; enforces PU loss.
    Data is loaded inside workers — the driver never holds the dataset.
    """

    def __init__(
        self,
        datasets: dict[str, ray.data.Dataset],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        *,
        _validated_config: PUTrainingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(datasets, config, run_config, **kwargs)
        self._pu_config = _validated_config or PUTrainingConfig.model_validate(config)
        self._features = build_features_from_config(self._pu_config.features)
        # PU is single-worker only — every worker re-loads the full
        # dataset, so num_workers > 1 multiplies the materialization footprint.
        if self._pu_config.ray.num_workers > 1:
            raise JobConfigurationError(
                "PU training supports num_workers=1 only; got "
                f"num_workers={self._pu_config.ray.num_workers}. "
                "Use DNNTrainer with loss.type=nnpu for distributed training."
            )

    def setup(self) -> None:
        """Build feature columns. Data loading is deferred to workers."""
        cfg = self._pu_config
        logger.info("Built %d feature columns", len(self._features))

        if cfg.data is None and "train" not in self.datasets:
            raise ValueError("data config or datasets must contain 'train'")

    def training_loop(self) -> Any:
        """Execute PU distributed training via Ray Train TorchTrainer."""
        import ray
        import ray.train
        from ray.train import FailureConfig, RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        from tributo.training.checkpoint import (
            checkpoint_config,
            load_initial_checkpoint,
        )

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        cfg = self._pu_config

        # Pass config to workers (no data on driver)
        train_loop_config = {
            "data": (
                cfg.data.model_dump() if hasattr(cfg.data, "model_dump") else cfg.data
            ),
            "features": [f.__dict__ for f in self._features],
            "label_col": cfg.label_col,
            "model": cfg.model.model_dump(),
            "pu": cfg.pu.model_dump(),
            "training": cfg.training.model_dump(),
            "resource": cfg.resource.model_dump(),
            "resume": cfg.ray.resume.model_dump(),
        }

        # Auto-detect storage_path
        storage_path = cfg.ray.storage_path
        if storage_path is None:
            import os
            import tempfile

            if os.path.isdir("/app/.ray_results"):
                storage_path = "/app/.ray_results"
            elif os.path.isdir("/workspace"):
                storage_path = "/workspace/ray_results"
            else:
                storage_path = os.path.join(tempfile.gettempdir(), "ray_results")

        logger.info("Using storage_path: %s", storage_path)

        # Build TorchTrainer (using the dedicated pu_train_loop_per_worker)
        trainer = TorchTrainer(
            train_loop_per_worker=pu_train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=cfg.ray.num_workers,
                use_gpu=cfg.ray.use_gpu,
                placement_strategy="SPREAD",
            ),
            run_config=RunConfig(
                name="tributo-pu",
                storage_path=storage_path,
                failure_config=FailureConfig(max_failures=cfg.ray.max_failures),
                checkpoint_config=checkpoint_config(cfg.ray.resume),
            ),
            resume_from_checkpoint=load_initial_checkpoint(
                cfg.ray.resume.checkpoint_path
            ),
        )

        logger.info(
            "Starting PU training (loss=%s, prior_method=%s)...",
            cfg.pu.loss_type,
            cfg.pu.class_prior_method,
        )
        result = trainer.fit()
        metrics = result.metrics or {}
        logger.info(
            "PU training done: epochs=%s, %s",
            cfg.training.epochs,
            {
                k: f"{float(v):.4f}"
                for k, v in metrics.items()
                if k not in ("features", "label_col")
                and not isinstance(v, (list, dict))
            },
        )
        return result

    def export_model(self, checkpoint: Any, output_path: str) -> None:
        """Export ONNX model package."""
        import json
        from pathlib import Path

        import numpy as np
        import torch

        from tributo.training.exporters.torch_onnx_exporter import export_model_package
        from tributo.training.features.column_types import SparseFeat
        from tributo.training.models.dnn import DNNModel

        result = checkpoint
        cfg = self._pu_config
        metrics = result.metrics or {}

        if result.checkpoint is None:
            raise RuntimeError("Training result has no checkpoint")

        logger.info("Loading checkpoint from: %s", result.checkpoint)

        with result.checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)

            model_state = torch.load(
                checkpoint_path / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            model = DNNModel(self._features, **cfg.model.model_dump())
            model.load_state_dict(model_state)

            preprocessor_state = json.loads(
                (checkpoint_path / "preprocessor.json").read_text()
            )

        # Prepare sample inputs
        sample_inputs = {}
        for feat in self._features:
            if isinstance(feat, SparseFeat):
                sample_inputs[feat.name] = np.array([0, 1], dtype=np.int64)
            else:
                sample_inputs[feat.name] = np.array([0.0, 1.0], dtype=np.float32)

        # Export model package
        output_dir = (
            Path(output_path)
            if output_path
            else Path(cfg.output.onnx_path or "model_output")
        )
        result_paths = export_model_package(
            model=model,
            sample_inputs=sample_inputs,
            output_dir=output_dir,
            feature_config=[f.__dict__ for f in self._features],
            preprocessor_state=preprocessor_state,
            metrics=metrics,
            opset_version=cfg.output.onnx_opset,
        )

        self._summary.update(
            {
                "onnx_path": str(result_paths.get("onnx_model")),
                "feature_config_path": str(result_paths.get("feature_config")),
                "preprocessor_path": str(result_paths.get("preprocessor")),
                "metrics": metrics,
            }
        )

    @staticmethod
    def _get_trainer_type() -> str:
        return "pu"


# ── Orchestration entry points ──


@PublicAPI(stability="beta")
def run_pu_training_with_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run PU distributed training from a configuration dict.

    This is the direct API for callers that already have a config dict
    in memory.  For YAML configs, convert to JSON first — YAML is no
    longer supported.

    Args:
        config: Training configuration dict.

    Returns:
        Training result summary dict.
    """
    cfg = PUTrainingConfig.model_validate(config)

    trainer = PUTrainerImpl(
        datasets={},
        config=config,
        _validated_config=cfg,
    )
    onnx_path = cfg.output.onnx_path or "model_output"
    return trainer.run(output_path=onnx_path)


@PublicAPI(stability="beta")
def run_pu_training_from_json(config_path: str) -> dict[str, Any]:
    """Run PU distributed training from a JSON configuration file.

    Args:
        config_path: Path to a JSON configuration file.

    Returns:
        Training result summary dict.

    Raises:
        ValueError: If *config_path* is a YAML file (no longer supported)
            or if the root is not a JSON mapping.
    """
    import json
    from pathlib import Path

    path = Path(config_path)
    if path.suffix in {".yaml", ".yml"}:
        raise ValueError(
            "YAML training config is no longer supported; please use JSON."
        )
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return run_pu_training_with_config(loaded)


# ── Built-in registration ──

_trainer_spec = AlgorithmSpec(
    name="pu",
    trainer_cls=PUTrainerImpl,
    problem_types=(ProblemType.PU_LEARNING,),
    data_modality=("tabular",),
    extras_group="identity",
    capabilities=(Capability.TUNABLE, Capability.EXPORTABLE),
    data_loading=DataLoadingMode.CANONICAL_TRAINER,
    resource_hints=ResourceHints(gpu_required=False),
    config_model=PUTrainingConfig,
)
register(_trainer_spec)

# Exported for entry_points discovery (see tributo.plugin)
trainer_spec = _trainer_spec
