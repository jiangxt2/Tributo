"""Ray Train/PyTorch DDP trainer shared by supervised DNN and PU adapters.

Supports Sparse/Dense features, PU Learning, Focal Loss and other identity mining scenarios.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from pydantic import Field, model_validator

from tributo._common.config import StrictConfigModel
from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.explainability.contracts import ExplainabilityConfig
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    DNN_DESCRIPTOR,
    build_legacy_spec,
)
from tributo.training.base import BaseTrainer
from tributo.training.checkpoint import ResumeConfig
from tributo.training.features.column_types import (
    DenseFeat,
    SparseFeat,
    features_from_dicts,
)
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

logger = logging.getLogger(__name__)


# ── Pydantic config models ──


class FeatureItemConfig(StrictConfigModel):
    """Single feature column configuration."""

    name: str
    type: str = Field(description="Feature type: sparse | dense")

    # Sparse feature attributes
    vocab_size: Optional[int] = Field(default=None, description="Number of categories")
    embedding_dim: int = Field(default=8, description="Embedding dimension")
    use_hash: bool = Field(default=False, description="Whether to use Hash Encoding")
    hash_bucket_size: int = Field(default=100000, description="Hash bucket size")

    # Dense feature attributes
    dimension: int = Field(default=1, description="Feature dimension")
    norm: str = Field(
        default="none",
        description="Normalization method: minmax | standard | log | none",
    )


PositiveClassPrior = Annotated[float, Field(gt=0, lt=1)]


class PULearningConfig(StrictConfigModel):
    """PU Learning configuration."""

    enabled: bool = Field(default=False, description="Whether to enable PU Learning")
    class_prior: Optional[PositiveClassPrior] = Field(
        default=None,
        description=(
            "Explicit positive class proportion. Required when PU learning is enabled."
        ),
    )
    class_prior_method: str = Field(
        default="simple",
        description=(
            "Compatibility metadata only; the trainer does not estimate a class "
            "prior automatically."
        ),
    )
    beta: float = Field(
        default=0.0,
        ge=0.0,
        description="Non-negative constraint threshold",
    )
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Correction-step gradient scale",
    )


class LossConfig(StrictConfigModel):
    """Loss function configuration."""

    type: Literal["bce", "focal", "nnpu"] = Field(
        default="bce",
        description="Loss type: bce | focal | nnpu",
    )
    alpha: float = Field(default=0.25, description="Focal Loss positive class weight")
    gamma: float = Field(default=2.0, description="Focal Loss focusing parameter")


class DNNModelConfig(StrictConfigModel):
    """DNN model configuration."""

    dnn_hidden_units: list[int] = Field(
        default=[256, 128, 64],
        description="DNN hidden layer dimensions",
    )
    dnn_dropout: float = Field(default=0.0, description="Dropout rate")
    use_batch_norm: bool = Field(default=False, description="Whether to use BatchNorm")


class DNNTrainingParams(StrictConfigModel):
    """DNN training hyperparameters."""

    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=256, ge=1, description="Batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    val_size: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="Validation set proportion"
    )
    seed: int = Field(default=42, description="Random seed")
    early_stopping_patience: Optional[int] = Field(
        default=None,
        description="Early stopping patience, None to disable",
    )


class DNNRayConfig(StrictConfigModel):
    """Ray Train worker-group configuration."""

    num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of DDP workers",
    )
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


class DNNOutputConfig(StrictConfigModel):
    """Output configuration."""

    bundle_uri: Optional[str] = None
    onnx_path: Optional[str] = None
    onnx_opset: int = Field(default=12, ge=1)
    metrics_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)

    @model_validator(mode="after")
    def validate_destination_contract(self) -> DNNOutputConfig:
        legacy_fields = {
            "onnx_path",
            "onnx_opset",
            "metrics_path",
            "preprocessor_path",
        }
        configured_legacy = sorted(legacy_fields & self.model_fields_set)
        if self.bundle_uri is not None and configured_legacy:
            raise ValueError(
                "output.bundle_uri cannot be combined with legacy output fields: "
                + ", ".join(configured_legacy)
            )
        if self.explainability.enabled and self.bundle_uri is None:
            raise ValueError("output.explainability.enabled requires output.bundle_uri")
        return self


class DNNTrainingConfig(StrictConfigModel):
    """Complete configuration for DNN training on Ray."""

    data: Any = Field(default=None, description="Data source configuration")
    features: list[FeatureItemConfig] = Field(
        default_factory=list,
        description="List of feature column configurations",
    )
    model: DNNModelConfig = Field(default_factory=DNNModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    pu_learning: PULearningConfig = Field(default_factory=PULearningConfig)
    training: DNNTrainingParams = Field(default_factory=DNNTrainingParams)
    ray: DNNRayConfig = Field(default_factory=DNNRayConfig)
    resource: ResourceBudget = Field(
        default_factory=ResourceBudget,
        description="Single-worker materialization budget",
    )
    output: DNNOutputConfig = Field(default_factory=DNNOutputConfig)
    label_col: str = Field(default="label", description="Label column name")

    @model_validator(mode="after")
    def validate_pu_loss_contract(self) -> DNNTrainingConfig:
        """Keep the PU feature flag, loss, and explicit prior consistent."""
        if self.loss.type == "nnpu":
            if not self.pu_learning.enabled:
                raise ValueError("loss.type='nnpu' requires pu_learning.enabled=true")
            if self.pu_learning.class_prior is None:
                raise ValueError(
                    "PU training requires an explicit pu_learning.class_prior"
                )
            if self.training.batch_size < 2:
                raise ValueError("nnPU training requires training.batch_size >= 2")
        elif self.pu_learning.enabled:
            raise ValueError("pu_learning.enabled=true requires loss.type='nnpu'")
        return self


# ── Config conversion functions ──


def build_features_from_config(
    feature_configs: list[FeatureItemConfig],
) -> list[SparseFeat | DenseFeat]:
    """Build a list of feature columns from configuration.

    Args:
        feature_configs: List of feature configurations.

    Returns:
        List of feature columns.
    """
    return features_from_dicts([cfg.model_dump() for cfg in feature_configs])


def validate_pu_labels(labels: Any, *, split: str) -> tuple[list[int], list[int]]:
    """Validate a PU split and return its positive and unlabeled row indices."""
    positive_indices: list[int] = []
    unlabeled_indices: list[int] = []
    for index, raw_label in enumerate(labels):
        try:
            label = float(raw_label)
        except (TypeError, ValueError) as exc:
            raise JobConfigurationError(
                f"PU {split} labels must contain only 1 (positive) or 0 "
                f"(unlabeled); got non-numeric {type(raw_label).__name__} "
                f"at row {index}"
            ) from exc
        if label == 1.0:
            positive_indices.append(index)
        elif label == 0.0:
            unlabeled_indices.append(index)
        else:
            raise JobConfigurationError(
                f"PU {split} labels must contain only 1 (positive) or 0 "
                f"(unlabeled); got {label} at row {index}"
            )
    if not positive_indices or not unlabeled_indices:
        raise JobConfigurationError(
            f"PU {split} split requires both positive and unlabeled examples; "
            f"got positive={len(positive_indices)}, "
            f"unlabeled={len(unlabeled_indices)}"
        )
    return positive_indices, unlabeled_indices


def parse_positive_class_prior(value: Any, *, config_path: str) -> float:
    """Parse an untyped worker config value using the shared prior contract."""
    if value is None:
        raise JobConfigurationError(
            f"PU training requires an explicit {config_path} in the range (0, 1)"
        )
    try:
        class_prior = float(value)
    except (TypeError, ValueError) as exc:
        raise JobConfigurationError(
            f"{config_path} must be a number in the range (0, 1)"
        ) from exc
    if not 0 < class_prior < 1:
        raise JobConfigurationError(
            f"{config_path} must be in the range (0, 1), got {class_prior}"
        )
    return class_prior


def split_pu_indices(
    labels: Any,
    *,
    val_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Build a deterministic stratified PU train/validation split."""
    positive_indices, unlabeled_indices = validate_pu_labels(labels, split="input")
    if val_size <= 0:
        return positive_indices + unlabeled_indices, []
    if len(positive_indices) < 2 or len(unlabeled_indices) < 2:
        raise JobConfigurationError(
            "PU validation requires at least two positive and two unlabeled "
            "examples so both train and validation splits preserve the PU contract; "
            "set training.val_size=0 to disable validation for smaller datasets"
        )

    rng = random.Random(seed)
    rng.shuffle(positive_indices)
    rng.shuffle(unlabeled_indices)

    def _validation_count(count: int) -> int:
        return max(1, min(count - 1, round(count * val_size)))

    positive_val_count = _validation_count(len(positive_indices))
    unlabeled_val_count = _validation_count(len(unlabeled_indices))
    val_indices = (
        positive_indices[:positive_val_count] + unlabeled_indices[:unlabeled_val_count]
    )
    train_indices = (
        positive_indices[positive_val_count:] + unlabeled_indices[unlabeled_val_count:]
    )
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


class _PairedPUBatchSampler:
    """Partition every P/U row once into bounded batches containing both classes."""

    def __init__(self, labels: Any, *, batch_size: int, seed: int) -> None:
        if batch_size < 2:
            raise JobConfigurationError("PU training requires batch_size >= 2")
        self._positive, self._unlabeled = validate_pu_labels(labels, split="train")
        self._positive_batch_size = max(1, batch_size // 2)
        self._unlabeled_batch_size = batch_size - self._positive_batch_size
        minimum_batches = math.ceil(
            (len(self._positive) + len(self._unlabeled)) / batch_size
        )
        if minimum_batches > min(len(self._positive), len(self._unlabeled)):
            raise JobConfigurationError(
                "PU class imbalance cannot form bounded batches that each contain "
                "positive and unlabeled rows without replaying samples; increase "
                "training.batch_size or reduce worker_count"
            )
        self._batch_count = minimum_batches
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self._batch_count

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic ordering for an absolute training epoch."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._epoch)
        positive = list(self._positive)
        unlabeled = list(self._unlabeled)
        rng.shuffle(positive)
        rng.shuffle(unlabeled)

        def _partition(values: list[int], *, reverse_large: bool) -> list[list[int]]:
            quotient, remainder = divmod(len(values), self._batch_count)
            sizes = [
                quotient + (1 if index < remainder else 0)
                for index in range(self._batch_count)
            ]
            if reverse_large:
                sizes.reverse()
            result: list[list[int]] = []
            start = 0
            for size in sizes:
                result.append(values[start : start + size])
                start += size
            return result

        positive_batches = _partition(positive, reverse_large=False)
        unlabeled_batches = _partition(unlabeled, reverse_large=True)
        for positive_batch, unlabeled_batch in zip(positive_batches, unlabeled_batches):
            batch = [*positive_batch, *unlabeled_batch]
            if len(batch) > self._positive_batch_size + self._unlabeled_batch_size:
                raise JobConfigurationError(
                    "PU paired batch partition exceeded training.batch_size"
                )
            rng.shuffle(batch)
            yield batch


def build_pu_train_loader(
    dataset: Any,
    labels: Any,
    *,
    batch_size: int,
    seed: int,
) -> Any:
    """Create a DataLoader whose every training batch contains P and U rows."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_sampler=_PairedPUBatchSampler(
            labels,
            batch_size=batch_size,
            seed=seed,
        ),
        num_workers=0,
    )


def warn_if_ignored_class_prior_method(
    method: Any,
    *,
    default_method: str,
    config_path: str,
    target_logger: logging.Logger,
) -> None:
    """Warn when legacy prior-estimation metadata no longer drives training."""
    if method != default_method:
        target_logger.warning(
            "%s=%r is compatibility metadata only and does not trigger "
            "class-prior estimation; the explicit class_prior value is used",
            config_path,
            method,
        )


def validate_finite_training_metrics(
    metrics: Mapping[str, int | float],
    *,
    algorithm: str,
) -> None:
    """Reject non-finite metrics before early stopping, reporting, or export."""
    non_finite = sorted(
        name for name, value in metrics.items() if not math.isfinite(float(value))
    )
    if non_finite:
        raise JobExecutionError(
            f"{algorithm} produced non-finite training metrics: {', '.join(non_finite)}"
        )


def _validate_distributed_resume_metadata(
    envelope: Any,
    *,
    expected_world_size: int,
    expected_distribution_digest: str | None,
) -> None:
    """Reject a checkpoint from a different collective execution contract."""
    metadata = envelope.payload_metadata
    checkpoint_world_size = metadata.get("world_size")
    if checkpoint_world_size is None:
        if expected_world_size != 1:
            raise JobConfigurationError(
                "Legacy resume checkpoints without world_size metadata cannot "
                "start a multi-worker training run"
            )
    elif checkpoint_world_size != expected_world_size:
        raise JobConfigurationError(
            "Resume checkpoint world_size does not match the current worker group"
        )
    if (
        expected_distribution_digest is not None
        and metadata.get("distribution_spec_digest") != expected_distribution_digest
    ):
        raise JobConfigurationError(
            "Resume checkpoint DistributionSpec digest does not match the "
            "current execution plan"
        )


def build_export_checkpoint_config(
    feature_configs: list[dict[str, Any]],
    model_config: dict[str, Any],
    *,
    trainer_type: str,
    task_type: str,
    framework_version: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable export metadata written into a Torch checkpoint.

    The model-specific reconstruction parameters remain at the top level for
    compatibility with the DNN source provider.  The envelope fields are
    deliberately explicit so exporters never need to infer a signature from
    a raw model file.
    """
    normalized_features = []
    input_schema: list[dict[str, Any]] = []
    for feature in feature_configs:
        normalized_feature = dict(feature)
        if "norm" in normalized_feature:
            normalized_feature["norm"] = getattr(
                normalized_feature["norm"], "value", normalized_feature["norm"]
            )
        normalized_features.append(normalized_feature)
        name = str(feature["name"])
        is_sparse = "vocab_size" in feature
        dimension = int(feature.get("dimension", 1))
        shape: list[int | str] = ["batch"]
        if not is_sparse and dimension > 1:
            shape.append(dimension)
        input_schema.append(
            {
                "name": name,
                "dtype": feature.get("dtype") or ("int64" if is_sparse else "float32"),
                "shape": shape,
            }
        )

    metadata: dict[str, Any] = {
        **model_config,
        "features": normalized_features,
        "schema_version": 1,
        "trainer_type": trainer_type,
        "architecture_id": "dnn",
        "input_schema": input_schema,
        "output_schema": [{"name": "output", "dtype": "float32", "shape": ["batch"]}],
        "preprocessing": {
            "artifact": "preprocessor.json",
            "type": "FeatureTransformer",
        },
        "task_type": task_type,
        "framework": "pytorch",
        "framework_version": framework_version,
        "checkpoint_format_version": 1,
        "required_artifacts": [
            "model.pt",
            "model_config.json",
            "preprocessor.json",
        ],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


# ── DNN BaseTrainer implementation ──


@PublicAPI(stability="beta")
class DNNTrainerImpl(BaseTrainer):
    """Distributed DNN implementation of BaseTrainer.

    Supports supervised Sparse/Dense BCE and Focal Loss training. Historical
    nnPU configuration must enter through the compatibility planner alias.
    """

    def __init__(
        self,
        datasets: dict[str, ray.data.Dataset],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        *,
        _validated_config: DNNTrainingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(datasets, config, run_config, **kwargs)
        self._train_config = _validated_config or DNNTrainingConfig.model_validate(
            config
        )
        if self._train_config.loss.type == "nnpu":
            raise JobConfigurationError(
                "DNN loss.type='nnpu' is a compatibility alias; use "
                "run_dnn_training_with_config() for migration or select the "
                "canonical 'pu' algorithm"
            )
        self._features: list[SparseFeat | DenseFeat] = []
        self._transformer: Any = None

    def setup(self) -> None:
        """Data preprocessing: build feature columns, fit preprocessor."""
        from tributo.training.data_loader import load_ray_dataset_from_config

        cfg = self._train_config

        # Build feature columns
        self._features = build_features_from_config(cfg.features)
        logger.info("Built %d feature columns", len(self._features))

        # Load data
        if cfg.data is not None:
            ds = load_ray_dataset_from_config(cfg.data)
            self.datasets = {"train": ds}

        ds = self.datasets.get("train")
        if ds is None:
            raise ValueError("datasets must contain 'train' key")

        # Split train/val
        val_size = cfg.training.val_size
        if val_size > 0:
            train_ds, val_ds = ds.train_test_split(
                test_size=val_size, seed=cfg.training.seed
            )
            self.datasets = {"train": train_ds, "val": val_ds}
            logger.info(
                "Split dataset: train=%d, val=%d", train_ds.count(), val_ds.count()
            )

    def training_loop(self) -> Any:
        """Invoke Ray Train TorchTrainer to execute distributed training."""
        cfg = self._train_config
        storage_path = self.run_config.get("storage_path")
        if storage_path is None:
            storage_path = cfg.ray.storage_path
        elif not isinstance(storage_path, str) or not storage_path:
            raise JobConfigurationError(
                "run_config 'storage_path' must be a non-empty string or None"
            )
        run_name = self.run_config.get("name")
        if run_name is None:
            run_name = "tributo-dnn"
        elif not isinstance(run_name, str) or not run_name:
            raise JobConfigurationError(
                "run_config 'name' must be a non-empty string or None"
            )

        import ray
        import ray.train
        from ray.train import FailureConfig, RunConfig, ScalingConfig

        from tributo.training.checkpoint import (
            checkpoint_config,
            load_initial_checkpoint,
        )

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        # Prepare training config
        train_loop_config = {
            "features": [f.__dict__ for f in self._features],
            "label_col": cfg.label_col,
            "model": cfg.model.model_dump(),
            "loss": cfg.loss.model_dump(),
            "pu_learning": cfg.pu_learning.model_dump(),
            "training": cfg.training.model_dump(),
            "resource": cfg.resource.model_dump(),
            "resume": cfg.ray.resume.model_dump(),
        }

        # Auto-detect storage_path
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

        from ray.train.torch import TorchTrainer

        # Build TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=dnn_train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=cfg.ray.num_workers,
                use_gpu=cfg.ray.use_gpu,
                placement_strategy="SPREAD",
            ),
            datasets=self.datasets,
            run_config=RunConfig(
                name=run_name,
                storage_path=storage_path,
                failure_config=FailureConfig(max_failures=cfg.ray.max_failures),
                checkpoint_config=checkpoint_config(cfg.ray.resume),
            ),
            resume_from_checkpoint=load_initial_checkpoint(
                cfg.ray.resume.checkpoint_path
            ),
        )

        logger.info("Starting DNN training...")
        result = trainer.fit()
        metrics = result.metrics or {}
        logger.info(
            "Training done: epochs=%s, %s",
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
        """Export ONNX to local or S3, save metrics.

        Args:
            checkpoint: Ray Train Result object returned by training_loop().
            output_path: Export path.
        """
        import json
        from pathlib import Path

        import numpy as np
        import torch

        from tributo.training.exporters.torch_onnx_exporter import export_model_package
        from tributo.training.models.dnn import DNNModel

        result = checkpoint
        cfg = self._train_config
        metrics = result.metrics or {}

        # Load model from checkpoint
        if result.checkpoint is None:
            raise RuntimeError("Training result has no checkpoint")

        logger.info("Loading checkpoint from: %s", result.checkpoint)

        # Load model state
        with result.checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)

            # Load model weights
            model_state = torch.load(
                checkpoint_path / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            model = DNNModel(self._features, **cfg.model.model_dump())
            model.load_state_dict(model_state)

            # Load preprocessor
            preprocessor_state = json.loads(
                (checkpoint_path / "preprocessor.json").read_text()
            )

        # Prepare sample inputs (for ONNX export)
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

        # Save metrics
        if cfg.output.metrics_path:
            metrics_path = Path(cfg.output.metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    metrics,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                )
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
        return "dnn"

    @staticmethod
    def _default_bundle_targets() -> tuple[Any, ...]:
        from tributo.exporting.models import ExportTarget

        return (ExportTarget(name="onnx-model", format="onnx", options={"opset": 18}),)

    @staticmethod
    def _default_bundle_roles() -> dict[str, str]:
        return {"inference": "onnx-model"}


# ── Training loop helpers ──


def _forward_step(
    model: Any,
    batch: dict[str, Any],
    criterion: Any,
    device: Any,
) -> tuple[Any, int, int]:
    """Execute a single forward pass + loss calculation + accuracy statistics.

    Args:
        model: DNN model.
        batch: Data batch (contains features and labels).
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (loss_tensor, correct_count, sample_count).
    """
    logits, labels, correct, total = _predict_step(model, batch, device)
    loss = criterion(logits, labels)
    return loss, correct, total


def _predict_step(
    model: Any,
    batch: dict[str, Any],
    device: Any,
) -> tuple[Any, Any, int, int]:
    """Run inference for one batch and return observed-label statistics."""
    import torch

    inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
    labels = batch["label"].to(device)

    logits = model(inputs)
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return logits, labels, correct, total


def evaluate_pu_split(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
) -> tuple[float, float]:
    """Evaluate one complete PU split without paired-sampler oversampling."""
    import torch

    risk = criterion.new_risk_accumulator()
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            logits, labels, batch_correct, batch_total = _predict_step(
                model,
                batch,
                device,
            )
            risk.update(logits, labels)
            correct += batch_correct
            total += batch_total
    from tributo.training.distributed_torch import all_reduce_values

    (
        positive_loss_sum,
        positive_as_negative_loss_sum,
        unlabeled_negative_loss_sum,
        positive_count,
        unlabeled_count,
        global_correct,
        global_total,
    ) = all_reduce_values(
        (
            risk.positive_loss_sum,
            risk.positive_as_negative_loss_sum,
            risk.unlabeled_negative_loss_sum,
            risk.positive_count,
            risk.unlabeled_count,
            correct,
            total,
        )
    )
    risk.positive_loss_sum = positive_loss_sum
    risk.positive_as_negative_loss_sum = positive_as_negative_loss_sum
    risk.unlabeled_negative_loss_sum = unlabeled_negative_loss_sum
    risk.positive_count = int(positive_count)
    risk.unlabeled_count = int(unlabeled_count)
    if global_total == 0:
        raise JobConfigurationError("PU evaluation split must not be empty")
    try:
        empirical_risk = risk.value()
    except ValueError as exc:
        raise JobConfigurationError(f"Invalid PU evaluation split: {exc}") from exc
    return empirical_risk, global_correct / global_total


# ── Training loop (Ray worker level) ──


def dnn_train_loop_per_worker(config: dict[str, Any]) -> None:
    """DNN training main loop executed by each Ray worker.

    Args:
        config: Training configuration dictionary.
    """
    import json
    import logging
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import ray.train
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from tributo.exceptions import JobConfigurationError
    from tributo.training.checkpoint import (
        ResumeConfig,
        capture_rng_state,
        checkpoint_directory,
        read_resume_manifest,
        restore_rng_state,
        write_resume_manifest,
    )
    from tributo.training.distributed_torch import (
        all_gather_objects,
        all_reduce_max,
        all_reduce_values,
        broadcast_bool,
        collective_execution_evidence,
        distributed_pu_loss,
        equalized_batches,
        fit_global_feature_transformer,
        prepare_model,
        unwrapped_model,
    )
    from tributo.training.features.column_types import (
        features_from_dicts,
    )
    from tributo.training.features.dataset import IdentityDataset
    from tributo.training.features.transformer import FeatureTransformer
    from tributo.training.losses.focal_loss import FocalLoss
    from tributo.training.losses.pu_loss import PULoss
    from tributo.training.models.dnn import DNNModel

    # Configure logger inside worker
    logger = logging.getLogger(__name__)

    # Parse config
    features = features_from_dicts(config["features"])

    label_col = config.get("label_col", "label")
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    pu_cfg = config.get("pu_learning", {})
    training_cfg = config.get("training", {})
    trainer_type = str(config.get("trainer_type", "dnn"))
    if trainer_type not in {"dnn", "pu"}:
        raise JobConfigurationError(
            f"invalid distributed Torch trainer_type: {trainer_type!r}"
        )
    context = ray.train.get_context()

    loss_type = loss_cfg.get("type", "bce")
    pu_mode = loss_type in {"nnpu", "upu"}
    if trainer_type == "dnn" and pu_mode:
        raise JobConfigurationError(
            "DNN nnPU is a compatibility alias; route it through the canonical "
            "PU trainer instead of invoking the DNN worker directly"
        )
    pu_criterion = None
    class_prior: float | None = None
    prior_method = pu_cfg.get(
        "class_prior_method",
        "label_frequency" if trainer_type == "pu" else "simple",
    )
    if pu_mode:
        if not pu_cfg.get("enabled", False):
            raise JobConfigurationError(
                "loss.type='nnpu' requires pu_learning.enabled=true"
            )
        raw_prior = pu_cfg.get("class_prior")
        if raw_prior is not None:
            class_prior = parse_positive_class_prior(
                raw_prior,
                config_path=(
                    "pu.class_prior"
                    if trainer_type == "pu"
                    else "pu_learning.class_prior"
                ),
            )
        elif prior_method != "label_frequency":
            raise JobConfigurationError(
                "PU training without an explicit class_prior supports only "
                "class_prior_method='label_frequency'"
            )
    elif pu_cfg.get("enabled", False):
        raise JobConfigurationError(
            "pu_learning.enabled=true requires loss.type='nnpu'"
        )

    resume_cfg = ResumeConfig.model_validate(config.get("resume") or {})
    resume_enabled = resume_cfg.effective_enabled
    checkpoint_interval = resume_cfg.checkpoint_interval
    resume_checkpoint = ray.train.get_checkpoint() if resume_enabled else None
    if resume_checkpoint is None and resume_enabled:
        resume_checkpoint = resume_cfg.checkpoint_path
    resume_transformer = None
    if resume_checkpoint is not None:
        with checkpoint_directory(resume_checkpoint) as checkpoint_dir:
            resume_envelope = read_resume_manifest(
                checkpoint_dir,
                expected_trainer_type=trainer_type,
                expected_resume_id=resume_cfg.resume_id,
            )
            _validate_distributed_resume_metadata(
                resume_envelope,
                expected_world_size=context.get_world_size(),
                expected_distribution_digest=config.get(
                    "_tributo_distribution_spec_digest"
                ),
            )
            resume_transformer = FeatureTransformer.load(
                checkpoint_dir / "preprocessor.json"
            )
        if resume_transformer.features != features:
            raise ValueError(
                "Resume checkpoint preprocessing features do not match the current run"
            )

    # Get data (StreamSplitDataIterator, must collect via iter_batches).  PU
    # receives independently split P/U datasets so every rank can construct
    # globally coordinated risk batches without re-opening the full source.
    # Supervised DNN receives a globally split train/validation pair from the
    # formal adapter, rather than selecting validation rows independently in
    # each rank.
    if trainer_type == "pu":
        train_shards = (
            ("positive", ray.train.get_dataset_shard("positive")),
            ("unlabeled", ray.train.get_dataset_shard("unlabeled")),
        )
        validation_enabled = float(training_cfg.get("val_size", 0.2)) > 0
        if validation_enabled:
            try:
                val_shards = (
                    ("positive_val", ray.train.get_dataset_shard("positive_val")),
                    ("unlabeled_val", ray.train.get_dataset_shard("unlabeled_val")),
                )
            except KeyError as exc:
                raise JobConfigurationError(
                    "PU validation is enabled but the global stratified input "
                    "adapter did not provide validation shards"
                ) from exc
            if any(shard is None for _, shard in val_shards):
                raise JobConfigurationError(
                    "PU validation is enabled but the global stratified input "
                    "adapter provided an empty validation shard"
                )
        else:
            val_shards = ()
    else:
        train_shards = (("train", ray.train.get_dataset_shard("train")),)
        validation_enabled = float(training_cfg.get("val_size", 0.2)) > 0
        try:
            val_ds = ray.train.get_dataset_shard("val")
        except KeyError as exc:
            # Compatibility callers may disable validation and provide only
            # the training Dataset. Formal DNN creates the split before
            # Ray Train assigns shards.
            if validation_enabled:
                raise JobConfigurationError(
                    "DNN validation is enabled but the formal input adapter did not "
                    "provide a 'val' dataset shard"
                ) from exc
            val_ds = None
        if validation_enabled and val_ds is None:
            raise JobConfigurationError(
                "DNN validation is enabled but the formal input adapter did not "
                "provide a 'val' dataset shard"
            )
        val_shards = (("val", val_ds),) if val_ds is not None else ()

    # Convert to pandas under the worker materialization budget.
    # train and val share one budget — both frames stay alive together —
    # and either split exceeding it fails fast before the unbounded concat.
    # Rows are never silently truncated.
    import pandas as pd

    budget = ResourceBudget.model_validate(config.get("resource") or {})
    worker_rank = context.get_world_rank()
    row_bytes = estimate_row_bytes_from_schema(train_shards[0][1].schema())
    preflight_check(
        rows=None,
        row_bytes=row_bytes,
        budget=budget,
        algorithm="dnn",
        split="train",
        worker_rank=worker_rank,
    )
    collector = BoundedCollector(
        budget, algorithm="dnn", split="train", worker_rank=worker_rank
    )
    input_rows: dict[str, int] = {name: 0 for name, _ in train_shards}
    # prefetch_batches=0: a prefetched batch would be held outside the
    # collector's accounting.
    train_batches = []
    for shard_name, train_ds in train_shards:
        for batch in train_ds.iter_batches(
            batch_size=DEFAULT_BATCH_SIZE,
            batch_format="pandas",
            prefetch_batches=0,
        ):
            collector.add(batch, split=shard_name)
            input_rows[shard_name] += len(batch)
            train_batches.append(batch)
    if not train_batches:
        raise JobConfigurationError("DNN/PU worker received an empty training shard")
    train_df = pd.concat(train_batches, ignore_index=True)
    del train_batches  # release the input list — the concat copy is the peak

    val_df = None
    if val_shards:
        if row_bytes is not None:
            preflight_check(
                rows=None,
                row_bytes=row_bytes,
                budget=budget,
                algorithm="dnn",
                split="val",
                worker_rank=worker_rank,
            )
        val_batches = []
        for shard_name, val_ds in val_shards:
            input_rows[shard_name] = 0
            for batch in val_ds.iter_batches(
                batch_size=DEFAULT_BATCH_SIZE,
                batch_format="pandas",
                prefetch_batches=0,
            ):
                collector.add(batch, split=shard_name)
                input_rows[shard_name] += len(batch)
                val_batches.append(batch)
        if val_batches:
            val_df = pd.concat(val_batches, ignore_index=True)
            del val_batches

    logger.info(
        "DNN worker materialization: rows=%d payload=%dB peak=%dB",
        collector.summary.rows_seen,
        collector.summary.payload_bytes,
        collector.summary.estimated_peak_bytes,
    )

    # Prepare data
    feature_names = [f.name for f in features]
    all_train_data = {name: train_df[name].values for name in feature_names}
    all_train_labels = train_df[label_col].values.astype(np.float32)
    if pu_mode:
        if class_prior is None:
            positive_count, total_count = all_reduce_values(
                (
                    float((all_train_labels == 1).sum()),
                    float(all_train_labels.size),
                )
            )
            class_prior = parse_positive_class_prior(
                positive_count / total_count if total_count else None,
                config_path="pu.class_prior[label_frequency]",
            )
        else:
            warn_if_ignored_class_prior_method(
                prior_method,
                default_method=(
                    "label_frequency" if trainer_type == "pu" else "simple"
                ),
                config_path=(
                    "pu.class_prior_method"
                    if trainer_type == "pu"
                    else "pu_learning.class_prior_method"
                ),
                target_logger=logger,
            )
        try:
            pu_criterion = PULoss(
                class_prior=class_prior,
                beta=float(pu_cfg.get("beta", 0.0)),
                gamma=float(pu_cfg.get("gamma", 1.0)),
                loss_type=loss_type,
            )
        except (TypeError, ValueError) as exc:
            raise JobConfigurationError(
                f"Invalid PU loss configuration: {exc}"
            ) from exc

    val_data = None
    val_labels = None
    if val_df is not None:
        train_data = all_train_data
        train_labels = all_train_labels
        val_data = {name: val_df[name].values for name in feature_names}
        val_labels = val_df[label_col].values.astype(np.float32)
        if pu_mode:
            validate_pu_labels(train_labels, split="train")
            validate_pu_labels(val_labels, split="validation")
    elif pu_mode:
        if float(training_cfg.get("val_size", 0.2)) > 0:
            raise JobConfigurationError(
                "PU validation must be split globally before worker sharding"
            )
        validate_pu_labels(all_train_labels, split="train")
        train_data = all_train_data
        train_labels = all_train_labels
    else:
        train_data = all_train_data
        train_labels = all_train_labels

    # Fit one global preprocessor from mergeable shard-local statistics.  Every
    # rank receives identical encoders and dense normalization parameters.
    transformer = resume_transformer or fit_global_feature_transformer(
        features,
        train_data,
    )
    train_processed = transformer.transform(train_data)

    val_processed = None
    if val_data is not None:
        val_processed = transformer.transform(val_data)

    # Create Dataset
    train_dataset = IdentityDataset(train_processed, train_labels, features)
    val_dataset = (
        IdentityDataset(val_processed, val_labels, features) if val_processed else None
    )

    # Create DataLoader
    torch_train_dataset = train_dataset.to_torch_dataset()
    batch_size = training_cfg.get("batch_size", 256)
    if pu_mode:
        train_loader = build_pu_train_loader(
            torch_train_dataset,
            train_labels,
            batch_size=batch_size,
            seed=training_cfg.get("seed", 42),
        )
    else:
        train_loader = DataLoader(
            torch_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
    pu_train_eval_loader = (
        DataLoader(
            torch_train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        if pu_mode
        else None
    )

    torch_val_dataset = val_dataset.to_torch_dataset() if val_dataset else None
    val_loader = (
        DataLoader(
            torch_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        if torch_val_dataset
        else None
    )

    # Create model
    seed = int(training_cfg.get("seed", 42)) + worker_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DNNModel(features, **model_cfg)

    # Configure loss function
    if pu_mode:
        assert pu_criterion is not None
        criterion = pu_criterion
    elif loss_type == "focal":
        criterion = FocalLoss(
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
        )
    else:
        # Standard BCE
        criterion = torch.nn.BCEWithLogitsLoss()

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
                expected_trainer_type=trainer_type,
                expected_resume_id=resume_cfg.resume_id,
            )
            _validate_distributed_resume_metadata(
                envelope,
                expected_world_size=context.get_world_size(),
                expected_distribution_digest=config.get(
                    "_tributo_distribution_spec_digest"
                ),
            )
            model_state = torch.load(
                checkpoint_dir / "model.pt", map_location="cpu", weights_only=True
            )
            optimizer_state = torch.load(
                checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True
            )
            rng_payload = json.loads((checkpoint_dir / "rng_state.json").read_text())
            training_state = json.loads(
                (checkpoint_dir / "training_state.json").read_text()
            )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        if isinstance(rng_payload, dict) and isinstance(
            rng_payload.get("rank_states"), list
        ):
            rank_states = rng_payload["rank_states"]
            if len(rank_states) != context.get_world_size():
                raise JobConfigurationError(
                    "Resume checkpoint RNG state count does not match world_size"
                )
            restore_rng_state(rank_states[worker_rank])
        else:
            # Compatibility with pre-distribution single-worker checkpoints.
            restore_rng_state(rng_payload)
        start_epoch = envelope.completed_step
        best_val_loss = float(training_state.get("best_val_loss", float("inf")))
        patience_counter = int(training_state.get("patience_counter", 0))
        logger.info(
            "Resuming DNN training from %s at epoch %d",
            envelope.resume_id,
            start_epoch,
        )

    # ``prepare_model`` moves the model and wraps it with DDP.  The optimizer
    # already references the same Parameter objects, so its restored state
    # remains valid after wrapping.
    model, device = prepare_model(model)

    # Training loop
    epochs = training_cfg.get("epochs", 10)
    patience = training_cfg.get("early_stopping_patience")
    local_batch_count = len(train_loader)
    collective_steps = all_reduce_max(local_batch_count)

    for epoch in range(start_epoch, epochs):
        stop_after_report = False
        # Training phase
        model.train()
        if pu_mode:
            train_loader.batch_sampler.set_epoch(epoch)
        train_objective = 0.0
        train_objective_weight = 0.0
        train_correct = 0
        train_total = 0

        for batch, active in equalized_batches(
            train_loader,
            collective_steps=collective_steps,
        ):
            if active:
                if pu_mode:
                    logits, labels, correct, total = _predict_step(model, batch, device)
                    loss = distributed_pu_loss(criterion, logits, labels)
                    world_size = (
                        torch.distributed.get_world_size()
                        if torch.distributed.is_initialized()
                        else 1
                    )
                else:
                    raw_loss, correct, total = _forward_step(
                        model, batch, criterion, device
                    )
                    (global_batch_total,) = all_reduce_values((total,))
                    if global_batch_total <= 0:
                        raise JobConfigurationError(
                            "DNN optimization step has no active rows"
                        )
                    world_size = (
                        torch.distributed.get_world_size()
                        if torch.distributed.is_initialized()
                        else 1
                    )
                    loss = raw_loss * world_size * total / global_batch_total
                    train_objective += raw_loss.item() * total
                    train_objective_weight += total
            else:
                # Empty tensors keep DDP/custom collective call order aligned
                # without replaying an observed row from a shorter shard.
                logits, labels, correct, total = _predict_step(model, batch, device)
                if pu_mode:
                    loss = distributed_pu_loss(criterion, logits, labels)
                else:
                    (global_batch_total,) = all_reduce_values((0.0,))
                    if global_batch_total <= 0:
                        raise JobConfigurationError(
                            "DNN optimization step has no active rows"
                        )
                    loss = logits.sum() * 0.0
            if pu_mode:
                world_size = (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized()
                    else 1
                )
                train_objective += loss.item() / world_size
                train_objective_weight += 1.0 / world_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_correct += correct
            train_total += total

        (
            objective_sum,
            objective_weight,
            global_correct,
            global_total,
        ) = all_reduce_values(
            (
                train_objective,
                train_objective_weight,
                train_correct,
                train_total,
            )
        )
        if global_total <= 0 or objective_weight <= 0:
            raise JobConfigurationError("DNN training received an empty worker group")
        train_objective = objective_sum / objective_weight
        if pu_train_eval_loader is not None:
            train_loss, train_observed_label_accuracy = evaluate_pu_split(
                unwrapped_model(model),
                pu_train_eval_loader,
                criterion,
                device,
            )
        else:
            train_loss = train_objective
            train_observed_label_accuracy = global_correct / global_total

        # Validation phase
        val_loss = 0.0
        val_observed_label_accuracy = 0.0
        if val_loader is not None:
            if pu_mode:
                val_loss, val_observed_label_accuracy = evaluate_pu_split(
                    unwrapped_model(model),
                    val_loader,
                    criterion,
                    device,
                )
            else:
                evaluation_model = unwrapped_model(model)
                evaluation_model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        loss, correct, total = _forward_step(
                            evaluation_model, batch, criterion, device
                        )
                        val_loss += loss.item() * total
                        val_correct += correct
                        val_total += total
                val_loss_sum, global_val_correct, global_val_total = all_reduce_values(
                    (val_loss, val_correct, val_total)
                )
                if global_val_total <= 0:
                    raise JobConfigurationError(
                        "DNN validation received an empty worker group"
                    )
                val_loss = val_loss_sum / global_val_total
                val_observed_label_accuracy = global_val_correct / global_val_total

        # Report metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
        }
        if pu_mode:
            assert pu_criterion is not None
            metrics["train_optimization_objective"] = train_objective
            metrics["train_observed_label_accuracy"] = train_observed_label_accuracy
            metrics["train_acc"] = train_observed_label_accuracy
            metrics["class_prior"] = pu_criterion.class_prior
        else:
            metrics["train_acc"] = train_observed_label_accuracy
        if val_loader is not None:
            metrics["val_loss"] = val_loss
            if pu_mode:
                metrics["val_observed_label_accuracy"] = val_observed_label_accuracy
                metrics["val_acc"] = val_observed_label_accuracy
            else:
                metrics["val_acc"] = val_observed_label_accuracy

        validate_finite_training_metrics(metrics, algorithm=trainer_type.upper())

        execution_workers, model_state_digest = collective_execution_evidence(
            model,
            shard_rows=len(train_dataset),
            input_binding_digest=config.get("_tributo_input_binding_digest"),
            input_rows=input_rows,
            batch_count=local_batch_count,
            collective_steps=collective_steps,
        )
        metrics["execution_workers"] = list(execution_workers)
        metrics["model_state_digest"] = model_state_digest
        metrics["world_size"] = len(execution_workers)
        metrics["state_coordination"] = "all_reduce"
        metrics["collective_backend"] = (
            str(torch.distributed.get_backend())
            if torch.distributed.is_initialized()
            else "none"
        )
        # Ray Train publishes the checkpoint reported by global rank zero.  This is
        # observed implementation evidence, not a value copied from the descriptor.
        metrics["checkpoint_owner_rank"] = 0
        metrics["metric_reducers"] = dict(config.get("_tributo_metric_reducers") or {})

        # Early stopping must never select a checkpoint using NaN or infinity.
        if val_loader is not None and patience is not None:
            if worker_rank == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    stop_after_report = patience_counter >= patience
            stop_after_report = broadcast_bool(stop_after_report, source_rank=0)
            if stop_after_report and worker_rank == 0:
                logger.info("Early stopping at epoch %d", epoch + 1)

        should_report = (
            not resume_enabled
            or (epoch + 1) % checkpoint_interval == 0
            or stop_after_report
            or epoch + 1 == epochs
        )
        if not should_report:
            continue

        rank_rng_states = (
            all_gather_objects(capture_rng_state()) if resume_enabled else ()
        )

        # Rank 0 owns the consolidated checkpoint and complete resume state.
        world_rank = ray.train.get_context().get_world_rank()
        if world_rank == 0:
            from ray.train import Checkpoint

            checkpoint_dir = Path(tempfile.mkdtemp(prefix=f"{trainer_type}_ckpt_"))
            try:
                # Save model
                torch.save(
                    unwrapped_model(model).state_dict(),
                    checkpoint_dir / "model.pt",
                )

                # Save preprocessor
                preprocessor_state = {
                    "features": [f.__dict__ for f in features],
                    "label_encoders": {
                        k: {
                            str(kk): (
                                int(vv)
                                if isinstance(vv, (np.integer, np.int64))
                                else vv
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
                    trainer_type=trainer_type,
                    task_type=(
                        "pu_classification"
                        if trainer_type == "pu"
                        else "classification"
                    ),
                    framework_version=torch.__version__,
                    extra_metadata={
                        "distribution": {
                            "strategy": "ray_train_collective",
                            "world_size": len(execution_workers),
                            "model_state_digest": model_state_digest,
                        },
                        **(
                            {
                                "pu": {
                                    "loss_type": loss_type,
                                    "class_prior": pu_criterion.class_prior,
                                    "beta": pu_criterion.beta,
                                    "gamma": pu_criterion.gamma,
                                }
                            }
                            if pu_mode and pu_criterion is not None
                            else {}
                        ),
                    },
                )
                (checkpoint_dir / "model_config.json").write_text(
                    json.dumps(model_config, ensure_ascii=False, default=str)
                )

                if resume_enabled:
                    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                    (checkpoint_dir / "rng_state.json").write_text(
                        json.dumps(
                            {"rank_states": list(rank_rng_states)},
                            ensure_ascii=False,
                        )
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
                        trainer_type=trainer_type,
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
                            "world_size": len(execution_workers),
                            "distribution_spec_digest": config.get(
                                "_tributo_distribution_spec_digest"
                            ),
                        },
                    )
                    metrics["resume_id"] = envelope.resume_id

                checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
                ray.train.report(metrics, checkpoint=checkpoint)
            finally:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        else:
            ray.train.report(metrics)

        if stop_after_report:
            break

    logger.info("%s training completed for worker", trainer_type.upper())


# ── Orchestration entry points ──


@PublicAPI(stability="beta")
def run_dnn_training_with_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run DNN training, routing the historical nnPU shape to canonical PU.

    Args:
        config: Training configuration dictionary.

    Returns:
        Training result summary dictionary.
    """
    from tributo.training.data_loader import load_ray_dataset_from_config

    # Pydantic validation
    cfg = DNNTrainingConfig.model_validate(config)

    if cfg.loss.type == "nnpu":
        from tributo.training.pu_trainer import run_pu_training_with_config

        pu_config = {
            "data": config.get("data"),
            "features": [item.model_dump() for item in cfg.features],
            "model": cfg.model.model_dump(),
            "pu": {
                "loss_type": "nnpu",
                "class_prior": cfg.pu_learning.class_prior,
                "class_prior_method": cfg.pu_learning.class_prior_method,
                "beta": cfg.pu_learning.beta,
                "gamma": cfg.pu_learning.gamma,
            },
            "training": cfg.training.model_dump(),
            "ray": cfg.ray.model_dump(),
            "resource": cfg.resource.model_dump(),
            "output": cfg.output.model_dump(),
            "label_col": cfg.label_col,
        }
        logger.warning(
            "DNN loss.type='nnpu' is a compatibility alias; routing to the "
            "canonical PU trainer"
        )
        return run_pu_training_with_config(pu_config)

    # Load data
    logger.info("Loading data...")
    ds = load_ray_dataset_from_config(cfg.data)

    # Execute training using DNNTrainerImpl
    trainer = DNNTrainerImpl(
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


@PublicAPI(stability="beta")
def run_dnn_training_from_json(config_path: str) -> dict[str, Any]:
    """Run DNN training from a JSON configuration file.

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
        loaded = json.load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return run_dnn_training_with_config(loaded)


# ── Built-in registration ──

_trainer_spec = build_legacy_spec(
    DNN_DESCRIPTOR,
    trainer_cls=DNNTrainerImpl,
    config_model=DNNTrainingConfig,
)

# Exported for the explicit Beta compatibility API.
trainer_spec = _trainer_spec
