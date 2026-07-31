"""DNN distributed trainer based on Ray Train TorchTrainer.

Supports Sparse/Dense features, PU Learning, Focal Loss and other identity mining scenarios.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import Field

from tributo._common.config import StrictConfigModel
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    DataLoadingMode,
    ProblemType,
    ResourceHints,
)
from tributo.training.base import BaseTrainer
from tributo.training.features.column_types import (
    DenseFeat,
    SparseFeat,
    features_from_dicts,
)
from tributo.training.registry import register
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


class PULearningConfig(StrictConfigModel):
    """PU Learning configuration."""

    enabled: bool = Field(default=False, description="Whether to enable PU Learning")
    class_prior: Optional[float] = Field(
        default=None,
        description="Positive class proportion, None for automatic estimation",
    )
    class_prior_method: str = Field(
        default="simple",
        description="Class prior estimation method: simple",
    )
    beta: float = Field(default=0.0, description="Non-negative constraint threshold")
    gamma: float = Field(default=1.0, description="Negative risk scaling factor")


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
    """Ray cluster configuration."""

    num_workers: int = Field(default=2, ge=1)
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


class DNNOutputConfig(StrictConfigModel):
    """Output configuration."""

    onnx_path: Optional[str] = None
    onnx_opset: int = Field(default=12, ge=1)
    metrics_path: Optional[str] = None
    preprocessor_path: Optional[str] = None


class DNNTrainingConfig(StrictConfigModel):
    """Complete configuration for DNN distributed training."""

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
    output: DNNOutputConfig = Field(default_factory=DNNOutputConfig)
    label_col: str = Field(default="label", description="Label column name")


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


# ── DNN BaseTrainer implementation ──


@PublicAPI(stability="beta")
class DNNTrainerImpl(BaseTrainer):
    """DNN distributed trainer, the deep learning implementation of BaseTrainer.

    Supports Sparse/Dense features, PU Learning, Focal Loss and other identity mining scenarios.
    """

    def __init__(
        self,
        datasets: dict[str, ray.data.Dataset],
        config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        *,
        _validated_config: DNNTrainingConfig | None = None,
    ) -> None:
        super().__init__(datasets, config, run_config)
        self._train_config = _validated_config or DNNTrainingConfig.model_validate(
            config
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
        import ray
        import ray.train
        from ray.train import FailureConfig, RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        cfg = self._train_config

        # Prepare training config
        train_loop_config = {
            "features": [f.__dict__ for f in self._features],
            "label_col": cfg.label_col,
            "model": cfg.model.model_dump(),
            "loss": cfg.loss.model_dump(),
            "pu_learning": cfg.pu_learning.model_dump(),
            "training": cfg.training.model_dump(),
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
                name="tributo-dnn",
                storage_path=storage_path,
                # resilience-only: retries restart from scratch, not from checkpoint
                failure_config=FailureConfig(max_failures=cfg.ray.max_failures),
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
                json.dumps(metrics, indent=2, ensure_ascii=False, default=str)
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
    import torch

    inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
    labels = batch["label"].to(device)

    logits = model(inputs)
    loss = criterion(logits, labels)

    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.size(0)

    return loss, correct, total


# ── Training loop (Ray worker level) ──


def dnn_train_loop_per_worker(config: dict[str, Any]) -> None:
    """DNN training main loop executed by each Ray worker.

    Args:
        config: Training configuration dictionary.
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

    from tributo.training.features.column_types import (
        features_from_dicts,
    )
    from tributo.training.features.dataset import IdentityDataset
    from tributo.training.features.transformer import FeatureTransformer
    from tributo.training.losses.focal_loss import FocalLoss
    from tributo.training.losses.pu_loss import PULoss
    from tributo.training.models.dnn import DNNModel
    from tributo.training.priors import estimate_class_prior

    # Configure logger inside worker
    logger = logging.getLogger(__name__)

    # Parse config
    features = features_from_dicts(config["features"])

    label_col = config.get("label_col", "label")
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    pu_cfg = config.get("pu_learning", {})
    training_cfg = config.get("training", {})

    # Get data (StreamSplitDataIterator, must collect via iter_batches)
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    # Convert to pandas and preprocess
    import pandas as pd

    train_batches = list(
        train_ds.iter_batches(
            batch_size=None,
            batch_format="pandas",
            prefetch_batches=1,
        )
    )
    train_df = pd.concat(train_batches, ignore_index=True)

    val_df = None
    if val_ds is not None:
        val_batches = list(
            val_ds.iter_batches(
                batch_size=None,
                batch_format="pandas",
                prefetch_batches=1,
            )
        )
        if val_batches:
            val_df = pd.concat(val_batches, ignore_index=True)

    # Prepare data
    feature_names = [f.name for f in features]
    train_data = {name: train_df[name].values for name in feature_names}
    train_labels = train_df[label_col].values.astype(np.float32)

    val_data = None
    val_labels = None
    if val_df is not None:
        val_data = {name: val_df[name].values for name in feature_names}
        val_labels = val_df[label_col].values.astype(np.float32)

    # Fit preprocessor
    transformer = FeatureTransformer(features)
    train_processed = transformer.fit_transform(train_data)

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
    train_loader = DataLoader(
        torch_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
    model = DNNModel(features, **model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Configure loss function
    loss_type = loss_cfg.get("type", "bce")
    if loss_type == "nnpu":
        # PU Learning
        if pu_cfg.get("enabled", False):
            class_prior = pu_cfg.get("class_prior")
            if class_prior is None:
                # Auto-estimate
                positive_count = int(train_labels.sum())
                total_count = len(train_labels)
                prior_method = pu_cfg.get("class_prior_method", "label_frequency")
                class_prior = estimate_class_prior(
                    positive_count,
                    total_count,
                    method=prior_method,
                )
                logger.info(
                    "Estimated class_prior: %.4f (method=%s)", class_prior, prior_method
                )

            criterion = PULoss(
                class_prior=class_prior,
                beta=pu_cfg.get("beta", 0.0),
                gamma=pu_cfg.get("gamma", 1.0),
                loss_type="nnpu",
            )
        else:
            criterion = PULoss(
                class_prior=0.5,
                beta=pu_cfg.get("beta", 0.0),
                gamma=pu_cfg.get("gamma", 1.0),
            )
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

    # Training loop
    epochs = training_cfg.get("epochs", 10)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = training_cfg.get("early_stopping_patience")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            loss, correct, total = _forward_step(model, batch, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * total
            train_correct += correct
            train_total += total

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    loss, correct, total = _forward_step(
                        model, batch, criterion, device
                    )
                    val_loss += loss.item() * total
                    val_correct += correct
                    val_total += total

            val_loss /= val_total
            val_acc = val_correct / val_total

            # Early stopping check
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        # Report metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }
        if val_loader is not None:
            metrics["val_loss"] = val_loss
            metrics["val_acc"] = val_acc

        # Save checkpoint (rank 0 reports full checkpoint, other ranks report metrics only)
        world_rank = ray.train.get_context().get_world_rank()
        if world_rank == 0:
            from ray.train import Checkpoint

            checkpoint_dir = Path(tempfile.mkdtemp(prefix="dnn_ckpt_"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")

            # Save preprocessor
            preprocessor_state = {
                "features": [f.__dict__ for f in features],
                "label_encoders": {
                    k: {
                        str(kk): int(vv)
                        if isinstance(vv, (np.integer, np.int64))
                        else vv
                        for kk, vv in v.items()
                    }
                    for k, v in transformer.label_encoders.items()
                },
                "norm_params": transformer.norm_params,
            }
            (checkpoint_dir / "preprocessor.json").write_text(
                json.dumps(preprocessor_state, ensure_ascii=False, default=str)
            )

            checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
            ray.train.report(metrics, checkpoint=checkpoint)
        else:
            ray.train.report(metrics)

    logger.info("Training completed for worker")


# ── Orchestration entry points ──


@PublicAPI(stability="beta")
def run_dnn_training_with_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run DNN distributed training from a configuration dictionary.

    Args:
        config: Training configuration dictionary.

    Returns:
        Training result summary dictionary.
    """
    from tributo.training.data_loader import load_ray_dataset_from_config

    # Pydantic validation
    cfg = DNNTrainingConfig.model_validate(config)

    # Load data
    logger.info("Loading data...")
    ds = load_ray_dataset_from_config(cfg.data)

    # Execute training using DNNTrainerImpl
    trainer = DNNTrainerImpl(
        datasets={"train": ds},
        config=config,
        _validated_config=cfg,
    )
    onnx_path = cfg.output.onnx_path or "model_output"
    return trainer.run(output_path=onnx_path)


@PublicAPI(stability="beta")
def run_dnn_training_from_json(config_path: str) -> dict[str, Any]:
    """Run DNN distributed training from a JSON configuration file.

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

register(
    AlgorithmSpec(
        name="dnn",
        trainer_cls=DNNTrainerImpl,
        problem_types=(ProblemType.BINARY_CLASSIFICATION,),
        data_modality=("tabular",),
        extras_group="identity",
        data_loading=DataLoadingMode.CANONICAL_DRIVER,
        resource_hints=ResourceHints(gpu_required=False),
        config_model=DNNTrainingConfig,
    )
)
