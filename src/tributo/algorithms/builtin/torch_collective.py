"""First-party DNN and PU implementations of the collective algorithm SPI."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionResult,
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    CollectivePolicy,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ImplementationDescriptor,
    InputDistribution,
    MetricReduction,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    StateCoordination,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.spi import CollectiveAlgorithm, TorchTrainingRecipe
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    DNN_DESCRIPTOR as LEGACY_DNN_DESCRIPTOR,
)
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    PU_DESCRIPTOR as LEGACY_PU_DESCRIPTOR,
)
from tributo.training.base import BaseTrainer
from tributo.util.annotations import DeveloperAPI, PublicAPI

_COLLECTIVE_CONTRACT = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_TRAIN_COLLECTIVE
]
_RAY_TRAIN_INPUT = QualifiedReference.parse(
    _COLLECTIVE_CONTRACT.worker_input_adapter_ref
)
_COLLECTIVE_FACTORY = QualifiedReference.parse(
    "tributo.algorithms.builtin.torch_collective:create_collective_algorithm"
)
_TORCH_RECIPE_FACTORY = QualifiedReference.parse(
    "tributo.integrations.algorithm_runtimes.torch_recipe:create_torch_recipe_algorithm"
)
_COLLECTIVE_EXPORTER = QualifiedReference.parse(
    "tributo.algorithms.builtin.torch_collective:export_collective_result"
)


def _transform_dnn_batch(
    batch: Mapping[str, Any],
    *,
    transformer_state: Mapping[str, Any],
    label_name: str,
) -> dict[str, Any]:
    import numpy as np

    from tributo.training.features.transformer import FeatureTransformer

    transformer = FeatureTransformer.from_state(dict(transformer_state))
    feature_names = [feature.name for feature in transformer.features]
    transformed = transformer.transform(
        {name: np.asarray(batch[name]) for name in feature_names}
    )
    transformed[label_name] = np.asarray(batch[label_name], dtype=np.float32)
    return transformed


class DNNTrainingRecipe(TorchTrainingRecipe):
    """First-party DNN semantics over the common Ray-owned Torch loop."""

    _trainer_type = "dnn"

    def __init__(self) -> None:
        self._transformer_state: dict[str, Any] | None = None

    def model_factory(self, config: Mapping[str, Any]) -> object:
        from tributo.training.features.column_types import features_from_dicts
        from tributo.training.models.dnn import DNNModel

        payload = dict(config)
        raw_features = payload.pop("features", None)
        if not isinstance(raw_features, (list, tuple)):
            raise AlgorithmConfigurationError("DNN Recipe requires feature config")
        return DNNModel(features_from_dicts(list(raw_features)), **payload)

    def loss_factory(self, config: Mapping[str, Any]) -> object:
        import torch

        from tributo.training.losses.focal_loss import FocalLoss

        loss_type = str(config.get("type", "bce"))
        if loss_type == "focal":
            return FocalLoss(
                alpha=float(config.get("alpha", 0.25)),
                gamma=float(config.get("gamma", 2.0)),
            )
        if loss_type != "bce":
            raise AlgorithmConfigurationError(
                "DNN Recipe supports only bce or focal loss"
            )
        return torch.nn.BCEWithLogitsLoss()

    def optimizer_factory(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> object:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise AlgorithmConfigurationError("DNN model must be torch.nn.Module")
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("learning_rate", 0.001)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    def metric_factories(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        del config

        def accuracy(predictions: object, targets: object) -> object:
            import torch

            if not isinstance(predictions, torch.Tensor) or not isinstance(
                targets, torch.Tensor
            ):
                raise TypeError("DNN accuracy requires Tensor values")
            return (torch.sigmoid(predictions) >= 0.5) == targets.bool()

        return {"train_acc": accuracy}

    def forward(self, model: object, features: object) -> object:
        if not callable(model) or not isinstance(features, Mapping):
            raise TypeError("DNN Recipe requires a callable model and feature mapping")
        return model(dict(features))

    def _bind_datasets(
        self,
        datasets: Mapping[str, object],
        *,
        config: Mapping[str, Any],
        worker_count: int,
        resume_from: str | None,
    ) -> Mapping[str, object]:
        from tributo.training.checkpoint import checkpoint_directory
        from tributo.training.features.column_types import features_from_dicts
        from tributo.training.features.transformer import FeatureTransformer

        if len(datasets) != 1:
            raise AlgorithmConfigurationError(
                "DNN Recipe requires one canonical Ray Dataset"
            )
        dataset: Any = next(iter(datasets.values()))
        training = config.get("training", {})
        if not isinstance(training, Mapping):
            raise AlgorithmConfigurationError("DNN training config must be a mapping")
        val_size = float(training.get("val_size", 0.2))
        seed = int(training.get("seed", 42))
        total_rows = int(dataset.count())
        required_rows = worker_count * (2 if val_size > 0 else 1)
        if total_rows < required_rows:
            raise AlgorithmConfigurationError(
                f"DNN has {total_rows} rows but requires at least {required_rows}"
            )
        if val_size > 0:
            train, validation = dataset.randomize_block_order(
                seed=seed
            ).split_proportionately([1.0 - val_size])
            raw_datasets = {"train": train, "val": validation}
        else:
            raw_datasets = {"train": dataset}
        if any(int(value.count()) < worker_count for value in raw_datasets.values()):
            raise AlgorithmConfigurationError(
                "DNN split must provide at least one row to every worker"
            )
        raw_features = config.get("features", [])
        if not isinstance(raw_features, (list, tuple)) or not raw_features:
            raise AlgorithmConfigurationError("DNN requires feature declarations")
        features = features_from_dicts(list(raw_features))
        if resume_from is not None:
            with checkpoint_directory(resume_from) as checkpoint_dir:
                transformer = FeatureTransformer.load(
                    checkpoint_dir / "preprocessor.json"
                )
        else:
            from tributo.integrations.algorithm_runtimes.ray_data_config import (
                fit_dnn_feature_transformer,
            )

            transformer = fit_dnn_feature_transformer(
                raw_datasets["train"],
                features,
            )
        if transformer.features != features:
            raise AlgorithmConfigurationError(
                "DNN preprocessing features do not match the current config"
            )
        self._transformer_state = transformer.to_state()
        label_name = str(config.get("label_col", "label"))
        return {
            name: value.map_batches(
                _transform_dnn_batch,
                batch_format="numpy",
                fn_kwargs={
                    "transformer_state": self._transformer_state,
                    "label_name": label_name,
                },
            )
            for name, value in raw_datasets.items()
        }

    def _lower_worker_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._transformer_state is None:
            raise AlgorithmConfigurationError("DNN preprocessing was not prepared")
        lowered = dict(config)
        training = dict(config.get("training") or {})
        model = dict(config.get("model") or {})
        model["features"] = list(config.get("features") or ())
        lowered["model"] = model
        lowered["optimizer"] = {
            "learning_rate": training.get("learning_rate", 0.001),
            "weight_decay": training.get("weight_decay", 0.0),
        }
        lowered["metrics"] = {}
        lowered["training"] = {
            "epochs": training.get("epochs", 10),
            "batch_size": training.get("batch_size", 256),
            "seed": training.get("seed", 42),
            "early_stopping_patience": training.get("early_stopping_patience"),
        }
        lowered["_tributo_dnn_features"] = list(config.get("features") or ())
        lowered["_tributo_dnn_model"] = dict(config.get("model") or {})
        return lowered

    def _prepare_batch(
        self,
        batch: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str,
        weight_name: str | None,
        config: Mapping[str, Any],
    ) -> tuple[object, object, object | None, int]:
        import torch

        del config
        if weight_name is not None or not isinstance(batch, Mapping):
            raise AlgorithmConfigurationError(
                "DNN Recipe requires an unweighted mapping batch"
            )
        labels = batch.get(label_name)
        if not isinstance(labels, torch.Tensor):
            raise AlgorithmConfigurationError("DNN label batch must be a Tensor")
        rows = int(labels.shape[0])
        feature_config = self._transformer_state or {}
        sparse_names = {
            str(item["name"])
            for item in feature_config.get("features", [])
            if item.get("type") == "sparse"
        }
        features: dict[str, Any] = {}
        for name in feature_names:
            value = batch.get(name)
            if not isinstance(value, torch.Tensor) or int(value.shape[0]) != rows:
                raise AlgorithmConfigurationError(
                    f"DNN feature {name!r} must share the batch dimension"
                )
            features[name] = value.long() if name in sparse_names else value.float()
        return features, labels.float(), None, rows

    def _checkpoint_contract(
        self,
        *,
        config: Mapping[str, Any],
        feature_count: int,
        output_shape: tuple[int, ...],
        framework_version: str,
        model_digest: str,
        world_size: int,
    ) -> dict[str, Any]:
        from tributo.training.dnn_trainer import build_export_checkpoint_config

        del feature_count, output_shape
        return build_export_checkpoint_config(
            list(config.get("_tributo_dnn_features") or ()),
            dict(config.get("_tributo_dnn_model") or {}),
            trainer_type="dnn",
            task_type="classification",
            framework_version=framework_version,
            extra_metadata={
                "distribution": {
                    "strategy": "ray_train_collective",
                    "world_size": world_size,
                    "model_state_digest": model_digest,
                }
            },
        )

    def _write_checkpoint_artifacts(self, checkpoint_dir: Path) -> tuple[str, ...]:
        from tributo.training.features.transformer import FeatureTransformer

        if self._transformer_state is None:
            raise AlgorithmConfigurationError("DNN preprocessing state is missing")
        directory = Path(checkpoint_dir)
        FeatureTransformer.from_state(self._transformer_state).save(
            directory / "preprocessor.json"
        )
        return ("preprocessor.json",)

    def _validate_checkpoint_artifacts(self, checkpoint_dir: Path) -> None:
        from tributo.training.features.transformer import FeatureTransformer

        restored = FeatureTransformer.load(Path(checkpoint_dir) / "preprocessor.json")
        if (
            self._transformer_state is None
            or restored.to_state() != self._transformer_state
        ):
            raise AlgorithmConfigurationError(
                "DNN resume preprocessing state does not match the current input"
            )


class _TorchCollectiveAlgorithm(CollectiveAlgorithm):
    """Shared constrained adapter over the DNN/PU DDP worker kernel."""

    trainer_type: str

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self._validate_runtime_config()

    def _validate_runtime_config(self) -> None:
        """Keep legacy Ray fields consistent with the formal execution request."""
        loss_config = self.plan.algorithm_config.get("loss", {})
        pu_learning = self.plan.algorithm_config.get("pu_learning", {})
        if self.trainer_type == "dnn" and (
            (isinstance(loss_config, Mapping) and loss_config.get("type") == "nnpu")
            or (
                isinstance(pu_learning, Mapping)
                and bool(pu_learning.get("enabled", False))
            )
        ):
            raise AlgorithmConfigurationError(
                "DNN nnPU is a compatibility alias and must resolve to the "
                "canonical PU algorithm before collective execution"
            )
        ray_config = self.plan.algorithm_config.get("ray", {})
        if not isinstance(ray_config, Mapping):
            raise AlgorithmConfigurationError("ray config must be a mapping")
        if (
            "num_workers" in ray_config
            and int(ray_config["num_workers"]) != self.plan.runtime.worker_count
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.num_workers must match "
                "ExecutionRequest.worker_count"
            )
        if "use_gpu" in ray_config and bool(ray_config["use_gpu"]) != (
            self.plan.runtime.num_gpus > 0
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.use_gpu must match the formal per-worker "
                "GPU request"
            )
        if int(ray_config.get("max_failures", 0)) != 0:
            raise AlgorithmConfigurationError(
                "formal collective training requires ray.max_failures=0 until "
                "failure-injection and replay semantics pass their own gate"
            )
        model_config = self.plan.algorithm_config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise AlgorithmConfigurationError("model config must be a mapping")
        if self.plan.runtime.worker_count > 1 and bool(
            model_config.get("use_batch_norm", False)
        ):
            raise AlgorithmConfigurationError(
                "distributed DNN/PU currently rejects model.use_batch_norm=true; "
                "per-rank BatchNorm buffers do not prove one synchronized model"
            )
        resume = ray_config.get("resume", {})
        if not isinstance(resume, Mapping):
            raise AlgorithmConfigurationError("ray.resume config must be a mapping")
        configured_checkpoint = resume.get("checkpoint_path")
        if configured_checkpoint is not None and (
            configured_checkpoint != self.plan.runtime.resume_from
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.resume.checkpoint_path must match "
                "ExecutionRequest.resume_from"
            )

    def build_model(self, config: Mapping[str, Any]) -> object:
        """Construct the declared DNN architecture for conformance tooling."""
        from tributo.training.dnn_trainer import (
            FeatureItemConfig,
            build_features_from_config,
        )
        from tributo.training.models.dnn import DNNModel

        feature_config = [
            FeatureItemConfig.model_validate(item)
            for item in config.get("features", [])
        ]
        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise AlgorithmConfigurationError("model config must be a mapping")
        return DNNModel(
            build_features_from_config(feature_config),
            **dict(model_config),
        )

    def build_optimizer(self, model: object, config: Mapping[str, Any]) -> object:
        """Build the same Adam optimizer used by the shared worker kernel."""
        import torch

        if not isinstance(model, torch.nn.Module):
            raise AlgorithmConfigurationError(
                "collective model must be torch.nn.Module"
            )
        training = config.get("training", {})
        if not isinstance(training, Mapping):
            raise AlgorithmConfigurationError("training config must be a mapping")
        return torch.optim.Adam(
            model.parameters(),
            lr=float(training.get("learning_rate", 0.001)),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )

    def build_loss(self, config: Mapping[str, Any]) -> object:
        """Build the algorithm-specific loss contract for conformance tooling."""
        import torch

        if self.trainer_type == "dnn":
            return torch.nn.BCEWithLogitsLoss()
        from tributo.training.losses.pu_loss import PULoss

        pu = config.get("pu", {})
        if not isinstance(pu, Mapping) or pu.get("class_prior") is None:
            raise AlgorithmConfigurationError(
                "standalone PU loss construction requires class_prior"
            )
        return PULoss(
            class_prior=float(pu["class_prior"]),
            beta=float(pu.get("beta", 0.0)),
            gamma=float(pu.get("gamma", 1.0)),
            loss_type=str(pu.get("loss_type", "nnpu")),
        )

    def checkpoint_state(self, model: object, optimizer: object) -> Mapping[str, Any]:
        """Return bounded model and optimizer states for checkpoint ownership."""
        from tributo.training.distributed_torch import unwrapped_model

        state_dict = getattr(unwrapped_model(model), "state_dict", None)
        optimizer_state = getattr(optimizer, "state_dict", None)
        if not callable(state_dict) or not callable(optimizer_state):
            raise AlgorithmConfigurationError(
                "collective checkpoint objects do not expose state_dict"
            )
        return {"model": state_dict(), "optimizer": optimizer_state()}

    def _worker_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        worker_config = dict(config)
        worker_config.pop("data", None)
        worker_config.pop("output", None)
        worker_config["_tributo_input_binding_digest"] = (
            self.plan.input_descriptor.binding_digest
        )
        worker_config["_tributo_distribution_spec_digest"] = (
            self.plan.runtime.distribution_digest
        )
        ray_config = worker_config.get("ray", {})
        resume = ray_config.get("resume", {}) if isinstance(ray_config, Mapping) else {}
        worker_config["resume"] = dict(resume) if isinstance(resume, Mapping) else {}
        if self.plan.runtime.resume_from is not None:
            worker_config["resume"]["enabled"] = True
            worker_config["resume"]["checkpoint_path"] = self.plan.runtime.resume_from
        return worker_config


@PublicAPI(stability="alpha")
class DistributedDNN(_TorchCollectiveAlgorithm):
    """Supervised DNN backed by Ray Train and PyTorch DDP."""

    trainer_type = "dnn"

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        """Split train/validation globally before Ray Train assigns shards."""
        if len(datasets) != 1:
            raise AlgorithmConfigurationError(
                "DNN collective input requires one canonical Ray Dataset"
            )
        dataset: Any = next(iter(datasets.values()))
        randomize = getattr(dataset, "randomize_block_order", None)
        split = getattr(dataset, "split_proportionately", None)
        count = getattr(dataset, "count", None)
        if not callable(randomize) or not callable(split) or not callable(count):
            raise AlgorithmConfigurationError(
                "DNN collective input must be a Ray Dataset"
            )
        training = self.plan.algorithm_config.get("training", {})
        if not isinstance(training, Mapping):
            raise AlgorithmConfigurationError("training config must be a mapping")
        val_size = float(training.get("val_size", 0.2))
        seed = int(training.get("seed", 42))
        total_rows = int(count())
        minimum_rows = self.plan.runtime.worker_count * (2 if val_size > 0 else 1)
        if total_rows < minimum_rows:
            raise AlgorithmConfigurationError(
                "distributed DNN has insufficient rows for every worker and its "
                f"global validation split; rows={total_rows}, required={minimum_rows}"
            )
        if val_size <= 0:
            return {"train": dataset}
        train, validation = randomize(seed=seed).split_proportionately([1.0 - val_size])
        if (
            int(train.count()) < self.plan.runtime.worker_count
            or int(validation.count()) < self.plan.runtime.worker_count
        ):
            raise AlgorithmConfigurationError(
                "global DNN train/validation split cannot provide every worker "
                "with a non-empty shard"
            )
        return {"train": train, "val": validation}

    def train_loop_per_worker(self, config: Mapping[str, Any]) -> None:
        """Delegate to the shared DNN/PU DDP worker kernel."""
        from tributo.training.dnn_trainer import dnn_train_loop_per_worker

        worker_config = self._worker_config(config)
        worker_config["trainer_type"] = "dnn"
        dnn_train_loop_per_worker(worker_config)


@PublicAPI(stability="alpha")
class DistributedPU(_TorchCollectiveAlgorithm):
    """PU domain adapter backed by the same DNN/PyTorch DDP kernel."""

    trainer_type = "pu"

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        """Create a global stratified PU split before Ray Train sharding."""
        if len(datasets) != 1:
            raise AlgorithmConfigurationError(
                "PU collective input requires one canonical Ray Dataset"
            )
        dataset: Any = next(iter(datasets.values()))
        required_methods = (
            getattr(dataset, name, None)
            for name in ("filter", "count", "random_shuffle", "split_proportionately")
        )
        if not all(callable(method) for method in required_methods):
            raise AlgorithmConfigurationError(
                "PU collective input must be a Ray Dataset"
            )
        from tributo.exceptions import JobConfigurationError
        from tributo.training.pu_trainer import split_pu_ray_datasets

        label_col = str(self.plan.algorithm_config.get("label_col", "label"))
        training = self.plan.algorithm_config.get("training", {})
        if not isinstance(training, Mapping):
            raise AlgorithmConfigurationError("training config must be a mapping")
        try:
            return split_pu_ray_datasets(
                dataset,
                label_col=label_col,
                worker_count=self.plan.runtime.worker_count,
                val_size=float(training.get("val_size", 0.2)),
                seed=int(training.get("seed", 42)),
            )
        except JobConfigurationError as exc:
            raise AlgorithmConfigurationError(str(exc)) from exc

    def train_loop_per_worker(self, config: Mapping[str, Any]) -> None:
        """Adapt PU configuration and invoke the shared DDP kernel."""
        from tributo.training.pu_trainer import pu_train_loop_per_worker

        pu_train_loop_per_worker(self._worker_config(config))


@DeveloperAPI
def create_collective_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> CollectiveAlgorithm:
    """Instantiate only a declared first-party CollectiveAlgorithm class."""
    del artifacts
    if implementation not in {DistributedDNN, DistributedPU}:
        raise AlgorithmConfigurationError(
            "collective implementation reference is not a first-party SPI class"
        )
    return implementation(plan)


@DeveloperAPI
def export_collective_result(
    *,
    result: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    """Publish one consolidated Ray Train checkpoint through BundleExportService."""
    output = plan.algorithm_config.get("output", {})
    if not isinstance(output, Mapping):
        raise AlgorithmConfigurationError("output config must be a mapping")
    bundle_uri = output.get("bundle_uri")
    if not isinstance(bundle_uri, str) or not bundle_uri:
        raise AlgorithmConfigurationError(
            "formal DNN/PU training requires output.bundle_uri"
        )
    if plan.resolution.algorithm == "dnn":
        from tributo.training.dnn_trainer import DNNTrainerImpl, DNNTrainingConfig

        dnn_config = DNNTrainingConfig.model_validate(plan.algorithm_config)
        trainer: BaseTrainer = DNNTrainerImpl(
            datasets={},
            config=dict(plan.algorithm_config),
            _validated_config=dnn_config,
        )
    elif plan.resolution.algorithm == "pu":
        from tributo.training.pu_trainer import PUTrainerImpl, PUTrainingConfig

        pu_config = PUTrainingConfig.model_validate(plan.algorithm_config)
        trainer = PUTrainerImpl(
            datasets={},
            config=dict(plan.algorithm_config),
            _validated_config=pu_config,
        )
    else:
        raise AlgorithmConfigurationError(
            "collective exporter received an unsupported algorithm identity"
        )
    from tributo.training.lifecycle import publish_existing_training_result

    published = publish_existing_training_result(
        trainer,
        result,
        bundle_uri=bundle_uri,
        run_id=run_id,
    )
    metrics = getattr(result, "metrics", None) or {}
    portable_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"execution_workers"}
        and isinstance(value, (str, int, float, bool, list, dict, type(None)))
    }
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics=portable_metrics,
        outputs={
            "bundle_id": published["bundle_id"],
            "bundle_uri": published["canonical_uri"],
            "execution_id": published["execution_id"],
            "manifest_sha256": published["manifest_sha256"],
        },
    )


def _registration(
    *,
    algorithm: str,
    implementation: type,
    executable_factory: QualifiedReference = _COLLECTIVE_FACTORY,
) -> AlgorithmRegistration:
    legacy = LEGACY_DNN_DESCRIPTOR if algorithm == "dnn" else LEGACY_PU_DESCRIPTOR
    return AlgorithmRegistration(
        spec=legacy.registration.spec,
        implementation=ImplementationDescriptor(
            implementation_id=f"tributo.{algorithm}.ray_train_collective",
            version="1.0.0",
            execution_mode=ExecutionMode.COLLECTIVE,
            implementation_ref=QualifiedReference.parse(
                f"{implementation.__module__}:{implementation.__qualname__}"
            ),
            executable_factory_ref=executable_factory,
            operations=(AlgorithmOperation.FIT,),
            input_compatibility=BackendInputCompatibility(
                accepted_input_views=("ray_data",),
                accepted_ingestion_engines=("tributo.ray_data",),
                required_input_capabilities=("shardable",),
                supported_explicit_adapters=(_RAY_TRAIN_INPUT,),
                distribution_policy=(RuntimeTopology.RAY_TRAIN_COLLECTIVE,),
            ),
            distribution="torch",
            framework="pytorch",
            allowed_config_keys=(
                "data",
                "features",
                "label_col",
                "loss",
                "model",
                "output",
                "pu",
                "pu_learning",
                "ray",
                "resource",
                "training",
            ),
            runtime_id=_COLLECTIVE_CONTRACT.runtime_id,
            worker_input_adapter_ref=_RAY_TRAIN_INPUT,
            exporter_ref=_COLLECTIVE_EXPORTER,
            flavor_id="onnx-runtime-v1",
        ),
        environment=EnvironmentSpec(
            environment_id=f"tributo.{algorithm}.collective.v1",
            dependencies=(
                "onnx>=1.16.0",
                "onnxruntime>=1.20.0",
                "ray==2.55.1",
                "torch>=2.5.0",
            ),
        ),
        distribution_spec=DistributionSpec(
            strategy=DistributionStrategy.RAY_TRAIN_COLLECTIVE,
            supported_worker_range=WorkerRange(1, 1024),
            supported_execution_profiles=(
                ExecutionProfile.LOCAL,
                ExecutionProfile.CLUSTER,
            ),
            resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
            input_distribution=InputDistribution.SHARDED,
            state_coordination=StateCoordination.ALL_REDUCE,
            policy=CollectivePolicy(
                backend="gloo",
                metric_reducers={
                    "train_loss": MetricReduction.SUM_COUNT,
                    "train_acc": MetricReduction.SUM_COUNT,
                    **(
                        {
                            "val_loss": MetricReduction.SUM_COUNT,
                            "val_acc": MetricReduction.SUM_COUNT,
                        }
                        if algorithm == "pu"
                        else {}
                    ),
                },
                checkpoint_owner_rank=0,
                same_world_size_resume=True,
                rank_seeded=True,
            ),
        ),
        is_default=True,
    )


DNN_REGISTRATION = _registration(
    algorithm="dnn",
    implementation=DNNTrainingRecipe,
    executable_factory=_TORCH_RECIPE_FACTORY,
)
PU_REGISTRATION = _registration(algorithm="pu", implementation=DistributedPU)

DNN_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=DNN_REGISTRATION,
    package_name="tributo",
    package_version="1.0.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(
        ExecutionProfile.LOCAL,
        ExecutionProfile.CLUSTER,
    ),
    limitations=(
        "CPU/Gloo is supported; GPU/NCCL requires a separate gate.",
        "Multi-worker BatchNorm is rejected until synchronized BatchNorm is gated.",
        "Automatic worker retries are rejected until failure injection is gated.",
    ),
)
PU_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=PU_REGISTRATION,
    package_name="tributo",
    package_version="1.0.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(
        ExecutionProfile.LOCAL,
        ExecutionProfile.CLUSTER,
    ),
    limitations=(
        "CPU/Gloo is supported; GPU/NCCL requires a separate gate.",
        "Multi-worker BatchNorm is rejected until synchronized BatchNorm is gated.",
        "Automatic worker retries are rejected until failure injection is gated.",
    ),
)


__all__ = [
    "DistributedDNN",
    "DistributedPU",
    "create_collective_algorithm",
    "export_collective_result",
]
