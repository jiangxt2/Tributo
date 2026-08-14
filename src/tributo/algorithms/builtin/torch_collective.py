"""First-party DNN and PU implementations of the collective algorithm SPI."""

from __future__ import annotations

from collections.abc import Mapping
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
from tributo.algorithms.spi import CollectiveAlgorithm
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    DNN_DESCRIPTOR as LEGACY_DNN_DESCRIPTOR,
)
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    PU_DESCRIPTOR as LEGACY_PU_DESCRIPTOR,
)
from tributo.training.base import BaseTrainer
from tributo.util.annotations import DeveloperAPI, PublicAPI

_RAY_TRAIN_INPUT = QualifiedReference.parse(
    "tributo.integrations.algorithm_inputs.ingestion:prepare_ray_train_input"
)
_COLLECTIVE_FACTORY = QualifiedReference.parse(
    "tributo.algorithms.builtin.torch_collective:create_collective_algorithm"
)
_COLLECTIVE_EXPORTER = QualifiedReference.parse(
    "tributo.algorithms.builtin.torch_collective:export_collective_result"
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
    implementation: type[CollectiveAlgorithm],
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
            executable_factory_ref=_COLLECTIVE_FACTORY,
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
            runtime_id="tributo.ray_train_collective",
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
                ExecutionProfile.KUBERNETES,
            ),
            resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
            input_distribution=InputDistribution.SHARDED,
            state_coordination=StateCoordination.ALL_REDUCE,
            policy=CollectivePolicy(
                backend="gloo",
                metric_reducers={
                    "train_loss": MetricReduction.SUM_COUNT,
                    "train_acc": MetricReduction.SUM_COUNT,
                    "val_loss": MetricReduction.SUM_COUNT,
                    "val_acc": MetricReduction.SUM_COUNT,
                },
                checkpoint_owner_rank=0,
                same_world_size_resume=True,
                rank_seeded=True,
            ),
        ),
        is_default=True,
    )


DNN_REGISTRATION = _registration(algorithm="dnn", implementation=DistributedDNN)
PU_REGISTRATION = _registration(algorithm="pu", implementation=DistributedPU)

DNN_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=DNN_REGISTRATION,
    package_name="tributo",
    package_version="1.0.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(ExecutionProfile.LOCAL,),
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
    validated_execution_profiles=(ExecutionProfile.LOCAL,),
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
