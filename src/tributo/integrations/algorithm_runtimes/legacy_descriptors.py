"""Lightweight first-party descriptors for the bounded legacy Trainer adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    ImplementationDescriptor,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
)
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    DataLoadingMode,
    ProblemType,
    ResourceHints,
)
from tributo.util.annotations import DeveloperAPI

_LEGACY_FACTORY = QualifiedReference.parse(
    "tributo.integrations.algorithm_runtimes.legacy_trainer:create_executable"
)
_MATERIALIZED_INPUT_ADAPTER = QualifiedReference.parse(
    "tributo.integrations.algorithm_inputs.ingestion:prepare_ingestion_input"
)


@DeveloperAPI
@dataclass(frozen=True)
class LegacyTrainerDescriptor:
    """Bind one portable contract to Worker-only legacy compatibility refs."""

    registration: AlgorithmRegistration
    trainer_ref: QualifiedReference
    config_model_ref: QualifiedReference
    limitations: tuple[str, ...]
    stability: Literal["beta"] = "beta"
    tested: bool = False
    supported: bool = False
    native_migration_complete: bool = False

    def __post_init__(self) -> None:
        if self.stability != "beta":
            raise ValueError("legacy Trainer descriptors must remain Beta")
        if self.registration.implementation.implementation_ref != self.trainer_ref:
            raise ValueError("legacy descriptor Trainer references must agree")
        if (
            self.registration.implementation.execution_mode
            is not ExecutionMode.LEGACY_TRAINER
        ):
            raise ValueError("legacy descriptor requires execution_mode=legacy_trainer")
        if self.supported and not self.tested:
            raise ValueError("supported legacy descriptors must also be tested")
        if self.native_migration_complete:
            raise ValueError(
                "LegacyTrainerDescriptor cannot represent a completed native migration"
            )
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def name(self) -> str:
        """Return the canonical algorithm identity."""
        return self.registration.spec.name


def build_legacy_spec(
    descriptor: LegacyTrainerDescriptor,
    *,
    trainer_cls: type,
    config_model: type,
) -> AlgorithmSpec:
    """Hydrate the Beta TrainerSpec view without duplicating algorithm facts."""
    return replace(
        descriptor.registration.spec,
        trainer_cls=trainer_cls,
        config_model=config_model,
    )


def _compatibility(*topologies: RuntimeTopology) -> BackendInputCompatibility:
    return BackendInputCompatibility(
        accepted_input_views=("ray_data",),
        accepted_ingestion_engines=("tributo.ray_data",),
        required_input_capabilities=("materializable",),
        supported_explicit_adapters=(_MATERIALIZED_INPUT_ADAPTER,),
        distribution_policy=topologies,
    )


def _descriptor(
    *,
    name: str,
    trainer_ref: str,
    config_model_ref: str,
    extras_group: str,
    problem_types: tuple[ProblemType, ...],
    capabilities: tuple[Capability, ...],
    data_loading: DataLoadingMode,
    learning_paradigm: str,
    model_family: str,
    allowed_config_keys: tuple[str, ...],
    environment_dependencies: tuple[str, ...],
    framework: str,
    topology: RuntimeTopology,
    native_mode: ExecutionMode,
    framework_parallelism: int = 1,
    limitations: tuple[str, ...] = (),
) -> LegacyTrainerDescriptor:
    trainer_reference = QualifiedReference.parse(trainer_ref)
    spec = AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        problem_types=problem_types,
        data_modality=("tabular",),
        extras_group=extras_group,
        capabilities=capabilities,
        data_loading=data_loading,
        resource_hints=ResourceHints(gpu_required=False),
        operations=(AlgorithmOperation.FIT.value,),
        learning_paradigm=learning_paradigm,
        model_family=model_family,
        data_modalities=("tabular",),
        lifecycle_kind="bounded_training",
        allowed_execution_modes=tuple(
            sorted((ExecutionMode.LEGACY_TRAINER.value, native_mode.value))
        ),
        config_contract_ref=f"tributo.algorithm-config.{name}.v1",
        input_contract_ref="tributo.algorithm-input.ray-tabular.v1",
        output_contract_ref="tributo.algorithm-result.execution-only.v1",
    )
    runtime = RuntimeBinding(
        runtime_id="tributo.ray_task",
        worker_input_adapter_ref=_MATERIALIZED_INPUT_ADAPTER,
        topology=topology,
        framework_parallelism=framework_parallelism,
        num_cpus=0 if topology is RuntimeTopology.FRAMEWORK_MANAGED else 1,
    )
    return LegacyTrainerDescriptor(
        registration=AlgorithmRegistration(
            spec=spec,
            implementation=ImplementationDescriptor(
                implementation_id=f"tributo.{name}.legacy_trainer",
                version="1.0.0",
                execution_mode=ExecutionMode.LEGACY_TRAINER,
                implementation_ref=trainer_reference,
                executable_factory_ref=_LEGACY_FACTORY,
                operations=(AlgorithmOperation.FIT,),
                input_compatibility=_compatibility(topology),
                framework=framework,
                artifact_format="none",
                allowed_config_keys=allowed_config_keys,
            ),
            environment=EnvironmentSpec(
                environment_id=f"tributo.{name}.legacy",
                dependencies=environment_dependencies,
            ),
            runtime=runtime,
            is_default=False,
        ),
        trainer_ref=trainer_reference,
        config_model_ref=QualifiedReference.parse(config_model_ref),
        limitations=limitations,
    )


XGBOOST_DESCRIPTOR = _descriptor(
    name="xgboost",
    trainer_ref="tributo.training.xgboost_trainer:XGBoostTrainerImpl",
    config_model_ref="tributo.training.xgboost_trainer:XGBoostTrainingConfig",
    extras_group="training",
    problem_types=(
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTI_CLASS_CLASSIFICATION,
        ProblemType.REGRESSION,
    ),
    capabilities=(
        Capability.TUNABLE,
        Capability.EXPORTABLE,
        Capability.DISTRIBUTED,
    ),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    learning_paradigm="supervised",
    model_family="gradient_boosted_trees",
    allowed_config_keys=("data", "model", "training", "ray", "resource", "output"),
    environment_dependencies=("ray==2.55.1", "xgboost>=2.1.0"),
    framework="xgboost",
    topology=RuntimeTopology.FRAMEWORK_MANAGED,
    native_mode=ExecutionMode.FRAMEWORK_NATIVE,
    framework_parallelism=4,
    limitations=(
        "Compatibility adapter materializes the bounded input before constructing the legacy Trainer.",
        "The adapter constructs one legacy Trainer while that Trainer manages its own distributed Ray workers.",
        "The formal framework-native XGBoost registration is the default; this adapter remains compatibility-only.",
    ),
)

DNN_DESCRIPTOR = _descriptor(
    name="dnn",
    trainer_ref="tributo.training.dnn_trainer:DNNTrainerImpl",
    config_model_ref="tributo.training.dnn_trainer:DNNTrainingConfig",
    extras_group="identity",
    problem_types=(ProblemType.BINARY_CLASSIFICATION,),
    capabilities=(
        Capability.TUNABLE,
        Capability.EXPORTABLE,
        Capability.DISTRIBUTED,
    ),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    learning_paradigm="supervised",
    model_family="deep_neural_network",
    allowed_config_keys=(
        "data",
        "features",
        "model",
        "loss",
        "pu_learning",
        "training",
        "ray",
        "resource",
        "output",
        "label_col",
    ),
    environment_dependencies=("ray==2.55.1", "torch>=2.5.0"),
    framework="pytorch",
    topology=RuntimeTopology.SINGLE_WORKER,
    native_mode=ExecutionMode.COLLECTIVE,
    limitations=(
        "The legacy Trainer adapter remains available during the portable API migration.",
    ),
)

PU_DESCRIPTOR = _descriptor(
    name="pu",
    trainer_ref="tributo.training.pu_trainer:PUTrainerImpl",
    config_model_ref="tributo.training.pu_trainer:PUTrainingConfig",
    extras_group="identity",
    problem_types=(ProblemType.PU_LEARNING,),
    capabilities=(
        Capability.TUNABLE,
        Capability.EXPORTABLE,
        Capability.DISTRIBUTED,
    ),
    data_loading=DataLoadingMode.CANONICAL_TRAINER,
    learning_paradigm="positive_unlabeled",
    model_family="deep_neural_network",
    allowed_config_keys=(
        "data",
        "features",
        "model",
        "pu",
        "training",
        "ray",
        "resource",
        "output",
        "label_col",
    ),
    environment_dependencies=("ray==2.55.1", "torch>=2.5.0"),
    framework="pytorch",
    topology=RuntimeTopology.SINGLE_WORKER,
    native_mode=ExecutionMode.COLLECTIVE,
    limitations=(
        "The legacy Trainer adapter remains available during the portable API migration.",
    ),
)

BUILTIN_LEGACY_DESCRIPTORS = (
    DNN_DESCRIPTOR,
    PU_DESCRIPTOR,
    XGBOOST_DESCRIPTOR,
)
