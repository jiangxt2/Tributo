"""Independent package fixture implementing Tributo's constrained MapReduce SPI."""

from __future__ import annotations

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ImplementationDescriptor,
    InputDistribution,
    MapReducePolicy,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    StateCoordination,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.builtin.multinomial_nb import DistributedMultinomialNB
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType


class ThirdPartyMultinomialNB(DistributedMultinomialNB):
    """External algorithm class admitted through the MapReduce ABC."""


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> ThirdPartyMultinomialNB:
    """Construct only the implementation named by this package descriptor."""
    del artifacts
    if implementation is not ThirdPartyMultinomialNB:
        raise AlgorithmConfigurationError(
            "third-party implementation does not match its descriptor"
        )
    return ThirdPartyMultinomialNB(plan)


_STATE_SCHEMA = (
    StateField("classes", "int64", (None,)),
    StateField("class_count", "float64", (None,)),
    StateField("feature_count", "float64", (None, None)),
    StateField("row_count", "int64", ()),
)
_INPUT_ADAPTER = QualifiedReference.parse(
    "tributo.integrations.algorithm_inputs.ingestion:prepare_ray_batch_input"
)

REGISTRATION = AlgorithmRegistration(
    spec=AlgorithmSpec(
        name="third_party_multinomial_nb",
        trainer_cls=None,
        version="0.1.0",
        default_config={"alpha": 1.0, "force_alpha": True, "fit_prior": True},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTI_CLASS_CLASSIFICATION,
        ),
        capabilities=(Capability.EXPORTABLE, Capability.DISTRIBUTED),
        extras_group="training",
        learning_paradigm="supervised",
        model_family="naive_bayes",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
        config_contract_ref="example.multinomial_nb.config.v1",
        input_contract_ref="tributo.tabular.nonnegative.v1",
        output_contract_ref="tributo.classification.onnx.v1",
    ),
    implementation=ImplementationDescriptor(
        implementation_id="example.multinomial_nb.map_reduce",
        version="0.1.0",
        execution_mode=ExecutionMode.MAP_REDUCE,
        implementation_ref=QualifiedReference.parse(
            "tributo_test_distributed_algorithm:ThirdPartyMultinomialNB"
        ),
        executable_factory_ref=QualifiedReference.parse(
            "tributo_test_distributed_algorithm:create_algorithm"
        ),
        operations=(AlgorithmOperation.FIT,),
        input_compatibility=BackendInputCompatibility(
            accepted_input_views=("ray_data",),
            accepted_ingestion_engines=("tributo.ray_data",),
            required_input_capabilities=("shardable",),
            supported_explicit_adapters=(_INPUT_ADAPTER,),
            distribution_policy=(RuntimeTopology.RAY_MAP_REDUCE,),
        ),
        distribution="tributo-test-distributed-algorithm",
        framework="sklearn",
        allowed_config_keys=(
            "alpha",
            "class_prior",
            "fit_prior",
            "force_alpha",
            "output",
        ),
        runtime_id="tributo.ray_map_reduce",
        worker_input_adapter_ref=_INPUT_ADAPTER,
        exporter_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.multinomial_nb:export_model"
        ),
        flavor_id="onnx-runtime-v1",
    ),
    environment=EnvironmentSpec(
        environment_id="example.multinomial_nb.v1",
        dependencies=(
            "onnx>=1.16.0",
            "onnxruntime>=1.20.0",
            "scikit-learn>=1.4,<2",
            "skl2onnx>=1.17",
            "tributo-test-distributed-algorithm==0.1.0",
        ),
    ),
    distribution_spec=DistributionSpec(
        strategy=DistributionStrategy.RAY_MAP_REDUCE,
        supported_worker_range=WorkerRange(1, 32),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.KUBERNETES,
        ),
        resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
        input_distribution=InputDistribution.SHARDED,
        state_coordination=StateCoordination.ASSOCIATIVE_REDUCE,
        policy=MapReducePolicy(
            state_schema=_STATE_SCHEMA,
            max_partial_state_bytes=64 * 1024 * 1024,
            reducer_ref=(
                "tributo_test_distributed_algorithm:"
                "ThirdPartyMultinomialNB.merge_states"
            ),
            finalizer_ref=(
                "tributo_test_distributed_algorithm:"
                "ThirdPartyMultinomialNB.finalize_model"
            ),
            commutative=True,
            max_retries=0,
        ),
    ),
    is_default=True,
)

DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=REGISTRATION,
    package_name="tributo-test-distributed-algorithm",
    package_version="0.1.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    limitations=("CPU-only conformance fixture.",),
)

__all__ = ["DESCRIPTOR", "REGISTRATION", "ThirdPartyMultinomialNB"]
