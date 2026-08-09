"""Shared fixtures for portable algorithm unit and conformance tests."""

from __future__ import annotations

import pytest

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmRequest,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    InputBinding,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
)
from tributo.algorithms.core import (
    AlgorithmBuilder,
    AlgorithmDispatcher,
    AlgorithmPlanner,
    AlgorithmRegistrationRegistry,
    AlgorithmRunCoordinator,
)
from tributo.algorithms.input import (
    FAKE_RESOLVER_ID,
    FakeInputResolver,
    FakeInputRuntimeAdapter,
)
from tributo.algorithms.spi import PortableRuntimeAdapter
from tributo.training.algorithm_spec import AlgorithmSpec, ProblemType


def make_spec(
    name: str,
    *,
    operations: tuple[str, ...],
    mode: ExecutionMode,
    default_config: dict[str, object] | None = None,
) -> AlgorithmSpec:
    """Create a framework-neutral AlgorithmSpec for tests."""
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config=default_config or {},
        supported_tasks=operations,
        operations=operations,
        problem_types=(ProblemType.BINARY_CLASSIFICATION,),
        learning_paradigm="supervised",
        model_family=(
            "linear_model"
            if mode is ExecutionMode.MANAGED_ESTIMATOR
            else "user_defined"
        ),
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(mode.value,),
        config_contract_ref=f"test.{name}.config.v1",
        input_contract_ref="test.tabular.binary.v1",
        output_contract_ref="test.binary.prediction.v1",
    )


def fake_runtime_binding(
    *,
    topology: RuntimeTopology = RuntimeTopology.SINGLE_WORKER,
    worker_count: int = 1,
    framework_parallelism: int = 1,
) -> RuntimeBinding:
    """Return the Ray task and in-memory Worker input binding."""
    return RuntimeBinding(
        runtime_id="tributo.ray_task",
        worker_input_adapter_ref=QualifiedReference.parse(
            "tributo.algorithms.input.fake:prepare_input"
        ),
        topology=topology,
        worker_count=worker_count,
        framework_parallelism=framework_parallelism,
        result_reducer_ref=(
            QualifiedReference.parse(
                "tests.support.portable_algorithms:reduce_custom_training_results"
            )
            if topology is RuntimeTopology.DATA_PARALLEL
            else None
        ),
        num_cpus=0,
        max_retries=0,
    )


def fake_input_compatibility(
    *topologies: RuntimeTopology,
) -> BackendInputCompatibility:
    """Declare the development-only Fake input plus production bridge facts."""
    return BackendInputCompatibility(
        accepted_input_views=(
            "daft_dataframe",
            "materialized_tabular",
            "ray_data",
        ),
        accepted_ingestion_engines=(
            "tributo.daft",
            "tributo.fake_tabular",
            "tributo.ray_data",
        ),
        required_input_capabilities=("materializable",),
        supported_explicit_adapters=tuple(
            QualifiedReference.parse(reference)
            for reference in (
                "tributo.algorithms.input.fake:prepare_input",
                "tributo.integrations.algorithm_inputs.ingestion:prepare_daft_input",
                "tributo.integrations.algorithm_inputs.ingestion:prepare_ingestion_input",
                "tributo.integrations.algorithm_inputs.ingestion:prepare_ray_data_input",
            )
        ),
        distribution_policy=topologies,
    )


def sklearn_registration(
    *,
    pipeline: bool = False,
    framework_managed: bool = False,
) -> AlgorithmRegistration:
    """Build an external-style managed sklearn registration."""
    factory = (
        "tests.support.portable_algorithms:ray_joblib_probe_factory"
        if framework_managed
        else (
            "tests.support.portable_algorithms:logistic_pipeline_factory"
            if pipeline
            else "tests.support.portable_algorithms:logistic_regression_factory"
        )
    )
    allowed = (
        ("n_jobs", "task_count")
        if framework_managed
        else (("model__C",) if pipeline else ("C", "max_iter"))
    )
    return AlgorithmBuilder.from_sklearn(
        spec=make_spec(
            "external_sklearn",
            operations=("fit", "evaluate", "predict"),
            mode=ExecutionMode.MANAGED_ESTIMATOR,
        ),
        implementation_id=(
            "tests.sklearn_pipeline" if pipeline else "tests.sklearn_logistic"
        ),
        implementation_version="1.0.0",
        estimator_factory=factory,
        environment=EnvironmentSpec(
            environment_id="tests.sklearn",
            dependencies=("scikit-learn>=1.4,<2",),
        ),
        runtime=(
            fake_runtime_binding(
                topology=RuntimeTopology.FRAMEWORK_MANAGED,
                framework_parallelism=2,
            )
            if framework_managed
            else fake_runtime_binding()
        ),
        input_compatibility=fake_input_compatibility(
            RuntimeTopology.SINGLE_WORKER,
            RuntimeTopology.FRAMEWORK_MANAGED,
        ),
        allowed_config_keys=allowed,
        trusted_pickle=True,
        is_default=True,
    )


def function_registration(
    function: str = "tests.support.portable_algorithms:custom_training_fragment",
    *,
    data_parallel: bool = False,
) -> AlgorithmRegistration:
    """Build an external-style custom function registration."""
    return AlgorithmBuilder.from_ray_function(
        spec=make_spec(
            "external_function",
            operations=("fit",),
            mode=ExecutionMode.CUSTOM_RAY_FUNCTION,
        ),
        implementation_id="tests.custom_function",
        implementation_version="1.0.0",
        function=function,
        environment=EnvironmentSpec(environment_id="tests.user_function"),
        runtime=(
            fake_runtime_binding(
                topology=RuntimeTopology.DATA_PARALLEL,
                worker_count=2,
            )
            if data_parallel
            else fake_runtime_binding()
        ),
        input_compatibility=fake_input_compatibility(
            RuntimeTopology.SINGLE_WORKER,
            RuntimeTopology.DATA_PARALLEL,
        ),
        allowed_config_keys=("threshold",),
        is_default=True,
    )


def request_for(
    algorithm: str,
    operation: AlgorithmOperation,
    *,
    config: dict[str, object] | None = None,
) -> AlgorithmRequest:
    """Create a request bound to the common binary test input."""
    return AlgorithmRequest(
        algorithm=algorithm,
        operation=operation,
        input_binding=InputBinding(
            name="train",
            resolver_id=FAKE_RESOLVER_ID,
            reference="binary-fixture",
            feature_names=("x0", "x1"),
            label_name="label",
        ),
        algorithm_config=config or {},
    )


def dispatcher_for(
    registration: AlgorithmRegistration,
    runtime: PortableRuntimeAdapter,
) -> AlgorithmDispatcher:
    """Compose Core with injected input and Runtime adapters."""
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})
    coordinator = AlgorithmRunCoordinator(
        resolvers={resolver.resolver_id: resolver},
        input_adapters={resolver.resolver_id: FakeInputRuntimeAdapter()},
        runtimes={runtime.runtime_id: runtime},
    )
    return AlgorithmDispatcher(planner, coordinator)


@pytest.fixture
def binary_columns() -> dict[str, tuple[object, ...]]:
    """Return a linearly separable binary classification fixture."""
    return {
        "x0": (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0),
        "x1": (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0),
        "label": (0, 0, 0, 0, 1, 1, 1, 1),
    }
