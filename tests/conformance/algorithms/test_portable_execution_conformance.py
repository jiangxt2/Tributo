"""Cross-channel conformance tests for one portable execution model."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from typing import Literal, cast

import pytest

from tests.algorithms.conftest import (
    dispatcher_for,
    function_registration,
    request_for,
    sklearn_registration,
)
from tests.support.algorithm_ingestion import (
    StubIngestionGateway,
    ingestion_invocation,
)
from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    ExecutionMode,
    InputBinding,
    QualifiedReference,
    RuntimeTopology,
    WorkerExecutionResult,
)
from tributo.algorithms.core import (
    AlgorithmDispatcher,
    AlgorithmPlanner,
    AlgorithmRegistrationRegistry,
    AlgorithmRunCoordinator,
)
from tributo.algorithms.core.worker import worker_bootstrap
from tributo.algorithms.input import (
    FakeInputInvocation,
    FakeInputResolver,
    FakeTabularPayload,
)
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    RuntimeExecutionEnvelope,
)
from tributo.data import IngestionGateway
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputResolver,
    IngestionInputRuntimeAdapter,
)


class _ConformanceRuntime:
    @property
    def runtime_id(self) -> str:
        return "tributo.ray_task"

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        return worker_bootstrap(
            envelope.worker_envelope(0),
            {"worker_id": "conformance-worker", "world_rank": 0, "world_size": 1},
        )


@pytest.mark.parametrize(
    ("registration_factory", "algorithm", "mode"),
    [
        (
            sklearn_registration,
            "external_sklearn",
            ExecutionMode.MANAGED_ESTIMATOR,
        ),
        (
            function_registration,
            "external_function",
            ExecutionMode.CUSTOM_RAY_FUNCTION,
        ),
    ],
)
def test_registration_and_plan_share_one_contract(
    registration_factory: Callable[[], AlgorithmRegistration],
    algorithm: str,
    mode: ExecutionMode,
) -> None:
    module = "tests.support.portable_algorithms"
    sys.modules.pop(module, None)
    registration = registration_factory()
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})

    plan = planner.plan(request_for(algorithm, AlgorithmOperation.FIT))

    assert plan.resolution.execution_mode is mode
    assert plan.implementation.input_compatibility.accepted_input_views
    assert plan.implementation.input_compatibility.accepted_ingestion_engines
    assert plan.implementation.input_compatibility.supported_explicit_adapters
    assert plan.implementation.input_compatibility.distribution_policy
    assert set(
        plan.implementation.input_compatibility.required_input_capabilities
    ).issubset(plan.input_descriptor.input_capabilities)
    assert plan.input_descriptor.engine_id == "tributo.fake_tabular"
    assert plan.input_descriptor.view_kind == "materialized_tabular"
    assert plan.runtime.runtime_id == "tributo.ray_task"
    assert plan.to_dict()["format_version"] == 2
    assert module not in sys.modules


def test_snapshot_order_is_deterministic_across_channels() -> None:
    registrations = (function_registration(), sklearn_registration())
    expected = ["external_function", "external_sklearn"]

    for ordered in (registrations, tuple(reversed(registrations))):
        registry = AlgorithmRegistrationRegistry()
        for registration in ordered:
            registry.register(registration)
        assert [item.spec.name for item in registry.snapshot()] == expected


@pytest.mark.parametrize(
    ("registration", "algorithm", "topology", "worker_count"),
    [
        (
            sklearn_registration(framework_managed=True),
            "external_sklearn",
            RuntimeTopology.FRAMEWORK_MANAGED,
            1,
        ),
        (
            function_registration(data_parallel=True),
            "external_function",
            RuntimeTopology.DATA_PARALLEL,
            2,
        ),
    ],
)
def test_distributed_channels_lower_to_the_shared_runtime_contract(
    registration: AlgorithmRegistration,
    algorithm: str,
    topology: RuntimeTopology,
    worker_count: int,
) -> None:
    module = "tests.support.portable_algorithms"
    sys.modules.pop(module, None)
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()

    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for(algorithm, AlgorithmOperation.FIT)
    )

    assert plan.runtime.topology is topology
    assert plan.runtime.worker_count == worker_count
    assert plan.to_dict()["runtime"]["topology"] == topology.value
    assert module not in sys.modules


@pytest.mark.parametrize(
    ("registration_factory", "algorithm"),
    [
        (sklearn_registration, "external_sklearn"),
        (function_registration, "external_function"),
    ],
)
def test_channels_share_the_bounded_execution_contract(
    registration_factory: Callable[[], AlgorithmRegistration],
    algorithm: str,
) -> None:
    binary_columns: dict[str, tuple[object, ...]] = {
        "x0": (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0),
        "x1": (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0),
        "label": (0, 0, 0, 0, 1, 1, 1, 1),
    }
    result = dispatcher_for(
        registration_factory(),
        _ConformanceRuntime(),
    ).execute(
        request_for(algorithm, AlgorithmOperation.FIT),
        InputExecutionContext(
            {"binary-fixture": FakeInputInvocation(FakeTabularPayload(binary_columns))}
        ),
    )

    assert result.execution.status == "succeeded"
    assert len(result.plan_id) == 64
    assert result.actual_versions["python"]
    assert result.input_provenance["resolver_id"] == "tributo.fake_tabular"


@pytest.mark.parametrize(
    ("registration_factory", "algorithm"),
    [
        (sklearn_registration, "external_sklearn"),
        (function_registration, "external_function"),
    ],
)
@pytest.mark.parametrize("engine", ["ray", "daft"])
def test_channels_share_the_production_ingestion_contract(
    registration_factory: Callable[[], AlgorithmRegistration],
    algorithm: str,
    engine: Literal["ray", "daft"],
) -> None:
    columns: dict[str, tuple[object, ...]] = {
        "x0": (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0),
        "x1": (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0),
        "label": (0, 0, 0, 0, 1, 1, 1, 1),
    }
    gateway = StubIngestionGateway(columns)
    resolver = IngestionInputResolver(cast(IngestionGateway, gateway))
    registration = registration_factory()
    registration = replace(
        registration,
        runtime=replace(
            registration.runtime,
            worker_input_adapter_ref=QualifiedReference.parse(
                "tributo.integrations.algorithm_inputs.ingestion:"
                "prepare_ingestion_input"
            ),
        ),
    )
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    dispatcher = AlgorithmDispatcher(
        AlgorithmPlanner(registry, {resolver.resolver_id: resolver}),
        AlgorithmRunCoordinator(
            resolvers={resolver.resolver_id: resolver},
            input_adapters={resolver.resolver_id: IngestionInputRuntimeAdapter()},
            runtimes={"tributo.ray_task": _ConformanceRuntime()},
        ),
    )
    request = replace(
        request_for(algorithm, AlgorithmOperation.FIT),
        input_binding=InputBinding(
            name="train",
            resolver_id=INGESTION_RESOLVER_ID,
            reference="tests.production-input",
            feature_names=("x0", "x1"),
            label_name="label",
        ),
    )
    invocation = ingestion_invocation(engine=engine)
    resolution_context = InputResolutionContext(
        values={request.input_binding.reference: invocation}
    )
    execution_context = InputExecutionContext(
        {request.input_binding.reference: invocation}
    )

    result = dispatcher.execute(
        request,
        execution_context,
        resolution_context=resolution_context,
    )

    assert result.execution.status == "succeeded"
    assert result.input_provenance["resolver_id"] == INGESTION_RESOLVER_ID
    assert (
        result.input_provenance["engine_id"]
        == f"tributo.{engine if engine == 'daft' else 'ray_data'}"
    )
    assert (
        result.input_provenance["receipt"]["request_digest"]
        == (result.input_provenance["request_digest"])
    )
    assert gateway.lifecycle_events == ["closed"]
    if engine == "daft":
        assert gateway.last_daft_frame is not None
        assert gateway.last_daft_frame.ray_conversion_calls == 0
