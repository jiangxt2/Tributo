"""Unit tests for the production ingestion-to-algorithm bridge."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from tests.support.algorithm_ingestion import (
    StubIngestionGateway,
    StubRayDataset,
    ingestion_invocation,
)
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmInputError,
    AlgorithmOperation,
    AlgorithmRequest,
    InputBinding,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    MaterializedTabularInputView,
    ResolvedInputLease,
    WorkerInputPayload,
)
from tributo.data import (
    IngestionGateway,
    IngestionOpenResult,
    IngestionRequest,
    IngestionRuntimeContext,
    RayDataHandle,
)
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputResolver,
    IngestionInputRuntimeAdapter,
    IngestionRequestRef,
    prepare_ingestion_input,
)
from tributo.integrations.algorithm_inputs import ingestion as ingestion_bridge

from .conftest import function_registration, request_for, sklearn_registration

_COLUMNS: dict[str, tuple[object, ...]] = {
    "x0": (-2.0, -1.0, 1.0, 2.0),
    "x1": (-1.0, -0.5, 0.5, 1.0),
    "label": (0, 0, 1, 1),
}


class _ReceiptDriftGateway(StubIngestionGateway):
    def open(
        self,
        request: IngestionRequest,
        runtime_context: object = None,
    ) -> IngestionOpenResult:
        result = super().open(request, runtime_context)
        result.receipt = result.receipt.model_copy(update={"dataset_ref": "f" * 64})
        return result


def _resolver(
    gateway: StubIngestionGateway,
    *,
    accepted: tuple[Literal["ray_data"], ...] = ("ray_data",),
) -> IngestionInputResolver:
    return IngestionInputResolver(
        cast(IngestionGateway, gateway),
        accepted_handle_kinds=accepted,
    )


def _binding() -> InputBinding:
    return InputBinding(
        name="train",
        resolver_id=INGESTION_RESOLVER_ID,
        reference="tests.algorithm-input",
        feature_names=("x0", "x1"),
        label_name="label",
    )


def test_request_ref_rejects_non_identity_values() -> None:
    assert IngestionRequestRef("tests.algorithm-input").resolver_id == (
        INGESTION_RESOLVER_ID
    )
    with pytest.raises(AlgorithmInputError, match="request key"):
        IngestionRequestRef("https://user:secret@example.test/data")


def test_invocation_keeps_runtime_context_as_second_positional_argument() -> None:
    invocation = ingestion_invocation()
    context = IngestionRuntimeContext()

    restored = type(invocation)(invocation.request, context)

    assert restored.request == invocation.request
    assert restored.runtime_context is context
    assert restored.handle_adapter_id is None


def test_removed_handle_adapter_is_rejected() -> None:
    invocation = ingestion_invocation()
    invocation_type: Any = type(invocation)
    with pytest.raises(
        AlgorithmInputError, match="handle adapters are no longer supported"
    ):
        invocation_type(invocation.request, handle_adapter_id="tributo.daft_to_ray")


def test_planned_descriptor_contains_gateway_facts_but_not_request_body() -> None:
    gateway = StubIngestionGateway(_COLUMNS)
    resolver = _resolver(gateway)
    request = replace(
        request_for("external_sklearn", AlgorithmOperation.FIT),
        input_binding=_binding(),
    )
    registration = sklearn_registration()
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
    invocation = ingestion_invocation(credentials=True)

    plan = AlgorithmPlanner(
        registry,
        {resolver.resolver_id: resolver},
    ).plan(
        request,
        InputResolutionContext(values={request.input_binding.reference: invocation}),
    )

    serialized = plan.to_json()
    assert plan.format_version == 2
    assert plan.input_descriptor.engine_id == "tributo.ray_data"
    assert plan.input_descriptor.view_kind == "ray_data"
    assert "materializable" in plan.input_descriptor.input_capabilities
    assert "shardable" in plan.input_descriptor.input_capabilities
    assert plan.input_descriptor.resolver_payload["bridge_descriptor_version"] == 1
    assert plan.input_descriptor.resolver_payload["handle_adapter_id"] is None
    assert "fixture-access-key" not in serialized
    assert "fixture-secret-key" not in serialized
    assert "fixture-bucket" not in serialized
    assert "trace_id" not in serialized
    assert gateway.open_calls == 0


def test_planner_rejects_incompatible_worker_adapter_before_gateway_open() -> None:
    gateway = StubIngestionGateway(_COLUMNS)
    resolver = _resolver(gateway)
    registry = AlgorithmRegistrationRegistry()
    registry.register(sklearn_registration())
    request = replace(
        request_for("external_sklearn", AlgorithmOperation.FIT),
        input_binding=_binding(),
    )

    with pytest.raises(AlgorithmConfigurationError, match="incompatible"):
        AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
            request,
            InputResolutionContext(
                values={
                    request.input_binding.reference: ingestion_invocation(engine="ray")
                }
            ),
        )

    assert gateway.open_calls == 0


def test_request_drift_fails_before_gateway_open() -> None:
    gateway = StubIngestionGateway(_COLUMNS)
    resolver = _resolver(gateway)
    binding = _binding()
    planned = ingestion_invocation(engine="ray")
    descriptor = resolver.describe(
        binding,
        InputResolutionContext(values={binding.reference: planned}),
    )

    with pytest.raises(AlgorithmInputError, match="drifted"):
        resolver.open(
            binding,
            descriptor,
            InputExecutionContext(
                {binding.reference: ingestion_invocation(credentials=True)}
            ),
        )

    assert gateway.open_calls == 0


def test_gateway_owner_close_and_cancel_are_delegated_exactly_once() -> None:
    gateway = StubIngestionGateway(_COLUMNS)
    resolver = _resolver(gateway)
    binding = _binding()
    invocation = ingestion_invocation()
    descriptor = resolver.describe(
        binding,
        InputResolutionContext(values={binding.reference: invocation}),
    )

    lease = resolver.open(
        binding,
        descriptor,
        InputExecutionContext({binding.reference: invocation}),
    )
    lease.cancel()
    lease.close()
    lease.cancel()

    assert gateway.lifecycle_events == ["cancelled"]


def test_opened_receipt_drift_is_rejected_and_owner_is_cancelled() -> None:
    gateway = _ReceiptDriftGateway(_COLUMNS)
    resolver = _resolver(gateway)
    binding = _binding()
    invocation = ingestion_invocation()
    descriptor = resolver.describe(
        binding,
        InputResolutionContext(values={binding.reference: invocation}),
    )

    with pytest.raises(AlgorithmInputError, match="drifted"):
        resolver.open(
            binding,
            descriptor,
            InputExecutionContext({binding.reference: invocation}),
        )

    assert gateway.lifecycle_events == ["cancelled"]


def test_managed_materialization_fails_closed_above_row_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingestion_bridge, "_MAX_MATERIALIZED_ROWS", 2)
    handle = RayDataHandle(StubRayDataset(_COLUMNS))

    with pytest.raises(AlgorithmInputError, match="row limit"):
        prepare_ingestion_input(WorkerInputPayload("train", _binding(), handle))


def test_managed_materialization_rejects_nested_column_values() -> None:
    handle = RayDataHandle(
        StubRayDataset(
            {
                "x0": ([1.0, 2.0],),
                "x1": (1.0,),
                "label": (1,),
            }
        )
    )

    with pytest.raises(AlgorithmInputError, match="one-dimensional scalar data"):
        prepare_ingestion_input(WorkerInputPayload("train", _binding(), handle))


def _data_parallel_plan(
    gateway: StubIngestionGateway,
    *,
    engine: Literal["ray"],
) -> tuple[
    IngestionInputResolver,
    AlgorithmRequest,
    ResolvedAlgorithmPlan,
    InputExecutionContext,
]:
    resolver = _resolver(gateway)
    registration = function_registration(data_parallel=True)
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
    request = replace(
        request_for("external_function", AlgorithmOperation.FIT),
        input_binding=_binding(),
    )
    invocation = ingestion_invocation(
        engine=engine,
    )
    resolution_context = InputResolutionContext(
        values={request.input_binding.reference: invocation}
    )
    execution_context = InputExecutionContext(
        {request.input_binding.reference: invocation}
    )
    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request,
        resolution_context,
    )
    return resolver, request, plan, execution_context


def test_data_parallel_ray_handle_uses_public_streaming_split() -> None:
    gateway = StubIngestionGateway(_COLUMNS)
    resolver, request, plan, execution_context = _data_parallel_plan(
        gateway, engine="ray"
    )
    lease = resolver.open(
        request.input_binding,
        plan.input_descriptor,
        execution_context,
    )
    dataset = cast(RayDataHandle, lease.handle).dataset

    binding = IngestionInputRuntimeAdapter().bind(lease, plan)
    prepared_inputs = []
    try:
        assert dataset.streaming_split_calls == [(2, False)]
        assert [payload.partition_index for payload in binding.payloads] == [0, 1]
        assert all(payload.partition_count == 2 for payload in binding.payloads)
        prepared_inputs = [
            prepare_ingestion_input(payload) for payload in binding.payloads
        ]
        values = [
            value
            for prepared in prepared_inputs
            for value in cast(
                MaterializedTabularInputView,
                prepared.views["train"],
            ).columns()["x0"]
        ]
        assert values == list(_COLUMNS["x0"])
        assert len(values) == len(set(values))
    finally:
        for prepared in prepared_inputs:
            prepared.close()
        binding.close()
        lease.close()


def test_map_reduce_binding_counts_input_before_public_streaming_split() -> None:
    dataset = StubRayDataset(_COLUMNS)
    lease = ResolvedInputLease(
        handle=RayDataHandle(dataset),
        provenance={},
    )
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            runtime=SimpleNamespace(
                topology=RuntimeTopology.RAY_MAP_REDUCE,
                worker_count=2,
            ),
            primary_input_binding=_binding(),
        ),
    )

    runtime_binding = IngestionInputRuntimeAdapter().bind(lease, plan)
    try:
        assert dataset.count_calls == 1
        assert dataset.streaming_split_calls == [(2, False)]
        assert [
            payload.expected_total_rows for payload in runtime_binding.payloads
        ] == [len(_COLUMNS["x0"]), len(_COLUMNS["x0"])]
    finally:
        runtime_binding.close()
        lease.close()
