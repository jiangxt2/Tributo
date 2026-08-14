"""Unit tests for the two-stage fake input contract and ownership."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import cast

import pytest

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    AlgorithmOperation,
    AlgorithmRegistration,
    InputBinding,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    ResolvedInputDescriptor,
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
    FakeInputRuntimeAdapter,
    FakeTabularPayload,
)
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    InputRuntimeAdapter,
    PortableRuntimeAdapter,
    ResolvedInputLease,
    RuntimeExecutionEnvelope,
    RuntimeInputBinding,
    WorkerInputPayload,
)

from .conftest import function_registration, request_for


class _SuccessfulRuntime:
    @property
    def runtime_id(self) -> str:
        return "tributo.ray_task"

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        del envelope
        return WorkerExecutionResult(
            execution=AlgorithmExecutionResult(status="succeeded"),
            actual_versions={},
        )


class _RunIdentityRuntime(_SuccessfulRuntime):
    def __init__(self) -> None:
        self.run_id: str | None = None

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        self.run_id = envelope.run_id
        return super().execute(envelope)


class _FailingRuntime(_SuccessfulRuntime):
    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        del envelope
        raise AlgorithmExecutionError("primary runtime failure")


class _DirectWorkerRuntime(_SuccessfulRuntime):
    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        return worker_bootstrap(
            envelope.worker_envelope(0),
            {"worker_id": "lifecycle-worker", "world_rank": 0, "world_size": 1},
        )


class _InvalidResultRuntime(_SuccessfulRuntime):
    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        del envelope
        return cast(WorkerExecutionResult, object())


class _CloseFailingBinding(RuntimeInputBinding):
    def close(self) -> None:
        super().close()
        raise RuntimeError("binding cleanup failure")


class _CloseFailingInputAdapter(FakeInputRuntimeAdapter):
    def bind(
        self,
        lease: ResolvedInputLease,
        plan: ResolvedAlgorithmPlan,
    ) -> RuntimeInputBinding:
        binding = super().bind(lease, plan)
        return _CloseFailingBinding(binding.payload)


class _TrackedFakeInputResolver(FakeInputResolver):
    def describe(
        self,
        binding: InputBinding,
        context: InputResolutionContext,
    ) -> ResolvedInputDescriptor:
        descriptor = super().describe(binding, context)
        return replace(
            descriptor,
            compatible_worker_input_adapter_refs=(
                *descriptor.compatible_worker_input_adapter_refs,
                "tests.support.portable_algorithms:prepare_tracked_input",
            ),
        )


def _dispatcher(
    runtime: PortableRuntimeAdapter,
    input_adapter: InputRuntimeAdapter | None = None,
    registration: AlgorithmRegistration | None = None,
    resolver: FakeInputResolver | None = None,
) -> AlgorithmDispatcher:
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration or function_registration())
    resolved_input_resolver = resolver or FakeInputResolver()
    return AlgorithmDispatcher(
        AlgorithmPlanner(
            registry,
            {resolved_input_resolver.resolver_id: resolved_input_resolver},
        ),
        AlgorithmRunCoordinator(
            resolvers={resolved_input_resolver.resolver_id: resolved_input_resolver},
            input_adapters={
                resolved_input_resolver.resolver_id: input_adapter
                or FakeInputRuntimeAdapter()
            },
            runtimes={runtime.runtime_id: runtime},
        ),
    )


def test_describe_is_payload_free_and_open_detects_drift(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    resolver = FakeInputResolver()
    request = request_for("external", AlgorithmOperation.FIT)
    descriptor = resolver.describe(request.input_binding, InputResolutionContext())
    assert descriptor.reference == "binary-fixture"
    assert not hasattr(descriptor, "payload")

    drifted = request_for("external", AlgorithmOperation.FIT).input_binding
    object.__setattr__(drifted, "feature_names", ("x0",))
    with pytest.raises(AlgorithmInputError, match="drifted"):
        resolver.open(
            drifted,
            descriptor,
            InputExecutionContext(
                {
                    "binary-fixture": FakeInputInvocation(
                        FakeTabularPayload(binary_columns)
                    )
                }
            ),
        )


def test_lease_close_and_cancel_delegate_exactly_once(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    request = request_for("external", AlgorithmOperation.FIT)
    resolver = FakeInputResolver()
    descriptor = resolver.describe(request.input_binding, InputResolutionContext())
    calls = {"close": 0, "cancel": 0}
    invocation = FakeInputInvocation(
        FakeTabularPayload(binary_columns),
        close_callback=lambda: calls.__setitem__("close", calls["close"] + 1),
        cancel_callback=lambda: calls.__setitem__("cancel", calls["cancel"] + 1),
    )

    lease = resolver.open(
        request.input_binding,
        descriptor,
        InputExecutionContext({"binary-fixture": invocation}),
    )
    lease.cancel()
    lease.close()
    lease.cancel()

    assert calls == {"close": 0, "cancel": 1}
    assert lease.closed


def test_coordinator_closes_lease_once_on_success(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    calls: list[str] = []
    result = _dispatcher(_SuccessfulRuntime()).execute(
        request_for("external_function", AlgorithmOperation.FIT),
        InputExecutionContext(
            {
                "binary-fixture": FakeInputInvocation(
                    FakeTabularPayload(binary_columns),
                    close_callback=lambda: calls.append("closed"),
                    cancel_callback=lambda: calls.append("cancelled"),
                )
            }
        ),
    )

    assert result.execution.status == "succeeded"
    assert calls == ["closed"]


def test_coordinator_uses_one_run_identity_for_runtime_and_result(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    runtime = _RunIdentityRuntime()

    result = _dispatcher(runtime).execute(
        request_for("external_function", AlgorithmOperation.FIT),
        InputExecutionContext(
            {"binary-fixture": FakeInputInvocation(FakeTabularPayload(binary_columns))}
        ),
    )

    assert runtime.run_id == result.run_id


def test_coordinator_cancels_lease_and_preserves_primary_failure(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    calls: list[str] = []
    with pytest.raises(AlgorithmExecutionError, match="primary runtime failure") as exc:
        _dispatcher(_FailingRuntime(), _CloseFailingInputAdapter()).execute(
            request_for("external_function", AlgorithmOperation.FIT),
            InputExecutionContext(
                {
                    "binary-fixture": FakeInputInvocation(
                        FakeTabularPayload(binary_columns),
                        close_callback=lambda: calls.append("closed"),
                        cancel_callback=lambda: calls.append("cancelled"),
                    )
                }
            ),
        )

    assert calls == ["cancelled"]
    assert "cleanup also failed: RuntimeError" in exc.value.__notes__


def test_coordinator_still_closes_lease_when_binding_cleanup_fails(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    calls: list[str] = []
    with pytest.raises(AlgorithmExecutionError, match="cleanup failed"):
        _dispatcher(_SuccessfulRuntime(), _CloseFailingInputAdapter()).execute(
            request_for("external_function", AlgorithmOperation.FIT),
            InputExecutionContext(
                {
                    "binary-fixture": FakeInputInvocation(
                        FakeTabularPayload(binary_columns),
                        close_callback=lambda: calls.append("closed"),
                    )
                }
            ),
        )

    assert calls == ["closed"]


def test_coordinator_cancels_lease_for_invalid_runtime_result(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    calls: list[str] = []
    with pytest.raises(AlgorithmExecutionError, match="invalid WorkerExecutionResult"):
        _dispatcher(_InvalidResultRuntime()).execute(
            request_for("external_function", AlgorithmOperation.FIT),
            InputExecutionContext(
                {
                    "binary-fixture": FakeInputInvocation(
                        FakeTabularPayload(binary_columns),
                        close_callback=lambda: calls.append("closed"),
                        cancel_callback=lambda: calls.append("cancelled"),
                    )
                }
            ),
        )

    assert calls == ["cancelled"]


@pytest.mark.parametrize(
    "function",
    [
        "tests.support.portable_algorithms:custom_training_fragment",
        "tests.support.portable_algorithms:failing_training_fragment",
    ],
)
def test_worker_prepared_input_closes_once_on_success_and_failure(
    binary_columns: dict[str, tuple[object, ...]],
    function: str,
) -> None:
    portable_algorithms = importlib.import_module("tests.support.portable_algorithms")
    portable_algorithms.WORKER_CLEANUP_EVENTS.clear()
    registration = function_registration(function)
    tracked_adapter = QualifiedReference.parse(
        "tests.support.portable_algorithms:prepare_tracked_input"
    )
    registration = replace(
        registration,
        implementation=replace(
            registration.implementation,
            input_compatibility=replace(
                registration.implementation.input_compatibility,
                supported_explicit_adapters=(
                    *registration.implementation.input_compatibility.supported_explicit_adapters,
                    tracked_adapter,
                ),
            ),
        ),
        runtime=replace(
            registration.runtime,
            worker_input_adapter_ref=tracked_adapter,
        ),
    )

    _dispatcher(
        _DirectWorkerRuntime(),
        registration=registration,
        resolver=_TrackedFakeInputResolver(),
    ).execute(
        request_for("external_function", AlgorithmOperation.FIT),
        InputExecutionContext(
            {"binary-fixture": FakeInputInvocation(FakeTabularPayload(binary_columns))}
        ),
    )

    assert portable_algorithms.WORKER_CLEANUP_EVENTS == ["closed"]


def test_fake_data_parallel_binding_partitions_every_row_once(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    registration = function_registration(data_parallel=True)
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    request = request_for("external_function", AlgorithmOperation.FIT)
    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(request)
    lease = resolver.open(
        request.input_binding,
        plan.input_descriptor,
        InputExecutionContext(
            {"binary-fixture": FakeInputInvocation(FakeTabularPayload(binary_columns))}
        ),
    )

    binding = FakeInputRuntimeAdapter().bind(lease, plan)
    try:
        assert [payload.partition_index for payload in binding.payloads] == [0, 1]
        assert all(payload.partition_count == 2 for payload in binding.payloads)
        assert all(
            payload.expected_total_rows == len(binary_columns["x0"])
            for payload in binding.payloads
        )
        shards = [
            cast(FakeTabularPayload, payload.value).columns_by_name["x0"]
            for payload in binding.payloads
        ]
        flattened = [value for shard in shards for value in shard]
        assert flattened == list(binary_columns["x0"])
        assert len(flattened) == len(set(flattened))
    finally:
        binding.close()
        lease.close()


def test_runtime_binding_rejects_inconsistent_expected_total_rows() -> None:
    input_binding = request_for(
        "external_function", AlgorithmOperation.FIT
    ).input_binding

    with pytest.raises(AlgorithmInputError, match="expected total rows"):
        RuntimeInputBinding(
            (
                WorkerInputPayload(
                    "train",
                    input_binding,
                    object(),
                    partition_index=0,
                    partition_count=2,
                    expected_total_rows=4,
                ),
                WorkerInputPayload(
                    "train",
                    input_binding,
                    object(),
                    partition_index=1,
                    partition_count=2,
                    expected_total_rows=3,
                ),
            )
        )


@pytest.mark.parametrize("value", (-1, True, 1.5))
def test_worker_payload_rejects_invalid_expected_total_rows(value: object) -> None:
    with pytest.raises(AlgorithmInputError, match="expected_total_rows"):
        WorkerInputPayload(
            "train",
            request_for("external_function", AlgorithmOperation.FIT).input_binding,
            object(),
            expected_total_rows=cast(int, value),
        )
