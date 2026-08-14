"""Contract tests for the native-engine writing control plane."""

from __future__ import annotations

from typing import Any

import pytest

from tributo.data.contracts.handles import DaftDataFrameHandle, RayDataHandle
from tributo.data.engine_ids import ENGINE_ALIASES
from tributo.data.writing import (
    WriteBindingError,
    WriteBindingRegistry,
    WriteCapability,
    WriteCapabilityError,
    WriteDescriptor,
    WriteExecutionContext,
    WriteGateway,
    WriteMode,
    WriteReceipt,
    WriteRequest,
    WriteTargetRegistry,
)
from tributo.data.writing.compatibility import ray_connector_write_request
from tributo.data.writing.targets import LogicalWritePlan


def _request(**overrides: Any) -> WriteRequest:
    values: dict[str, Any] = {
        "engine": "ray",
        "target_kind": "parquet",
        "target": "/tmp/output",
        "mode": WriteMode.OVERWRITE,
    }
    values.update(overrides)
    return WriteRequest(**values)


def _descriptor(*, binding_id: str = "test.ray.parquet") -> WriteDescriptor:
    return WriteDescriptor(
        engine_id="ray",
        target_kind="parquet",
        binding_id=binding_id,
        engine_version_spec="==2.55.1",
        binding_distribution="test-write-binding",
        binding_distribution_version="1.0.0",
        capabilities=WriteCapability(supported_modes=frozenset({WriteMode.OVERWRITE})),
    )


class _Binding:
    describe_calls = 0
    execute_calls = 0

    def describe(self, plan: LogicalWritePlan, input_handle: Any) -> WriteDescriptor:
        type(self).describe_calls += 1
        assert isinstance(input_handle, RayDataHandle)
        assert plan.target_kind == "parquet"
        return _descriptor()

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: Any,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        type(self).execute_calls += 1
        assert isinstance(input_handle, RayDataHandle)
        assert context.request_digest == plan.request_digest
        return WriteReceipt(
            request_digest=plan.request_digest,
            engine_id=plan.engine_id,
            binding_id="test.ray.parquet",
            target_kind=plan.target_kind,
            target_ref=plan.target,
            mode=plan.mode,
            committed=True,
        )


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> WriteBindingRegistry:
    versions = {
        "ray": "2.55.1",
        "test-write-binding": "1.0.0",
    }
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: versions[name],
    )
    _Binding.describe_calls = 0
    _Binding.execute_calls = 0
    result = WriteBindingRegistry()
    result.register(_descriptor(), _Binding)
    return result


def test_write_request_normalizes_engine_and_requires_explicit_mode() -> None:
    request = _request()

    assert request.engine == "tributo.ray_data"
    assert request.request_digest == _request().request_digest
    assert request.request_digest != _request(target="/tmp/other").request_digest
    assert (
        request.request_digest != _request(binding_id="test.ray.parquet").request_digest
    )
    with pytest.raises(ValueError):
        WriteRequest(
            engine="ray",
            target_kind="parquet",
            target="/tmp/output",
        )

    with pytest.raises(ValueError, match="top-level WriteRequest"):
        _request(options={"binding_id": "test.ray.parquet"})
    with pytest.raises(ValueError, match="mode must be a top-level WriteRequest field"):
        _request(options={"mode": "append"})


def test_write_capability_does_not_assume_empty_input_support() -> None:
    assert WriteCapability().supports_empty_input is False


def test_read_and_write_engine_aliases_are_identical() -> None:
    assert ENGINE_ALIASES["ray"] == "tributo.ray_data"
    assert ENGINE_ALIASES["daft"] == "tributo.daft"
    assert (
        WriteRequest(
            engine="daft",
            target_kind="parquet",
            target="/tmp/output",
            mode=WriteMode.OVERWRITE,
        ).engine
        == "tributo.daft"
    )


def test_gateway_plans_and_executes_without_closing_input(
    registry: WriteBindingRegistry,
) -> None:
    class TrackedDataset:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    dataset = TrackedDataset()
    handle = RayDataHandle(dataset)
    gateway = WriteGateway(registry)

    descriptor = gateway.plan(_request(), handle)
    receipt = gateway.execute(_request(), handle)

    assert descriptor.binding_id == "test.ray.parquet"
    assert receipt.committed is True
    assert _Binding.describe_calls == 1
    assert _Binding.execute_calls == 1
    assert dataset.close_calls == 0


def test_gateway_rejects_mismatched_engine_handle(
    registry: WriteBindingRegistry,
) -> None:
    with pytest.raises(
        WriteCapabilityError, match="Ray writes require a RayDataHandle"
    ):
        WriteGateway(registry).execute(_request(), DaftDataFrameHandle(object()))


def test_gateway_rejects_inconsistent_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InconsistentBinding(_Binding):
        def describe(
            self, plan: LogicalWritePlan, input_handle: Any
        ) -> WriteDescriptor:
            return _descriptor(binding_id="test.ray.parquet.other")

    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {
            "ray": "2.55.1",
            "test-write-binding": "1.0.0",
        }[name],
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), InconsistentBinding)

    with pytest.raises(WriteCapabilityError, match="descriptor is inconsistent"):
        WriteGateway(registry).plan(_request(), RayDataHandle(object()))


def test_gateway_rejects_inconsistent_target_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InconsistentTargetProvider:
        provider_id = "test.target.parquet"

        def plan(self, request: WriteRequest) -> LogicalWritePlan:
            return LogicalWritePlan(
                plan_version=1,
                provider_id=self.provider_id,
                request_digest=request.request_digest,
                engine_id=request.engine,
                target_kind=request.target_kind,
                target="/tmp/other",
                mode=request.mode,
                options=request.options,
                runtime_options=request.runtime_options,
            )

    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {
            "ray": "2.55.1",
            "test-write-binding": "1.0.0",
        }[name],
    )
    target_registry = WriteTargetRegistry(register_builtin_formats=False)
    target_registry.register("parquet", lambda: InconsistentTargetProvider())
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), _Binding)

    with pytest.raises(WriteCapabilityError, match="plan inconsistent"):
        WriteGateway(registry, target_registry).plan(
            _request(), RayDataHandle(object())
        )


def test_gateway_rejects_inconsistent_receipt_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InconsistentBinding(_Binding):
        def execute(
            self,
            plan: LogicalWritePlan,
            input_handle: Any,
            context: WriteExecutionContext,
        ) -> WriteReceipt:
            return WriteReceipt(
                request_digest=plan.request_digest,
                engine_id=plan.engine_id,
                binding_id="test.ray.parquet",
                target_kind=plan.target_kind,
                target_ref="/tmp/other",
                mode=plan.mode,
                committed=True,
            )

    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {
            "ray": "2.55.1",
            "test-write-binding": "1.0.0",
        }[name],
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), InconsistentBinding)

    with pytest.raises(WriteCapabilityError, match="receipt inconsistent"):
        WriteGateway(registry).execute(_request(), RayDataHandle(object()))


def test_gateway_rejects_inconsistent_receipt_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InconsistentBinding(_Binding):
        def execute(
            self,
            plan: LogicalWritePlan,
            input_handle: Any,
            context: WriteExecutionContext,
        ) -> WriteReceipt:
            return WriteReceipt(
                request_digest=plan.request_digest,
                engine_id=plan.engine_id,
                binding_id="test.ray.parquet",
                target_kind=plan.target_kind,
                target_ref=plan.target,
                mode=WriteMode.APPEND,
                committed=True,
            )

    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {
            "ray": "2.55.1",
            "test-write-binding": "1.0.0",
        }[name],
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), InconsistentBinding)

    with pytest.raises(WriteCapabilityError, match="receipt inconsistent"):
        WriteGateway(registry).execute(_request(), RayDataHandle(object()))


def test_gateway_rejects_unsupported_mode(
    registry: WriteBindingRegistry,
) -> None:
    request = _request(mode=WriteMode.APPEND)

    with pytest.raises(WriteCapabilityError, match="does not support mode 'append'"):
        WriteGateway(registry).plan(request, RayDataHandle(object()))


def test_gateway_sanitizes_binding_exception(
    registry: WriteBindingRegistry,
) -> None:
    class FailingBinding(_Binding):
        def execute(
            self,
            plan: LogicalWritePlan,
            input_handle: Any,
            context: WriteExecutionContext,
        ) -> WriteReceipt:
            raise RuntimeError("password=do-not-leak")

    failing_registry = WriteBindingRegistry()
    failing_registry.register(
        _descriptor(binding_id="test.ray.parquet.failing"), FailingBinding
    )

    with pytest.raises(WriteBindingError) as raised:
        WriteGateway(failing_registry).execute(
            _request(binding_id="test.ray.parquet.failing"),
            RayDataHandle(object()),
        )

    assert "do-not-leak" not in str(raised.value)
    assert raised.value.source_error_type == "RuntimeError"


def test_gateway_sanitizes_binding_capability_error(
    registry: WriteBindingRegistry,
) -> None:
    class FailingBinding(_Binding):
        def describe(
            self, plan: LogicalWritePlan, input_handle: Any
        ) -> WriteDescriptor:
            raise WriteCapabilityError("token=do-not-leak")

    failing_registry = WriteBindingRegistry()
    failing_registry.register(
        _descriptor(binding_id="test.ray.parquet.failing"), FailingBinding
    )

    with pytest.raises(WriteBindingError) as raised:
        WriteGateway(failing_registry).plan(
            _request(binding_id="test.ray.parquet.failing"),
            RayDataHandle(object()),
        )

    assert "do-not-leak" not in str(raised.value)
    assert raised.value.source_error_type == "WriteCapabilityError"


def test_gateway_sanitizes_binding_factory_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = WriteBindingRegistry()
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {
            "ray": "2.55.1",
            "test-write-binding": "1.0.0",
        }[name],
    )

    def failing_factory() -> Any:
        raise RuntimeError("password=do-not-leak")

    registry.register(_descriptor(), failing_factory)

    with pytest.raises(WriteBindingError) as raised:
        WriteGateway(registry).execute(_request(), RayDataHandle(object()))

    assert "do-not-leak" not in str(raised.value)
    assert raised.value.source_error_type == "RuntimeError"


def test_gateway_sanitizes_uri_and_delimited_credentials(
    registry: WriteBindingRegistry,
) -> None:
    class FailingBinding(_Binding):
        def execute(
            self,
            plan: LogicalWritePlan,
            input_handle: Any,
            context: WriteExecutionContext,
        ) -> WriteReceipt:
            raise RuntimeError(
                "failed s3://user:uri-secret@bucket/output?token=query-secret "
                "password=comma,secret"
            )

    failing_registry = WriteBindingRegistry()
    failing_registry.register(
        _descriptor(binding_id="test.ray.parquet.redaction"), FailingBinding
    )

    with pytest.raises(WriteBindingError) as raised:
        WriteGateway(failing_registry).execute(
            _request(binding_id="test.ray.parquet.redaction"),
            RayDataHandle(object()),
        )

    message = str(raised.value)
    assert "uri-secret" not in message
    assert "query-secret" not in message
    assert "comma,secret" not in message
    assert "<redacted>@bucket" in message
    assert "password=<redacted>" in message


def test_gateway_rejects_invalid_factory_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {
            "ray": "2.55.1",
            "test-write-binding": "1.0.0",
        }[name],
    )
    registry = WriteBindingRegistry()
    registry.register(_descriptor(), lambda: object())

    with pytest.raises(WriteCapabilityError, match="expected WriteBinding"):
        WriteGateway(registry).execute(_request(), RayDataHandle(object()))


def test_legacy_conversion_injects_overwrite_without_mutating_options() -> None:
    options: dict[str, Any] = {"compression": "zstd"}
    dataset = object()

    request, handle = ray_connector_write_request(
        dataset=dataset,
        target_kind="parquet",
        target="/tmp/output",
        options=options,
    )

    assert request.mode is WriteMode.OVERWRITE
    assert request.engine == "tributo.ray_data"
    assert request.options == {"compression": "zstd"}
    assert handle.dataset is dataset
    assert options == {"compression": "zstd"}


def test_legacy_conversion_preserves_explicit_mode() -> None:
    request, _ = ray_connector_write_request(
        dataset=object(),
        target_kind="iceberg",
        target="catalog.table",
        options={"mode": "append"},
    )

    assert request.mode is WriteMode.APPEND


def test_legacy_conversion_keeps_binding_id_out_of_native_options() -> None:
    request, _ = ray_connector_write_request(
        dataset=object(),
        target_kind="parquet",
        target="/tmp/output",
        options={"binding_id": "test.ray.parquet", "compression": "zstd"},
    )

    assert request.binding_id == "test.ray.parquet"
    assert request.options == {"compression": "zstd"}


def test_legacy_conversion_rejects_invalid_mode_with_context() -> None:
    with pytest.raises(ValueError, match="legacy write mode"):
        ray_connector_write_request(
            dataset=object(),
            target_kind="parquet",
            target="/tmp/output",
            options={"mode": "APPEND"},
        )
