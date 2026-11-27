"""Public gateway contract for the Ray Doris write adapter."""

from __future__ import annotations

import importlib.metadata
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from ray.exceptions import RayTaskError, WorkerCrashedError

from tests.support.runtime_versions import RAY_VERSION
from tributo.data import RayDataHandle
from tributo.data.writing import (
    WriteAmbiguousError,
    WriteBindingError,
    WriteBindingRegistry,
    WriteCapabilityError,
    WriteGateway,
    WriteMode,
    WriteRequest,
    default_write_gateway,
)
from tributo.data.writing.native_bindings import RayDorisWriteBinding


class _NativeAmbiguousError(Exception):
    pass


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = SimpleNamespace(
        DorisAmbiguousWriteError=_NativeAmbiguousError,
        DorisConnection=MagicMock(),
        DorisTable=MagicMock(side_effect=lambda database, table: (database, table)),
        write_doris=MagicMock(
            return_value=SimpleNamespace(
                status="success",
                batches=2,
                attempted_rows=4,
                loaded_rows=3,
                filtered_rows=1,
                uploaded_bytes=123,
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "ray_doris", module)
    return module


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> WriteGateway:
    monkeypatch.setattr(
        "tributo.data.writing.registry._installed_distribution_version",
        lambda name: {"ray": RAY_VERSION, "tributo": "1.3.0.dev0", "ray-doris": "1.3"}[
            name
        ],
    )
    registry = WriteBindingRegistry()
    registry.register(RayDorisWriteBinding._descriptor, RayDorisWriteBinding)
    return WriteGateway(registry)


def _request(**overrides: Any) -> WriteRequest:
    values: dict[str, Any] = {
        "engine": "ray",
        "target_kind": "doris",
        "target": "doris://fe:18030/db/events",
        "mode": WriteMode.APPEND,
    }
    values.update(overrides)
    return WriteRequest(**values)


def test_default_gateway_registers_doris_binding(
    native: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_version = importlib.metadata.version

    def version(name: str) -> str:
        return "1.3" if name == "ray-doris" else original_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr("tributo.data.writing.builtins._default_gateway", None)
    receipt = default_write_gateway().execute(_request(), RayDataHandle(object()))
    assert receipt.binding_id == "tributo.ray.doris"
    native.write_doris.assert_called_once()


@pytest.mark.parametrize("operation", ["load", "upsert", "partial_update"])
def test_gateway_delegates_operations_and_returns_native_receipt(
    native: SimpleNamespace,
    gateway: WriteGateway,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setenv("DORIS_TEST_PASSWORD", "secret-value")
    request = _request(
        options={
            "operation": operation,
            "mysql_port": 19030,
            "batch_rows": 2,
            "redirect_hosts": ["be"],
        },
        runtime_options={
            "user": "writer",
            "credential_ref": "env://DORIS_TEST_PASSWORD",
        },
    )
    dataset = object()
    assert (
        gateway.plan(request, RayDataHandle(dataset)).binding_id == "tributo.ray.doris"
    )
    receipt = gateway.execute(request, RayDataHandle(dataset))

    native.DorisConnection.assert_called_once_with(
        host="fe",
        http_port=18030,
        mysql_port=19030,
        username="writer",
        password_env="DORIS_TEST_PASSWORD",
        redirect_hosts=["be"],
    )
    native.write_doris.assert_called_once_with(
        dataset,
        connection=native.DorisConnection.return_value,
        table=("db", "events"),
        operation=operation,
        batch_rows=2,
    )
    assert receipt.committed is True
    assert (receipt.rows_written, receipt.bytes_written) == (3, 123)
    assert dict(receipt.metadata) == {
        "native_status": "success",
        "batches": 2,
        "attempted_rows": 4,
        "filtered_rows": 1,
    }
    assert "secret-value" not in receipt.model_dump_json()


@pytest.mark.parametrize(
    ("status", "message", "source_error_type"),
    [
        (
            "publish_timeout",
            "visibility may be delayed; wait or check visibility and do not resubmit",
            "DorisPublishTimeout",
        ),
        (
            "mixed",
            "visibility may be delayed; wait or check visibility and do not resubmit",
            "DorisPublishTimeout",
        ),
        ("unexpected", "reconcile", "DorisWriteResult"),
    ],
)
def test_gateway_does_not_confirm_unsettled_status(
    native: SimpleNamespace,
    gateway: WriteGateway,
    status: str,
    message: str,
    source_error_type: str,
) -> None:
    native.write_doris.return_value.status = status
    with pytest.raises(WriteAmbiguousError, match=message) as captured:
        gateway.execute(_request(), RayDataHandle(object()))
    assert captured.value.source_error_type == source_error_type


@pytest.mark.parametrize("wrapped", [False, True])
def test_gateway_preserves_ambiguous_native_failure(
    native: SimpleNamespace,
    gateway: WriteGateway,
    wrapped: bool,
) -> None:
    error: Exception = _NativeAmbiguousError("password=secret-value")
    if wrapped:
        error = RayTaskError("write", "traceback", error)
    native.write_doris.side_effect = error
    with pytest.raises(WriteAmbiguousError) as captured:
        gateway.execute(_request(), RayDataHandle(object()))
    assert captured.value.source_error_type == "DorisAmbiguousWriteError"
    assert "secret-value" not in str(captured.value)
    assert captured.value.__cause__ is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_gateway_marks_worker_crash_as_ambiguous(
    native: SimpleNamespace,
    gateway: WriteGateway,
    wrapped: bool,
) -> None:
    error: Exception = WorkerCrashedError()
    if wrapped:
        error = RayTaskError("write", "traceback", error)
    native.write_doris.side_effect = error

    with pytest.raises(WriteAmbiguousError, match="reconcile") as captured:
        gateway.execute(_request(), RayDataHandle(object()))

    assert captured.value.source_error_type == "WorkerCrashedError"
    assert native.write_doris.call_count == 1


def test_known_failure_keeps_binding_error_category(
    native: SimpleNamespace,
    gateway: WriteGateway,
) -> None:
    native.write_doris.side_effect = ValueError("password=secret-value")
    with pytest.raises(WriteBindingError) as captured:
        gateway.execute(_request(), RayDataHandle(object()))
    assert not isinstance(captured.value, WriteAmbiguousError)
    assert captured.value.source_error_type == "ValueError"
    assert "secret-value" not in str(captured.value)


@pytest.mark.parametrize(
    "target",
    [
        "doris://user:password@fe/db/table",
        "doris://fe/db",
        "doris://fe:0/db/table",
        "doris://fe/db/a%2Fb",
        "doris://fe/db/table?password=secret",
    ],
)
def test_invalid_targets_are_rejected_before_write(
    native: SimpleNamespace,
    gateway: WriteGateway,
    target: str,
) -> None:
    with pytest.raises((ValueError, WriteBindingError)):
        gateway.plan(_request(target=target), RayDataHandle(object()))
    native.write_doris.assert_not_called()


@pytest.mark.parametrize("mode", [WriteMode.CREATE, WriteMode.OVERWRITE])
def test_unsupported_modes_fail_before_write(
    native: SimpleNamespace,
    gateway: WriteGateway,
    mode: WriteMode,
) -> None:
    with pytest.raises(WriteCapabilityError, match="does not support mode"):
        gateway.execute(_request(mode=mode), RayDataHandle(object()))
    native.write_doris.assert_not_called()
