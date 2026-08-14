"""Contract tests for the built-in native write adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from tributo.data import DaftDataFrameHandle, RayDataHandle
from tributo.data.base import S3Config
from tributo.data.contracts.modes import WriteMode
from tributo.data.writing import (
    GenericWriteTargetProvider,
    WriteExecutionContext,
    WriteRequest,
    default_write_gateway,
)
from tributo.data.writing.native_bindings import (
    DaftLanceWriteBinding,
    RayLanceWriteBinding,
)


class _RayDataset:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def write_parquet(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("parquet", args, kwargs))

    def write_csv(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("csv", args, kwargs))

    def schema(self) -> pa.Schema:
        return pa.schema([("id", pa.int64())])


class _DaftDataFrame:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def write_parquet(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("parquet", args, kwargs))

    def write_csv(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("csv", args, kwargs))

    def write_lance(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("lance", args, kwargs))


def _execute_optional_binding(
    binding: Any,
    request: WriteRequest,
    handle: RayDataHandle | DaftDataFrameHandle,
) -> Any:
    plan = GenericWriteTargetProvider(request.target_kind).plan(request)
    context = WriteExecutionContext(
        request_digest=request.request_digest,
        runtime_options=request.runtime_options,
    )
    return binding.execute(plan, handle, context)


def test_lance_bindings_declare_native_dependency_distributions() -> None:
    assert RayLanceWriteBinding._descriptor.dependency_distributions == (
        "lance-ray",
        "pylance",
    )
    assert DaftLanceWriteBinding._descriptor.dependency_distributions == (
        "pylance",
        "daft-lance",
    )


@pytest.mark.parametrize(
    ("engine", "target_kind", "handle_type", "method", "mode_key"),
    [
        ("ray", "parquet", RayDataHandle, "parquet", "mode"),
        ("ray", "csv", RayDataHandle, "csv", "mode"),
        ("daft", "parquet", DaftDataFrameHandle, "parquet", "write_mode"),
        ("daft", "csv", DaftDataFrameHandle, "csv", "write_mode"),
    ],
)
def test_builtin_binding_calls_only_native_writer(
    tmp_path: Path,
    engine: str,
    target_kind: str,
    handle_type: type,
    method: str,
    mode_key: str,
) -> None:
    native = _RayDataset() if engine == "ray" else _DaftDataFrame()
    handle = handle_type(native)
    request = WriteRequest(
        engine=engine,
        target_kind=target_kind,
        target=str(tmp_path / target_kind),
        mode=WriteMode.OVERWRITE,
        options={"compression": "zstd"} if target_kind == "parquet" else {},
    )

    receipt = default_write_gateway().execute(request, handle)

    assert receipt.committed is True
    assert native.calls[0][0] == method
    observed_mode = native.calls[0][2][mode_key]
    assert (observed_mode.value if engine == "ray" else observed_mode) == "overwrite"


def test_daft_lance_binding_delegates_to_native_writer(tmp_path: Path) -> None:
    native = _DaftDataFrame()
    request = WriteRequest(
        engine="daft",
        target_kind="lance",
        target=str(tmp_path / "lance"),
        mode=WriteMode.OVERWRITE,
    )

    receipt = _execute_optional_binding(
        DaftLanceWriteBinding(), request, DaftDataFrameHandle(native)
    )

    assert receipt.committed is True
    assert native.calls == [
        (
            "lance",
            (str(tmp_path / "lance"),),
            {"mode": "overwrite", "io_config": None},
        )
    ]


@pytest.mark.parametrize("mode", tuple(WriteMode))
def test_ray_lance_binding_delegates_common_contract_to_lance_ray(
    mode: WriteMode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_lance = MagicMock()
    monkeypatch.setitem(
        sys.modules, "lance_ray", SimpleNamespace(write_lance=write_lance)
    )
    native = _RayDataset()
    target = f"file://{tmp_path / 'output.lance'}"
    request = WriteRequest(
        engine="ray",
        target_kind="lance",
        target=target,
        binding_id="tributo.ray.lance",
        mode=mode,
        options={
            "min_rows_per_file": 10,
            "max_rows_per_file": 20,
            "data_storage_version": "2.1",
        },
    )

    receipt = _execute_optional_binding(
        RayLanceWriteBinding(), request, RayDataHandle(native)
    )

    write_lance.assert_called_once_with(
        native,
        str(tmp_path / "output.lance"),
        mode=mode.value,
        min_rows_per_file=10,
        max_rows_per_file=20,
        data_storage_version="2.1",
        storage_options=None,
        stream=False,
    )
    assert receipt.binding_id == "tributo.ray.lance"
    assert receipt.diagnostics == ("lance_ray.write_lance",)


def test_ray_lance_binding_preserves_s3_uri_and_runtime_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_lance = MagicMock()
    monkeypatch.setitem(
        sys.modules, "lance_ray", SimpleNamespace(write_lance=write_lance)
    )
    native = _RayDataset()
    request = WriteRequest(
        engine="ray",
        target_kind="lance",
        target="s3://bucket/output.lance",
        binding_id="tributo.ray.lance",
        mode=WriteMode.CREATE,
        runtime_options={
            "s3": S3Config(
                endpoint="http://minio:9000",
                access_key_id="access-key",
                secret_access_key="secret-key",
            )
        },
    )

    receipt = _execute_optional_binding(
        RayLanceWriteBinding(), request, RayDataHandle(native)
    )

    assert write_lance.call_args.args == (native, "s3://bucket/output.lance")
    storage_options = write_lance.call_args.kwargs["storage_options"]
    assert storage_options["access_key_id"] == "access-key"
    assert storage_options["secret_access_key"] == "secret-key"
    assert "access-key" not in receipt.model_dump_json()
    assert "secret-key" not in receipt.model_dump_json()


@pytest.mark.parametrize(
    ("engine", "target_kind", "handle_type", "method", "mode_key"),
    [
        ("ray", "parquet", RayDataHandle, "parquet", "mode"),
        ("ray", "csv", RayDataHandle, "csv", "mode"),
        ("daft", "parquet", DaftDataFrameHandle, "parquet", "write_mode"),
        ("daft", "csv", DaftDataFrameHandle, "csv", "write_mode"),
    ],
)
def test_builtin_file_binding_preserves_explicit_append(
    tmp_path: Path,
    engine: str,
    target_kind: str,
    handle_type: type,
    method: str,
    mode_key: str,
) -> None:
    native = _RayDataset() if engine == "ray" else _DaftDataFrame()
    request = WriteRequest(
        engine=engine,
        target_kind=target_kind,
        target=str(tmp_path / f"{target_kind}-append"),
        mode=WriteMode.APPEND,
        options={"compression": "zstd"} if target_kind == "parquet" else {},
    )

    receipt = default_write_gateway().execute(request, handle_type(native))

    assert receipt.committed is True
    assert native.calls[0][0] == method
    observed_mode = native.calls[0][2][mode_key]
    assert (observed_mode.value if engine == "ray" else observed_mode) == "append"


def test_native_binding_receives_engine_runtime_credentials_without_receipt_leak(
    tmp_path: Path,
) -> None:
    native = _RayDataset()
    request = WriteRequest(
        engine="ray",
        target_kind="parquet",
        target=f"s3://bucket/{tmp_path.name}",
        mode=WriteMode.OVERWRITE,
        runtime_options={
            "s3": S3Config(
                endpoint="http://minio:9000",
                access_key_id="access-key",
                secret_access_key="secret-key",
            )
        },
    )

    receipt = default_write_gateway().execute(request, RayDataHandle(native))

    assert "access-key" not in repr(receipt)
    assert "secret-key" not in receipt.model_dump_json()
    assert request.credential_free_runtime_options == {
        "s3": {"endpoint": "http://minio:9000"}
    }
