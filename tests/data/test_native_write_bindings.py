"""Contract tests for the built-in native write adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from tributo.data import DaftDataFrameHandle, RayDataHandle
from tributo.data.base import S3Config
from tributo.data.contracts.modes import WriteMode
from tributo.data.writing import WriteRequest, default_write_gateway
from tributo.data.writing.native_bindings import RayLanceWriteBinding


class _RayDataset:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def write_parquet(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("parquet", args, kwargs))

    def write_csv(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("csv", args, kwargs))

    def write_lance(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("lance", args, kwargs))

    def schema(self) -> pa.Schema:
        return pa.schema([("id", pa.int64())])


class _DaftDataFrame:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def write_parquet(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("parquet", args, kwargs))

    def write_csv(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("csv", args, kwargs))


@pytest.mark.parametrize(
    ("engine", "target_kind", "handle_type", "method", "mode_key"),
    [
        ("ray", "parquet", RayDataHandle, "parquet", "mode"),
        ("ray", "csv", RayDataHandle, "csv", "mode"),
        ("ray", "lance", RayDataHandle, "lance", "mode"),
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
    if (
        engine == "ray"
        and target_kind == "lance"
        and not RayLanceWriteBinding.is_available()
    ):
        pytest.skip("installed pylance does not support Ray's Lance sink contract")
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
