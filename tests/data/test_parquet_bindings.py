"""Parquet bindings delegate to engine public APIs without materializing rows."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from tributo._common.storage_profiles import StorageProfile
from tributo.data.bindings import (
    _daft_parquet_descriptor,
    _ray_parquet_descriptor,
)
from tributo.data.bindings._shared import canonical_engine_schema
from tributo.data.bindings.daft_parquet import DaftParquetBinding
from tributo.data.bindings.ray_parquet import RayParquetBinding
from tributo.data.engine_binding import BindingCompileRequest, EngineBindings
from tributo.data.ingestion import (
    DaftDataFrameHandle,
    IngestionRuntimeContext,
    RayDataHandle,
    ReadOptions,
)
from tributo.data.scan_plan import FileScan
from tributo.data.transform_ir import TransformPipeline
from tributo.exceptions import JobConfigurationError

SCHEMA = pa.schema([("id", pa.int64()), ("score", pa.float64())])


def test_canonical_engine_schema_removes_engine_specific_differences() -> None:
    ray_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("label", pa.string()),
            pa.field("items", pa.list_(pa.string())),
            pa.field("encoded", pa.string()),
        ],
        metadata={b"ray": b"private"},
    )
    daft_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=True),
            pa.field("label", pa.large_string()),
            pa.field("items", pa.large_list(pa.large_string())),
            pa.field("encoded", pa.dictionary(pa.int32(), pa.large_string())),
        ],
        metadata={b"daft": b"private"},
    )

    assert canonical_engine_schema(ray_schema) == canonical_engine_schema(daft_schema)
    assert canonical_engine_schema(ray_schema).field("id").nullable
    assert canonical_engine_schema(ray_schema).metadata is None


def _request(
    *,
    engine_path: str = "/tmp/input.parquet",
    filesystem_id: str = "local",
    runtime_options: dict[str, Any] | None = None,
    read_options: ReadOptions | None = None,
    runtime_context: IngestionRuntimeContext | None = None,
) -> BindingCompileRequest:
    return BindingCompileRequest(
        plan=FileScan(
            provider_id="tributo.parquet",
            connector_id="parquet",
            uri=engine_path,
            filesystem_id=filesystem_id,
            options={"columns": ["id"]},
        ),
        runtime_options=runtime_options or {},
        transforms=TransformPipeline(),
        read_options=read_options or ReadOptions(),
        source_ref="0" * 64,
        runtime_context=runtime_context or IngestionRuntimeContext(),
    )


class _RayDataset:
    def schema(self) -> pa.Schema:
        return SCHEMA


class _DaftSchema:
    def to_pyarrow_schema(self) -> pa.Schema:
        return pa.schema([("id", pa.int64())])


class _DaftDataFrame:
    def __init__(self) -> None:
        self.selected: tuple[str, ...] = ()

    def select(self, *columns: str) -> "_DaftDataFrame":
        self.selected = columns
        return self

    def schema(self) -> _DaftSchema:
        return _DaftSchema()


def test_ray_binding_calls_public_reader_with_native_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    dataset = _RayDataset()

    def read_parquet(path: Any, **kwargs: Any) -> _RayDataset:
        calls.append((path, kwargs))
        return dataset

    monkeypatch.setattr("ray.data.read_parquet", read_parquet)
    monkeypatch.setattr(
        "tributo.data.bindings.ray_parquet.importlib.metadata.version",
        lambda name: "2.55.1",
    )

    result = RayParquetBinding().compile(
        _request(read_options=ReadOptions(target_parallelism=4, batch_size=128))
    )

    assert isinstance(result.handle, RayDataHandle)
    assert result.handle.dataset is dataset
    assert calls == [
        (
            "/tmp/input.parquet",
            {"columns": ["id"], "override_num_blocks": 4, "batch_size": 128},
        )
    ]
    assert result.metadata_fetched is True


def test_ray_s3_binding_only_maps_filesystem_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    filesystem = object()
    monkeypatch.setattr(
        "tributo.data.bindings.ray_parquet.pafs.S3FileSystem",
        lambda **kwargs: filesystem,
    )
    monkeypatch.setattr(
        "ray.data.read_parquet",
        lambda path, **kwargs: calls.append((path, kwargs)) or _RayDataset(),
    )
    monkeypatch.setattr(
        "tributo.data.bindings.ray_parquet.importlib.metadata.version",
        lambda name: "2.55.1",
    )

    RayParquetBinding().compile(
        _request(
            engine_path="s3://bucket/input.parquet",
            filesystem_id="s3",
            runtime_options={
                "s3": {
                    "access_key_id": "key",
                    "secret_access_key": "secret",
                    "region": "us-east-1",
                }
            },
        )
    )

    assert calls[0][0] == "bucket/input.parquet"
    assert calls[0][1]["filesystem"] is filesystem


def test_ray_s3_binding_resolves_shared_storage_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filesystem_kwargs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "tributo.data.bindings.ray_parquet.pafs.S3FileSystem",
        lambda **kwargs: filesystem_kwargs.append(kwargs) or object(),
    )
    monkeypatch.setattr("ray.data.read_parquet", lambda *args, **kwargs: _RayDataset())
    monkeypatch.setattr(
        "tributo.data.bindings.ray_parquet.importlib.metadata.version",
        lambda name: "2.55.1",
    )

    RayParquetBinding().compile(
        _request(
            engine_path="s3://bucket/input.parquet",
            filesystem_id="s3",
            runtime_options={
                "s3_profile": StorageProfile(
                    endpoint="http://minio:9000",
                    region="us-east-1",
                    access_key_id="profile-key",
                    secret_access_key="profile-secret",
                    path_style=True,
                )
            },
        )
    )

    assert filesystem_kwargs == [
        {
            "access_key": "profile-key",
            "secret_key": "profile-secret",
            "region": "us-east-1",
            "endpoint_override": "minio:9000",
            "scheme": "http",
            "force_virtual_addressing": False,
        }
    ]


def test_daft_binding_calls_public_reader_and_keeps_native_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    dataframe = _DaftDataFrame()

    def read_parquet(path: str, **kwargs: Any) -> _DaftDataFrame:
        calls.append((path, kwargs))
        return dataframe

    monkeypatch.setattr("daft.read_parquet", read_parquet)
    monkeypatch.setattr(
        "tributo.data.bindings.daft_parquet.importlib.metadata.version",
        lambda name: "0.7.21",
    )

    result = DaftParquetBinding().compile(_request())

    assert isinstance(result.handle, DaftDataFrameHandle)
    assert result.handle.dataframe is dataframe
    assert dataframe.selected == ("id",)
    assert calls == [("/tmp/input.parquet", {})]


def test_daft_binding_rejects_unmapped_read_hints() -> None:
    bindings = EngineBindings()
    bindings.register(_daft_parquet_descriptor())

    with pytest.raises(JobConfigurationError, match="target_parallelism"):
        bindings.describe(
            engine_id="tributo.daft",
            plan=_request().plan,
            read_options=ReadOptions(target_parallelism=2),
        )


def test_ray_binding_rejects_unmapped_read_hints() -> None:
    bindings = EngineBindings()
    bindings.register(_ray_parquet_descriptor())

    with pytest.raises(JobConfigurationError, match="target_split_size_bytes"):
        bindings.describe(
            engine_id="tributo.ray_data",
            plan=_request().plan,
            read_options=ReadOptions(target_split_size_bytes=1024),
        )
