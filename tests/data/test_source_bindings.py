"""Source Bindings delegate scans to public engine or connector APIs."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

from tributo.data.bindings._postgresql import compile_table_query
from tributo.data.bindings._sql_shared import resolve_sql_target
from tributo.data.bindings.daft_csv import DaftCsvBinding
from tributo.data.bindings.daft_iceberg import DaftIcebergBinding
from tributo.data.bindings.daft_lance import DaftLanceBinding
from tributo.data.bindings.daft_olap import (
    DaftClickHouseBinding,
    DaftDorisBinding,
)
from tributo.data.bindings.daft_postgresql import DaftPostgreSqlBinding
from tributo.data.bindings.ray_csv import RayCsvBinding
from tributo.data.bindings.ray_doris import RayDorisBinding
from tributo.data.bindings.ray_hdfs import (
    RayHdfsCsvBinding,
    RayHdfsParquetBinding,
)
from tributo.data.bindings.ray_iceberg import RayIcebergBinding
from tributo.data.bindings.ray_lance import RayLanceBinding
from tributo.data.bindings.ray_postgresql import RayPostgreSqlBinding
from tributo.data.engine_binding import (
    BindingCompileRequest,
    BindingStageError,
    EngineBinding,
)
from tributo.data.ingestion import (
    DaftDataFrameHandle,
    IngestionRuntimeContext,
    RayDataHandle,
    ReadOptions,
)
from tributo.data.scan_plan import (
    CatalogTableRef,
    FileScan,
    NumericVersionRef,
    SnapshotVersionRef,
    SqlScan,
    SqlShardMode,
    SqlShardRequirement,
    SqlTableRead,
    TableScan,
    UriTableRef,
)
from tributo.data.transform_ir import TransformPipeline
from tributo.exceptions import JobConfigurationError

_SCHEMA = pa.schema([("id", pa.int64())])


class _RayDataset:
    def schema(self, fetch_if_missing: bool = True) -> pa.Schema:
        del fetch_if_missing
        return _SCHEMA


class _DaftSchema:
    def to_pyarrow_schema(self) -> pa.Schema:
        return _SCHEMA


class _DaftDataFrame:
    def __init__(self) -> None:
        self.selected: tuple[str, ...] = ()
        self.predicates: tuple[str, ...] = ()

    def where(self, predicate: str) -> "_DaftDataFrame":
        self.predicates = (*self.predicates, predicate)
        return self

    def select(self, *columns: str) -> "_DaftDataFrame":
        self.selected = columns
        return self

    def schema(self) -> _DaftSchema:
        return _DaftSchema()


def _request(
    plan: FileScan | TableScan | SqlScan,
    *,
    runtime_options: dict[str, Any] | None = None,
    read_options: ReadOptions | None = None,
) -> BindingCompileRequest:
    return BindingCompileRequest(
        plan=plan,
        runtime_options=runtime_options or {},
        transforms=TransformPipeline(),
        read_options=read_options or ReadOptions(),
        source_ref="0" * 64,
        runtime_context=IngestionRuntimeContext(),
    )


@pytest.mark.parametrize(
    ("binding", "engine_version", "handle_type"),
    [
        (RayCsvBinding(), "2.55.1", RayDataHandle),
        (DaftCsvBinding(), "0.7.21", DaftDataFrameHandle),
    ],
)
def test_csv_bindings_use_public_engine_readers(
    monkeypatch: pytest.MonkeyPatch,
    binding: EngineBinding,
    engine_version: str,
    handle_type: type[RayDataHandle] | type[DaftDataFrameHandle],
) -> None:
    ray_calls: list[tuple[str, dict[str, Any]]] = []
    daft_calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "ray.data.read_csv",
        lambda path, **kwargs: ray_calls.append((path, kwargs)) or _RayDataset(),
    )
    monkeypatch.setattr(
        "daft.read_csv",
        lambda path, **kwargs: daft_calls.append((path, kwargs)) or _DaftDataFrame(),
    )
    monkeypatch.setattr(
        f"{type(binding).__module__}.importlib.metadata.version",
        lambda name: engine_version,
    )
    plan = FileScan(
        provider_id="tributo.csv",
        connector_id="csv",
        uri="/data/input.csv",
        filesystem_id="local",
        options={"columns": ["id"]},
    )

    result = binding.compile(_request(plan))

    assert isinstance(result.handle, handle_type)
    assert result.reader_api in {"ray.data.read_csv", "daft.read_csv"}
    assert len(ray_calls) + len(daft_calls) == 1


@pytest.mark.parametrize(
    ("binding", "connector_id", "reader_name"),
    [
        (RayHdfsParquetBinding(), "parquet", "read_parquet"),
        (RayHdfsCsvBinding(), "csv", "read_csv"),
    ],
)
def test_hdfs_bindings_use_pyarrow_filesystem_and_ray_reader(
    monkeypatch: pytest.MonkeyPatch,
    binding: EngineBinding,
    connector_id: str,
    reader_name: str,
) -> None:
    filesystem = object()
    calls: list[tuple[str, dict[str, Any]]] = []

    class _HadoopFileSystem:
        @staticmethod
        def from_uri(uri: str) -> tuple[object, str]:
            return filesystem, "/warehouse/input"

    monkeypatch.setattr(
        "tributo.data.bindings.ray_hdfs.pafs.HadoopFileSystem",
        _HadoopFileSystem,
    )
    monkeypatch.setattr(
        f"ray.data.{reader_name}",
        lambda path, **kwargs: calls.append((path, kwargs)) or _RayDataset(),
    )
    monkeypatch.setattr(
        "tributo.data.bindings.ray_hdfs.importlib.metadata.version",
        lambda name: "2.55.1",
    )
    plan = FileScan(
        provider_id=f"tributo.{connector_id}",
        connector_id=connector_id,
        uri=f"hdfs://namenode/warehouse/input.{connector_id}",
        filesystem_id="hdfs",
        options={"columns": ["id"]},
    )

    result = binding.compile(
        _request(plan, read_options=ReadOptions(target_parallelism=4))
    )

    assert isinstance(result.handle, RayDataHandle)
    assert calls[0][0] == "/warehouse/input"
    assert calls[0][1]["filesystem"] is filesystem
    assert calls[0][1]["override_num_blocks"] == 4
    assert result.transport_id == "hdfs"


def _iceberg_plan() -> TableScan:
    return TableScan(
        provider_id="tributo.iceberg",
        connector_id="iceberg",
        table=CatalogTableRef(
            catalog_id="prod", namespace=("analytics",), table="events"
        ),
        options={"selected_fields": ["id"]},
    )


def test_ray_iceberg_binding_delegates_catalog_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "ray.data.read_iceberg",
        lambda **kwargs: calls.append(kwargs) or _RayDataset(),
    )
    monkeypatch.setattr(
        "tributo.data.bindings.ray_iceberg.importlib.metadata.version",
        lambda name: "2.55.1",
    )

    result = RayIcebergBinding().compile(
        _request(
            _iceberg_plan(),
            runtime_options={
                "catalog_name": "prod",
                "catalog_properties": {"type": "rest", "uri": "http://catalog"},
            },
        )
    )

    assert isinstance(result.handle, RayDataHandle)
    assert calls[0]["table_identifier"] == "analytics.events"
    assert calls[0]["selected_fields"] == ("id",)
    assert calls[0]["catalog_kwargs"]["name"] == "prod"
    assert result.reader_api == "ray.data.read_iceberg"


def test_daft_iceberg_binding_delegates_catalog_and_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("daft")
    pytest.importorskip("pyiceberg")
    loaded_tables: list[str] = []
    read_calls: list[tuple[Any, dict[str, Any]]] = []
    table = object()

    class _Catalog:
        def load_table(self, identifier: str) -> object:
            loaded_tables.append(identifier)
            return table

    catalog_module = ModuleType("pyiceberg.catalog")
    catalog_module.load_catalog = lambda *args, **kwargs: _Catalog()
    monkeypatch.setitem(sys.modules, "pyiceberg.catalog", catalog_module)
    monkeypatch.setattr(
        "daft.read_iceberg",
        lambda value, **kwargs: read_calls.append((value, kwargs)) or _DaftDataFrame(),
    )
    monkeypatch.setattr(
        "tributo.data.bindings.daft_iceberg.importlib.metadata.version",
        lambda name: "0.7.21",
    )

    plan = _iceberg_plan()
    plan = TableScan(
        provider_id=plan.provider_id,
        connector_id=plan.connector_id,
        table=plan.table,
        options={"selected_fields": ["id"], "row_filter": "id > 0"},
    )
    result = DaftIcebergBinding().compile(
        _request(
            plan,
            runtime_options={
                "catalog_name": "prod",
                "catalog_properties": {"type": "rest", "uri": "http://catalog"},
            },
        )
    )

    assert isinstance(result.handle, DaftDataFrameHandle)
    assert loaded_tables == ["analytics.events"]
    assert read_calls[0][0] is table
    assert result.handle.dataframe.predicates == ("id > 0",)
    assert result.handle.dataframe.selected == ("id",)
    assert any("Daft lazy residual filter" in item for item in result.diagnostics)
    assert result.reader_api == "daft.read_iceberg"


@pytest.mark.parametrize(
    ("binding", "engine_version", "handle_type"),
    [
        (RayLanceBinding(), "2.55.1", RayDataHandle),
        (DaftLanceBinding(), "0.7.21", DaftDataFrameHandle),
    ],
)
def test_lance_bindings_use_public_engine_readers(
    monkeypatch: pytest.MonkeyPatch,
    binding: EngineBinding,
    engine_version: str,
    handle_type: type[RayDataHandle] | type[DaftDataFrameHandle],
) -> None:
    ray_calls: list[tuple[str, dict[str, Any]]] = []
    daft_calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "ray.data.read_lance",
        lambda uri, **kwargs: ray_calls.append((uri, kwargs)) or _RayDataset(),
    )
    monkeypatch.setattr(
        "daft.read_lance",
        lambda uri, **kwargs: daft_calls.append((uri, kwargs)) or _DaftDataFrame(),
    )
    monkeypatch.setattr(
        f"{type(binding).__module__}.importlib.metadata.version",
        lambda name: engine_version,
    )
    plan = TableScan(
        provider_id="tributo.lance",
        connector_id="lance",
        table=UriTableRef(uri="/data/table.lance"),
        version_ref=NumericVersionRef(version=7),
        options={"columns": ["id"], "filter": "id > 0"},
    )

    result = binding.compile(_request(plan))

    assert isinstance(result.handle, handle_type)
    assert result.reader_api in {"ray.data.read_lance", "daft.read_lance"}
    assert len(ray_calls) + len(daft_calls) == 1


@pytest.mark.parametrize("binding", [RayLanceBinding(), DaftLanceBinding()])
def test_lance_bindings_reject_iceberg_snapshot_refs(
    binding: EngineBinding,
) -> None:
    plan = TableScan(
        provider_id="tributo.lance",
        connector_id="lance",
        table=UriTableRef(uri="/data/table.lance"),
        version_ref=SnapshotVersionRef(snapshot_id=7),
    )

    with pytest.raises(BindingStageError) as exc_info:
        binding.compile(_request(plan))

    assert exc_info.value.diagnostic_code == "unsupported_lance_snapshot_ref"


def test_daft_olap_default_auto_sharding_is_delegated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    module = ModuleType("daft_olap")
    module.read_clickhouse = lambda **kwargs: calls.append(kwargs) or _DaftDataFrame()
    module.read_doris = lambda **kwargs: _DaftDataFrame()
    monkeypatch.setitem(sys.modules, "daft_olap", module)
    monkeypatch.setattr(
        "tributo.data.bindings.daft_olap.importlib.metadata.version",
        lambda name: "0.7.21",
    )
    plan = SqlScan(
        provider_id="tributo.clickhouse",
        connector_id="clickhouse",
        target=SqlTableRead(schema="analytics", table="events"),
        sharding=SqlShardRequirement(mode=SqlShardMode.AUTO),
    )

    DaftClickHouseBinding().compile(
        _request(
            plan,
            runtime_options={
                "host": "db.example",
                "port": 8123,
                "database": "analytics",
            },
        )
    )

    assert calls[0]["split"] == "auto"


def test_daft_olap_single_read_rejects_parallelism_with_actionable_modes() -> None:
    plan = SqlScan(
        provider_id="tributo.clickhouse",
        connector_id="clickhouse",
        target=SqlTableRead(schema="analytics", table="events"),
    )

    with pytest.raises(
        BindingStageError,
        match=("partitioning.mode to 'auto' or 'parallel'.*remove target_parallelism"),
    ) as exc_info:
        DaftClickHouseBinding().compile(
            _request(plan, read_options=ReadOptions(target_parallelism=4))
        )

    assert exc_info.value.diagnostic_code == "single_sql_read_rejects_parallelism_hint"


def test_sql_binding_port_error_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRIBUTO_POSTGRESQL_PORT", "not-a-port")
    plan = SqlScan(
        provider_id="tributo.postgresql",
        connector_id="postgresql",
        target=SqlTableRead(schema="public", table="events"),
    )

    with pytest.raises(JobConfigurationError, match="port must be an integer"):
        resolve_sql_target(plan, {"host": "postgres", "database": "analytics"})


def _sql_plan(connector_id: str) -> SqlScan:
    return SqlScan(
        provider_id=f"tributo.{connector_id}",
        connector_id=connector_id,
        target=SqlTableRead(schema="analytics", table="events", projection=("id",)),
        sharding=SqlShardRequirement(
            mode=SqlShardMode.PARALLEL,
            columns=("id",),
            target_partitions=6,
        ),
    )


@pytest.mark.parametrize(
    ("binding", "connector_id", "reader_name", "transport_id"),
    [
        (
            DaftClickHouseBinding(),
            "clickhouse",
            "read_clickhouse",
            "clickhouse_native",
        ),
        (DaftDorisBinding(), "doris", "read_doris", "mysql"),
    ],
)
def test_daft_olap_bindings_delegate_to_external_connector(
    monkeypatch: pytest.MonkeyPatch,
    binding: EngineBinding,
    connector_id: str,
    reader_name: str,
    transport_id: str,
) -> None:
    calls: list[dict[str, Any]] = []
    module = ModuleType("daft_olap")

    def reader(**kwargs: Any) -> _DaftDataFrame:
        calls.append(kwargs)
        return _DaftDataFrame()

    module.read_clickhouse = reader
    module.read_doris = reader
    monkeypatch.setitem(sys.modules, "daft_olap", module)
    monkeypatch.setattr(
        "tributo.data.bindings.daft_olap.importlib.metadata.version",
        lambda name: "0.7.21",
    )

    result = binding.compile(
        _request(
            _sql_plan(connector_id),
            runtime_options={
                "host": "db.example",
                "port": 8123 if connector_id == "clickhouse" else 9030,
                "database": "analytics",
                "user": "reader",
                "password": "secret",
                "protocol": "mysql",
            },
            read_options=ReadOptions(batch_size=128),
        )
    )

    assert isinstance(result.handle, DaftDataFrameHandle)
    assert calls[0]["host"] == "db.example"
    assert calls[0]["table"] == "events"
    assert calls[0]["target_tasks"] == 6
    assert result.reader_api == f"daft_olap.{reader_name}"
    assert result.transport_id == transport_id


def test_ray_doris_binding_delegates_to_external_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    module = ModuleType("ray_doris")
    module.read_doris = lambda **kwargs: calls.append(kwargs) or _RayDataset()
    monkeypatch.setitem(sys.modules, "ray_doris", module)
    monkeypatch.setattr(
        "tributo.data.bindings.ray_doris.importlib.metadata.version",
        lambda name: "2.55.1",
    )

    result = RayDorisBinding().compile(
        _request(
            _sql_plan("doris"),
            runtime_options={
                "host": "doris.example",
                "port": 9030,
                "database": "analytics",
                "user": "reader",
                "password": "secret",
                "protocol": "mysql",
            },
            read_options=ReadOptions(target_parallelism=2, batch_size=128),
        )
    )

    assert isinstance(result.handle, RayDataHandle)
    assert calls[0]["table"] == "analytics.events"
    assert calls[0]["override_num_blocks"] == 6
    assert calls[0]["batch_size"] == 128
    assert result.reader_api == "ray_doris.read_doris"
    assert result.transport_id == "mysql"


@pytest.mark.parametrize(
    ("binding", "engine_version", "handle_type"),
    [
        (RayPostgreSqlBinding(), "2.55.1", RayDataHandle),
        (DaftPostgreSqlBinding(), "0.7.21", DaftDataFrameHandle),
    ],
)
def test_postgresql_bindings_delegate_to_public_sql_readers(
    monkeypatch: pytest.MonkeyPatch,
    binding: EngineBinding,
    engine_version: str,
    handle_type: type[RayDataHandle] | type[DaftDataFrameHandle],
) -> None:
    ray_calls: list[tuple[str, Any, dict[str, Any]]] = []
    daft_calls: list[tuple[str, Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        "ray.data.read_sql",
        lambda query, factory, **kwargs: (
            ray_calls.append((query, factory, kwargs)) or _RayDataset()
        ),
    )
    monkeypatch.setattr(
        "daft.read_sql",
        lambda query, factory, **kwargs: (
            daft_calls.append((query, factory, kwargs)) or _DaftDataFrame()
        ),
    )
    monkeypatch.setattr(
        f"{type(binding).__module__}.importlib.metadata.version",
        lambda name: engine_version,
    )
    parallel = isinstance(binding, DaftPostgreSqlBinding)
    plan = SqlScan(
        provider_id="tributo.postgresql",
        connector_id="postgresql",
        target=SqlTableRead(
            schema="feature_store",
            table='event"facts',
            projection=("id",),
        ),
        sharding=(
            SqlShardRequirement(
                mode=SqlShardMode.PARALLEL,
                columns=("id",),
                target_partitions=6,
            )
            if parallel
            else SqlShardRequirement()
        ),
        options={"partition_bound_strategy": "min-max"},
    )

    result = binding.compile(
        _request(
            plan,
            runtime_options={
                "host": "postgres.example",
                "port": 5432,
                "database": "analytics",
                "user": "reader",
                "password": "top-secret",
            },
        )
    )

    assert isinstance(result.handle, handle_type)
    calls = ray_calls or daft_calls
    assert calls[0][0] == ('SELECT "id" FROM "feature_store"."event""facts"')
    assert "top-secret" not in repr(calls[0][1])
    assert result.reader_api in {"ray.data.read_sql", "daft.read_sql"}
    if daft_calls:
        assert daft_calls[0][2]["partition_col"] == "id"
        assert daft_calls[0][2]["num_partitions"] == 6


def test_ray_postgresql_parallel_read_fails_closed() -> None:
    plan = SqlScan(
        provider_id="tributo.postgresql",
        connector_id="postgresql",
        target=SqlTableRead(schema="public", table="events"),
        sharding=SqlShardRequirement(
            mode=SqlShardMode.PARALLEL,
            columns=("id",),
            target_partitions=6,
        ),
    )

    with pytest.raises(BindingStageError) as exc_info:
        RayPostgreSqlBinding().compile(_request(plan))

    assert exc_info.value.diagnostic_code == "unsupported_postgresql_parallel_read"
    assert "select Daft" in str(exc_info.value)


def test_postgresql_query_compiler_rejects_raw_query_plan() -> None:
    from tributo.data.scan_plan import ParameterizedQuery

    plan = SqlScan(
        provider_id="tributo.postgresql",
        connector_id="postgresql",
        target=ParameterizedQuery(query_digest="0" * 64),
    )

    with pytest.raises(JobConfigurationError, match="structured SqlTableRead"):
        compile_table_query(plan)


def test_postgresql_database_env_is_not_confused_with_table_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRIBUTO_POSTGRESQL_DB", "warehouse")
    plan = SqlScan(
        provider_id="tributo.postgresql",
        connector_id="postgresql",
        target=SqlTableRead(schema="feature_store", table="events"),
    )

    target = resolve_sql_target(plan, {"host": "postgres.example"})

    assert target.database == "warehouse"
    assert plan.target.schema == "feature_store"
