"""Distributed HiveServer2 training reader tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pytest

from tributo.data.hive import (
    HiveDataConnector,
    HiveDatasource,
    HiveReadConfig,
    _hive_connection,
)
from tributo.exceptions import EmptyDatasetError


class _HiveCursor:
    def __init__(
        self,
        rows: list[tuple[object, ...]] | None = None,
        *,
        description: list[tuple[object, ...]] | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.rows = list(rows or [])
        self.description = description or [
            ("t.id", "BIGINT_TYPE", None, None, None, None, None),
            ("t.name", "STRING_TYPE", None, None, None, None, None),
        ]
        self.executions: list[tuple[str, dict[str, object] | None]] = []
        self.closed = False
        self.close_error = close_error

    def execute(self, sql: str, parameters=None) -> None:
        self.executions.append((sql, parameters))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchmany(self, size: int):
        rows, self.rows = self.rows[:size], self.rows[size:]
        return rows

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _HiveConnection:
    def __init__(
        self,
        cursor: _HiveCursor,
        *,
        cursor_error: Exception | None = None,
    ) -> None:
        self._cursor = cursor
        self.cursor_error = cursor_error
        self.closed = False

    def cursor(self) -> _HiveCursor:
        if self.cursor_error is not None:
            raise self.cursor_error
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _hive_module(connections: list[_HiveConnection]):
    return SimpleNamespace(connect=lambda **_kwargs: connections.pop(0))


@pytest.mark.parametrize("auth", ["NONE", "NOSASL"])
def test_hive_connection_passes_passwordless_auth_explicitly(auth: str) -> None:
    connection = _HiveConnection(_HiveCursor())
    captured: list[dict[str, object]] = []
    fake_pyhive = SimpleNamespace(
        hive=SimpleNamespace(
            connect=lambda **kwargs: captured.append(kwargs) or connection
        )
    )

    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        assert (
            _hive_connection(HiveReadConfig(sql="SELECT 1", auth=auth.lower()))
            is connection
        )

    assert captured == [
        {
            "host": "localhost",
            "port": 10000,
            "database": "default",
            "username": "default",
            "auth": auth,
        }
    ]


@pytest.mark.parametrize("auth", ["LDAP", "CUSTOM"])
def test_hive_connection_passes_password_for_password_auth(auth: str) -> None:
    connection = _HiveConnection(_HiveCursor())
    captured: list[dict[str, object]] = []
    fake_pyhive = SimpleNamespace(
        hive=SimpleNamespace(
            connect=lambda **kwargs: captured.append(kwargs) or connection
        )
    )
    config = HiveReadConfig(sql="SELECT 1", auth=auth, password="secret")

    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        assert _hive_connection(config) is connection

    assert captured[0]["auth"] == auth
    assert captured[0]["password"] == "secret"


@pytest.mark.parametrize("auth", ["NONE", "NOSASL"])
def test_hive_password_requires_password_auth(auth: str) -> None:
    with pytest.raises(ValueError, match="password requires auth LDAP or CUSTOM"):
        HiveReadConfig(sql="SELECT 1", auth=auth, password="secret")


@pytest.mark.parametrize("auth", ["LDAP", "CUSTOM"])
def test_hive_password_auth_requires_password(auth: str) -> None:
    with pytest.raises(ValueError, match=f"auth {auth} requires password"):
        HiveReadConfig(sql="SELECT 1", auth=auth)


def test_hive_rejects_unsupported_kerberos_auth() -> None:
    with pytest.raises(ValueError, match="auth"):
        HiveReadConfig(sql="SELECT 1", auth="KERBEROS")


def test_hive_module_import_is_lazy_when_pyhive_is_missing() -> None:
    with patch.dict(sys.modules, {"pyhive": None}):
        with pytest.raises(ImportError, match=r"tributo\[hive\]"):
            HiveDataConnector().read(sql="SELECT id FROM events")


def test_hash_tasks_are_exclusive_null_safe_streamed_and_capped() -> None:
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
    cursors = [_HiveCursor([(index, f"n{index}")]) for index in range(3)]
    all_connections = [_HiveConnection(cursor) for cursor in cursors]
    available = list(all_connections)
    source = HiveDatasource(
        HiveReadConfig(
            sql="SELECT id, name FROM events WHERE tenant = %(tenant)s",
            params={"tenant": "acme"},
            hash_column="id",
            hash_shards=3,
            batch_size=1,
        ),
        total_rows=20,
        schema=schema,
    )
    fake_pyhive = SimpleNamespace(hive=_hive_module(available))
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        tasks = source.get_read_tasks(parallelism=10)
        blocks = [block for task in tasks for block in task()]

    assert len(tasks) == 3
    assert len(blocks) == 3
    sql = [cursor.executions[0][0] for cursor in cursors]
    assert all("pmod(" in query for query in sql)
    assert all("coalesce(cast(`id` as string)" in query for query in sql)
    assert [query.rsplit("=", 1)[1].strip() for query in sql] == ["0", "1", "2"]
    assert all(cursor.executions[0][1] == {"tenant": "acme"} for cursor in cursors)
    assert all(connection.closed for connection in all_connections)


def test_hive_hash_parallelism_greater_than_rows_creates_no_empty_tasks() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    source = HiveDatasource(
        HiveReadConfig(sql="SELECT id FROM events", hash_shards=64),
        total_rows=2,
        schema=schema,
    )

    assert len(source.get_read_tasks(parallelism=100)) == 2


def test_hive_offset_tasks_use_deterministic_ordering() -> None:
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
    cursors = [_HiveCursor(), _HiveCursor()]
    all_connections = [_HiveConnection(cursor) for cursor in cursors]
    available = list(all_connections)
    source = HiveDatasource(
        HiveReadConfig(sql="SELECT id, name FROM events", shard_mode="offset"),
        total_rows=2,
        schema=schema,
    )
    fake_pyhive = SimpleNamespace(hive=_hive_module(available))
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        for task in source.get_read_tasks(parallelism=8):
            list(task())

    sql = [cursor.executions[0][0] for cursor in cursors]
    assert all("ORDER BY `id`, `name`" in query for query in sql)
    assert sql[0].endswith("LIMIT 0, 1")
    assert sql[1].endswith("LIMIT 1, 1")
    assert all(" OFFSET " not in query for query in sql)
    assert all(connection.closed for connection in all_connections)


@pytest.mark.parametrize("identifier", ["id; DROP TABLE events", "id--", "a`b"])
def test_hive_rejects_unsafe_hash_identifier(identifier: str) -> None:
    with pytest.raises(ValueError, match="SQL identifier"):
        HiveReadConfig(sql="SELECT id FROM events", hash_column=identifier)


def test_hive_rejects_zero_parallelism_and_defaults_to_64_hash_shards() -> None:
    assert HiveReadConfig(sql="SELECT id FROM events").hash_shards == 64
    with pytest.raises(ValueError, match="parallelism"):
        HiveReadConfig(sql="SELECT id FROM events", parallelism=0)


def test_hive_empty_query_raises_common_error_and_closes() -> None:
    cursor = _HiveCursor([(0,)])
    connection = _HiveConnection(cursor)
    fake_pyhive = SimpleNamespace(hive=_hive_module([connection]))
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(EmptyDatasetError, match="Hive"):
            HiveDataConnector().read(sql="SELECT id FROM empty_table")
    assert "SELECT count(*)" in cursor.executions[0][0]
    assert connection.closed


def test_hive_connector_driver_probes_count_and_schema_with_params() -> None:
    count_cursor = _HiveCursor([(2,)])
    schema_cursor = _HiveCursor()
    count_connection = _HiveConnection(count_cursor)
    schema_connection = _HiveConnection(schema_cursor)
    fake_pyhive = SimpleNamespace(
        hive=_hive_module([count_connection, schema_connection])
    )

    with (
        patch.dict(sys.modules, {"pyhive": fake_pyhive}),
        patch("ray.data.read_datasource", return_value="dataset") as read_datasource,
    ):
        result = HiveDataConnector().read(
            sql="SELECT id, name FROM events WHERE tenant = %(tenant)s",
            params={"tenant": "acme"},
            hash_column="id",
        )

    assert result == "dataset"
    assert "SELECT count(*)" in count_cursor.executions[0][0]
    assert count_cursor.executions[0][1] == {"tenant": "acme"}
    assert schema_cursor.executions[0][0].endswith("LIMIT 0")
    assert schema_cursor.executions[0][1] == {"tenant": "acme"}
    assert count_connection.closed and schema_connection.closed
    assert isinstance(read_datasource.call_args.args[0], HiveDatasource)


def test_hive_schema_uses_description_metadata_without_fetching_rows() -> None:
    description = [
        ("t.flag", "BOOLEAN_TYPE", None, None, None, None, None),
        ("t.count", "INT_TYPE", None, None, None, None, None),
        ("t.ratio", "DOUBLE_TYPE", None, None, None, None, None),
        ("t.amount", "DECIMAL_TYPE", None, None, 12, 3, None),
        ("t.day", "DATE_TYPE", None, None, None, None, None),
        ("t.created", "TIMESTAMP_TYPE", None, None, None, None, None),
        ("t.payload", "BINARY_TYPE", None, None, None, None, None),
        ("t.name", "STRING_TYPE", None, None, None, None, None),
    ]
    count_cursor = _HiveCursor([(1,)])
    schema_cursor = _HiveCursor(description=description)
    connections = [_HiveConnection(count_cursor), _HiveConnection(schema_cursor)]
    fake_pyhive = SimpleNamespace(hive=_hive_module(connections))

    with (
        patch.dict(sys.modules, {"pyhive": fake_pyhive}),
        patch("ray.data.read_datasource", return_value="dataset") as read_datasource,
    ):
        HiveDataConnector().read(sql="SELECT * FROM typed_events")

    schema = read_datasource.call_args.args[0]._schema
    assert schema == pa.schema(
        [
            pa.field("flag", pa.bool_()),
            pa.field("count", pa.int64()),
            pa.field("ratio", pa.float64()),
            pa.field("amount", pa.decimal128(12, 3)),
            pa.field("day", pa.date32()),
            pa.field("created", pa.timestamp("us")),
            pa.field("payload", pa.binary()),
            pa.field("name", pa.string()),
        ]
    )
    assert schema_cursor.rows == []


def test_hive_unknown_schema_type_fails_fast_and_closes() -> None:
    description = [
        ("t.payload", "UNION_TYPE", None, None, None, None, None),
    ]
    count_connection = _HiveConnection(_HiveCursor([(1,)]))
    schema_connection = _HiveConnection(_HiveCursor(description=description))
    fake_pyhive = SimpleNamespace(
        hive=_hive_module([count_connection, schema_connection])
    )

    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(ValueError, match="Unsupported Hive result type"):
            HiveDataConnector().read(sql="SELECT payload FROM events")

    assert count_connection.closed and schema_connection.closed


def test_hive_decimal_without_precision_scale_fails_closed() -> None:
    description = [
        ("t.amount", "DECIMAL_TYPE", None, None, None, None, None),
    ]
    count_connection = _HiveConnection(_HiveCursor([(1,)]))
    schema_connection = _HiveConnection(_HiveCursor(description=description))
    fake_pyhive = SimpleNamespace(
        hive=_hive_module([count_connection, schema_connection])
    )

    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(ValueError, match="precision and scale"):
            HiveDataConnector().read(sql="SELECT amount FROM events")

    assert count_connection.closed and schema_connection.closed


def test_hive_decimal_reads_precision_scale_from_thrift_operation_schema() -> None:
    description = [
        ("t.amount", "DECIMAL_TYPE", None, None, None, None, None),
    ]
    count_connection = _HiveConnection(_HiveCursor([(1,)]))
    schema_cursor = _HiveCursor(description=description)
    schema_connection = _HiveConnection(schema_cursor)
    schema_cursor._connection = schema_connection
    schema_cursor._operationHandle = "operation-1"
    qualifiers = SimpleNamespace(
        qualifiers={
            "precision": SimpleNamespace(i32Value=20),
            "scale": SimpleNamespace(i32Value=7),
        }
    )
    primitive = SimpleNamespace(typeQualifiers=qualifiers)
    column = SimpleNamespace(
        typeDesc=SimpleNamespace(types=[SimpleNamespace(primitiveEntry=primitive)])
    )
    schema_connection.client = SimpleNamespace(
        GetResultSetMetadata=lambda _request: SimpleNamespace(
            schema=SimpleNamespace(columns=[column])
        )
    )
    thrift_types = SimpleNamespace(TGetResultSetMetadataReq=lambda operation: operation)
    fake_pyhive = SimpleNamespace(
        hive=_hive_module([count_connection, schema_connection])
    )

    with (
        patch.dict(
            sys.modules,
            {
                "pyhive": fake_pyhive,
                "TCLIService": SimpleNamespace(ttypes=thrift_types),
            },
        ),
        patch("ray.data.read_datasource", return_value="dataset") as read_datasource,
    ):
        HiveDataConnector().read(sql="SELECT amount FROM events")

    assert read_datasource.call_args.args[0]._schema == pa.schema(
        [pa.field("amount", pa.decimal128(20, 7))]
    )


def test_hive_duplicate_unqualified_columns_fail_in_probe_and_stream() -> None:
    duplicate_description = [
        ("left.id", "INT_TYPE", None, None, None, None, None),
        ("right.id", "INT_TYPE", None, None, None, None, None),
    ]
    count_connection = _HiveConnection(_HiveCursor([(1,)]))
    probe_connection = _HiveConnection(_HiveCursor(description=duplicate_description))
    fake_pyhive = SimpleNamespace(
        hive=_hive_module([count_connection, probe_connection])
    )
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(ValueError, match="duplicate Hive result column"):
            HiveDataConnector().read(sql="SELECT * FROM joined")
    assert probe_connection.closed

    stream_cursor = _HiveCursor(
        [(1, 2)],
        description=duplicate_description,
    )
    stream_connection = _HiveConnection(stream_cursor)
    source = HiveDatasource(
        HiveReadConfig(sql="SELECT * FROM joined", hash_column="id"),
        total_rows=1,
        schema=pa.schema([pa.field("id", pa.int64())]),
    )
    fake_pyhive = SimpleNamespace(hive=_hive_module([stream_connection]))
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(ValueError, match="duplicate Hive result column"):
            list(source.get_read_tasks(1)[0]())
    assert stream_connection.closed


def test_hive_cursor_creation_and_close_failures_still_close_connection() -> None:
    cursor_creation_connection = _HiveConnection(
        _HiveCursor(), cursor_error=RuntimeError("cursor failed")
    )
    fake_pyhive = SimpleNamespace(hive=_hive_module([cursor_creation_connection]))
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(RuntimeError, match="cursor failed"):
            HiveDataConnector().read(sql="SELECT id FROM events")
    assert cursor_creation_connection.closed

    close_cursor = _HiveCursor(close_error=RuntimeError("cursor close failed"))
    close_connection = _HiveConnection(close_cursor)
    source = HiveDatasource(
        HiveReadConfig(sql="SELECT id, name FROM events", hash_shards=1),
        total_rows=1,
        schema=pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())]),
    )
    fake_pyhive = SimpleNamespace(hive=_hive_module([close_connection]))
    with patch.dict(sys.modules, {"pyhive": fake_pyhive}):
        with pytest.raises(RuntimeError, match="cursor close failed"):
            list(source.get_read_tasks(1)[0]())
    assert close_cursor.closed
    assert close_connection.closed
