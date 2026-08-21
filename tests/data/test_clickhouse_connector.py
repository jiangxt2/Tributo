"""Distributed ClickHouse training reader tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pytest

from tributo.data.clickhouse import (
    ClickHouseDataConnector,
    ClickHouseDatasource,
    ClickHouseReadConfig,
)
from tributo.exceptions import EmptyDatasetError


class _ClickHouseClient:
    def __init__(
        self,
        *,
        count: int = 1,
        schema: pa.Schema | None = None,
        bounds: tuple[int, int | None, int | None] | None = None,
    ) -> None:
        self.count = count
        self.bounds = bounds
        self.schema = schema or pa.schema([pa.field("id", pa.int64())])
        self.queries: list[tuple[str, dict[str, object] | None]] = []
        self.closed = False

    def query(self, sql: str, parameters=None):
        self.queries.append((sql, parameters))
        if "min(" in sql:
            return SimpleNamespace(result_rows=[self.bounds])
        return SimpleNamespace(result_rows=[(self.count,)])

    def query_arrow(self, sql: str, parameters=None):
        self.queries.append((sql, parameters))
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in self.schema],
            schema=self.schema,
        )

    def close(self) -> None:
        self.closed = True


def test_range_tasks_are_half_open_cover_max_and_include_null_once() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    clients: list[_ClickHouseClient] = []

    def connect(**_kwargs):
        client = _ClickHouseClient(schema=schema)
        clients.append(client)
        return client

    source = ClickHouseDatasource(
        ClickHouseReadConfig(
            sql="SELECT id FROM events WHERE tenant = {tenant:String}",
            params={"tenant": "acme"},
            sort_key="id",
        ),
        total_rows=4,
        schema=schema,
        bounds=(4, 1, 3),
    )
    with patch("clickhouse_connect.get_client", side_effect=connect):
        tasks = source.get_read_tasks(parallelism=10)
        for task in tasks:
            list(task())

    assert len(tasks) == 3
    sql = [client.queries[0][0] for client in clients]
    assert "`id` >= 1 AND `id` < 2" in sql[0]
    assert "`id` IS NULL" in sql[0]
    assert "`id` >= 2 AND `id` < 3" in sql[1]
    assert "IS NULL" not in sql[1]
    assert "`id` >= 3 AND `id` < 4" in sql[2]
    assert sum("IS NULL" in query for query in sql) == 1
    assert all(client.queries[0][1] == {"tenant": "acme"} for client in clients)
    assert all(client.closed for client in clients)


def test_range_tasks_min_equals_max_create_one_complete_shard() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    client = _ClickHouseClient(schema=schema)
    source = ClickHouseDatasource(
        ClickHouseReadConfig(sql="SELECT id FROM events", sort_key="id"),
        total_rows=2,
        schema=schema,
        bounds=(2, 7, 7),
    )
    with patch("clickhouse_connect.get_client", return_value=client):
        tasks = source.get_read_tasks(parallelism=8)
        list(tasks[0]())

    assert len(tasks) == 1
    assert "`id` >= 7 AND `id` < 8" in client.queries[0][0]
    assert "`id` IS NULL" in client.queries[0][0]
    assert client.closed


def test_sparse_range_caps_tasks_by_rows_without_losing_domain_coverage() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    clients: list[_ClickHouseClient] = []

    def connect(**_kwargs):
        client = _ClickHouseClient(schema=schema)
        clients.append(client)
        return client

    source = ClickHouseDatasource(
        ClickHouseReadConfig(sql="SELECT id FROM events", sort_key="id"),
        total_rows=2,
        schema=schema,
        bounds=(2, 1, 100),
    )
    with patch("clickhouse_connect.get_client", side_effect=connect):
        tasks = source.get_read_tasks(parallelism=32)
        for task in tasks:
            list(task())

    assert len(tasks) == 2
    assert "`id` >= 1 AND `id` < 51" in clients[0].queries[0][0]
    assert "`id` >= 51 AND `id` < 101" in clients[1].queries[0][0]


def test_offset_tasks_are_deterministic_and_capped_by_rows() -> None:
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
    clients: list[_ClickHouseClient] = []

    def connect(**_kwargs):
        client = _ClickHouseClient(schema=schema)
        clients.append(client)
        return client

    source = ClickHouseDatasource(
        ClickHouseReadConfig(sql="SELECT id, name FROM events"),
        total_rows=2,
        schema=schema,
    )
    with patch("clickhouse_connect.get_client", side_effect=connect):
        tasks = source.get_read_tasks(parallelism=20)
        for task in tasks:
            list(task())

    assert len(tasks) == 2
    sql = [client.queries[0][0] for client in clients]
    assert all("ORDER BY `id`, `name`" in query for query in sql)
    assert sql[0].endswith("LIMIT 1 OFFSET 0")
    assert sql[1].endswith("LIMIT 1 OFFSET 1")
    assert all(client.closed for client in clients)


@pytest.mark.parametrize("identifier", ["id; DROP TABLE events", "id--", "a`b"])
def test_clickhouse_rejects_unsafe_sort_identifier(identifier: str) -> None:
    with pytest.raises(ValueError, match="SQL identifier"):
        ClickHouseReadConfig(sql="SELECT id FROM events", sort_key=identifier)


def test_clickhouse_rejects_zero_parallelism() -> None:
    with pytest.raises(ValueError, match="parallelism"):
        ClickHouseReadConfig(sql="SELECT id FROM events", parallelism=0)


def test_connector_driver_only_probes_count_schema_and_closes() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    count_client = _ClickHouseClient(count=3, schema=schema)
    schema_client = _ClickHouseClient(count=3, schema=schema)
    clients = iter([count_client, schema_client])

    with (
        patch("clickhouse_connect.get_client", side_effect=lambda **_: next(clients)),
        patch("ray.data.read_datasource", return_value="dataset") as read_datasource,
    ):
        result = ClickHouseDataConnector().read(
            sql="SELECT id FROM events WHERE tenant = {tenant:String}",
            params={"tenant": "acme"},
        )

    assert result == "dataset"
    assert "SELECT count()" in count_client.queries[0][0]
    assert count_client.queries[0][1] == {"tenant": "acme"}
    assert schema_client.queries[0][0].endswith("LIMIT 0")
    assert count_client.closed and schema_client.closed
    source = read_datasource.call_args.args[0]
    assert isinstance(source, ClickHouseDatasource)


def test_connector_sort_key_probes_count_min_max_and_schema() -> None:
    schema = pa.schema([pa.field("id", pa.int64())])
    bounds_client = _ClickHouseClient(bounds=(4, 1, 3), schema=schema)
    schema_client = _ClickHouseClient(schema=schema)
    clients = iter([bounds_client, schema_client])

    with (
        patch("clickhouse_connect.get_client", side_effect=lambda **_: next(clients)),
        patch("ray.data.read_datasource", return_value="dataset") as read_datasource,
    ):
        result = ClickHouseDataConnector().read(
            sql="SELECT id FROM events WHERE tenant = {tenant:String}",
            params={"tenant": "acme"},
            sort_key="id",
        )

    assert result == "dataset"
    probe_sql, probe_params = bounds_client.queries[0]
    assert "SELECT count(), min(`id`), max(`id`)" in probe_sql
    assert (
        "FROM (SELECT id FROM events WHERE tenant = {tenant:String}) AS t" in probe_sql
    )
    assert probe_params == {"tenant": "acme"}
    assert schema_client.queries[0][0].endswith("LIMIT 0")
    assert bounds_client.closed and schema_client.closed
    source = read_datasource.call_args.args[0]
    assert source._bounds == (4, 1, 3)


def test_clickhouse_empty_query_raises_common_error_and_closes() -> None:
    client = _ClickHouseClient(count=0)
    with patch("clickhouse_connect.get_client", return_value=client):
        with pytest.raises(EmptyDatasetError, match="ClickHouse"):
            ClickHouseDataConnector().read(sql="SELECT id FROM empty_table")
    assert client.closed
