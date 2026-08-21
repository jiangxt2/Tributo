"""Lazy, distributed ClickHouse reader for Ray training datasets."""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from ray.data.block import BlockMetadata
from ray.data.datasource import Datasource, ReadTask

from tributo.data.bindings._distributed_sql import (
    deterministic_order_by,
    quote_sql_identifier,
    schema_row_width,
    validate_sql_identifier,
)
from tributo.exceptions import EmptyDatasetError

if TYPE_CHECKING:
    import pyarrow as pa
    import ray.data


def _clean_query(sql: str) -> str:
    return sql.strip().rstrip(";").rstrip()


class ClickHouseReadConfig(BaseModel):
    """Configuration for a distributed ClickHouse query."""

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    port: int = 8123
    database: str = "default"
    username: str = "default"
    password: str | None = None
    sql: str
    params: dict[str, Any] = Field(default_factory=dict)
    sort_key: str | None = Field(
        default=None,
        description=(
            "Integer result column used for half-open range shards. Sparse key "
            "domains can produce uneven or empty shards; task count is capped by "
            "the result row count, without histogram probing."
        ),
    )
    parallelism: int = -1

    @field_validator("sort_key")
    @classmethod
    def _validate_sort_key(cls, value: str | None) -> str | None:
        if value is not None:
            validate_sql_identifier(value)
        return value

    @field_validator("parallelism")
    @classmethod
    def _validate_parallelism(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("parallelism must be -1 or at least 1")
        return value


def _clickhouse_client(config: ClickHouseReadConfig) -> Any:
    try:
        import clickhouse_connect
    except ImportError as exc:  # pragma: no cover - dependency is in the dev env
        raise ImportError(
            "ClickHouse data sources require tributo[clickhouse]."
        ) from exc
    return clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        database=config.database,
        username=config.username,
        password=config.password or "",
    )


class ClickHouseDatasource(Datasource):
    """Ray datasource whose tasks independently query one ClickHouse shard."""

    def __init__(
        self,
        config: ClickHouseReadConfig,
        total_rows: int,
        schema: pa.Schema,
        bounds: tuple[int, int | None, int | None] | None = None,
    ) -> None:
        self._config = config
        self._total_rows = total_rows
        self._schema = schema
        self._bounds = bounds

    def estimate_inmemory_data_size(self) -> int | None:
        return self._total_rows * schema_row_width(self._schema)

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: Any | None = None,
    ) -> list[ReadTask]:
        del per_task_row_limit, data_context
        if self._bounds is not None:
            return self._range_tasks(parallelism)
        return self._offset_tasks(parallelism)

    def _metadata(self, rows: int) -> BlockMetadata:
        return BlockMetadata(
            num_rows=rows,
            size_bytes=rows * schema_row_width(self._schema),
            exec_stats=None,
        )

    def _offset_tasks(self, parallelism: int) -> list[ReadTask]:
        task_count = min(max(1, parallelism), self._total_rows)
        rows_per_task = ceil(self._total_rows / task_count)
        tasks: list[ReadTask] = []
        for offset in range(0, self._total_rows, rows_per_task):
            limit = min(rows_per_task, self._total_rows - offset)

            def read(offset: int = offset, limit: int = limit) -> Iterator[pa.Table]:
                yield self._query_offset(offset, limit)

            tasks.append(ReadTask(read, self._metadata(limit), schema=self._schema))
        return tasks

    def _range_tasks(self, parallelism: int) -> list[ReadTask]:
        _count, minimum, maximum = self._bounds or (0, None, None)
        if minimum is None or maximum is None:

            def read_nulls() -> Iterator[pa.Table]:
                yield self._query_range(None, None, include_null=True)

            return [
                ReadTask(
                    read_nulls,
                    self._metadata(self._total_rows),
                    schema=self._schema,
                )
            ]

        span = maximum - minimum + 1
        task_count = min(max(1, parallelism), self._total_rows, span)
        width = ceil(span / task_count)
        tasks: list[ReadTask] = []
        low = minimum
        while low <= maximum:
            high = min(maximum + 1, low + width)
            include_null = not tasks

            def read(
                low: int = low,
                high: int = high,
                include_null: bool = include_null,
            ) -> Iterator[pa.Table]:
                yield self._query_range(low, high, include_null=include_null)

            estimated_rows = max(1, ceil(self._total_rows / task_count))
            tasks.append(
                ReadTask(read, self._metadata(estimated_rows), schema=self._schema)
            )
            low = high
        return tasks

    def _query_offset(self, offset: int, limit: int) -> pa.Table:
        client = _clickhouse_client(self._config)
        try:
            wrapped = (
                f"SELECT * FROM ({_clean_query(self._config.sql)}) AS t"
                f"{deterministic_order_by(self._schema)}"
                f" LIMIT {limit} OFFSET {offset}"
            )
            return client.query_arrow(
                wrapped,
                parameters=self._config.params or None,
            )
        finally:
            client.close()

    def _query_range(
        self,
        low: int | None,
        high: int | None,
        *,
        include_null: bool,
    ) -> pa.Table:
        sort_key = self._config.sort_key
        if sort_key is None:  # defensive; range mode is built only with a key
            raise ValueError("sort_key is required for range sharding")
        quoted = quote_sql_identifier(sort_key)
        conditions: list[str] = []
        if low is not None and high is not None:
            conditions.append(f"({quoted} >= {low} AND {quoted} < {high})")
        if include_null:
            conditions.append(f"{quoted} IS NULL")
        predicate = " OR ".join(conditions)
        client = _clickhouse_client(self._config)
        try:
            wrapped = (
                f"SELECT * FROM ({_clean_query(self._config.sql)}) AS t"
                f" WHERE {predicate}"
            )
            return client.query_arrow(
                wrapped,
                parameters=self._config.params or None,
            )
        finally:
            client.close()


class ClickHouseDataConnector:
    """Read-only training connector backed by ``clickhouse-connect``."""

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        config = ClickHouseReadConfig(**kwargs)
        if not _clean_query(config.sql):
            raise ValueError("missing sql in clickhouse data config")

        bounds = self._probe_bounds(config) if config.sort_key else None
        total_rows = bounds[0] if bounds is not None else self._count_rows(config)
        if total_rows <= 0:
            raise EmptyDatasetError("ClickHouse query returned empty result")
        schema = self._probe_schema(config)
        if config.sort_key is not None and config.sort_key not in schema.names:
            raise ValueError(
                f"sort_key {config.sort_key!r} not found in query result columns "
                f"{schema.names!r}"
            )

        import ray.data

        return cast(
            "ray.data.Dataset",
            ray.data.read_datasource(
                ClickHouseDatasource(config, total_rows, schema, bounds),
                parallelism=config.parallelism,
            ),
        )

    def _count_rows(self, config: ClickHouseReadConfig) -> int:
        client = _clickhouse_client(config)
        try:
            result = client.query(
                f"SELECT count() FROM ({_clean_query(config.sql)}) AS t",
                parameters=config.params or None,
            )
            rows = getattr(result, "result_rows", None)
            return int(rows[0][0]) if rows and rows[0] else 0
        finally:
            client.close()

    def _probe_bounds(
        self, config: ClickHouseReadConfig
    ) -> tuple[int, int | None, int | None]:
        sort_key = quote_sql_identifier(config.sort_key or "")
        client = _clickhouse_client(config)
        try:
            result = client.query(
                f"SELECT count(), min({sort_key}), max({sort_key}) "
                f"FROM ({_clean_query(config.sql)}) AS t",
                parameters=config.params or None,
            )
            rows = getattr(result, "result_rows", None)
            if not rows or not rows[0]:
                return (0, None, None)
            count, minimum, maximum = rows[0]
            if minimum is not None and not isinstance(minimum, int):
                raise ValueError("ClickHouse sort_key must contain integer values")
            if maximum is not None and not isinstance(maximum, int):
                raise ValueError("ClickHouse sort_key must contain integer values")
            return int(count), minimum, maximum
        finally:
            client.close()

    def _probe_schema(self, config: ClickHouseReadConfig) -> pa.Schema:
        client = _clickhouse_client(config)
        try:
            table = client.query_arrow(
                f"SELECT * FROM ({_clean_query(config.sql)}) AS t LIMIT 0",
                parameters=config.params or None,
            )
            return table.schema
        finally:
            client.close()

    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        raise NotImplementedError(
            "ClickHouse writes are outside the training reader API"
        )
