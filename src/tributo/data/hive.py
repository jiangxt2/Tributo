"""Lazy, distributed HiveServer2 reader for Ray training datasets."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import ceil
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


class HiveReadConfig(BaseModel):
    """Configuration for a distributed HiveServer2 query."""

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    port: int = 10000
    database: str = "default"
    username: str = "default"
    password: str | None = None
    auth: Literal["NONE", "NOSASL", "LDAP", "CUSTOM"] = "NONE"
    sql: str
    params: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=10_000, ge=1)
    shard_mode: Literal["auto", "hash", "offset"] = "auto"
    hash_column: str | None = Field(
        default=None,
        description=(
            "Stable, preferably high-cardinality result column. Low-cardinality "
            "values and NULL-heavy columns remain correct but can concentrate rows "
            "in a small number of buckets."
        ),
    )
    hash_shards: int = Field(
        default=64,
        ge=1,
        description="Maximum hash buckets and therefore maximum scan amplification.",
    )
    parallelism: int = -1

    @field_validator("hash_column")
    @classmethod
    def _validate_hash_column(cls, value: str | None) -> str | None:
        if value is not None:
            validate_sql_identifier(value)
        return value

    @field_validator("auth", mode="before")
    @classmethod
    def _normalize_auth(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_auth_credentials(self) -> HiveReadConfig:
        has_password = self.password not in (None, "")
        if has_password and self.auth not in {"LDAP", "CUSTOM"}:
            raise ValueError("Hive password requires auth LDAP or CUSTOM")
        if not has_password and self.auth in {"LDAP", "CUSTOM"}:
            raise ValueError(f"Hive auth {self.auth} requires password")
        return self

    @field_validator("parallelism")
    @classmethod
    def _validate_parallelism(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("parallelism must be -1 or at least 1")
        return value


def _hive_connection(config: HiveReadConfig) -> Any:
    try:
        from pyhive import hive
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("Hive data sources require tributo[hive].") from exc
    kwargs: dict[str, object] = {
        "host": config.host,
        "port": config.port,
        "database": config.database,
        "username": config.username,
        "auth": config.auth,
    }
    if config.password not in (None, ""):
        kwargs["password"] = config.password
    return hive.connect(
        **kwargs,
    )


@contextmanager
def _hive_cursor(config: HiveReadConfig) -> Iterator[Any]:
    """Yield a cursor while guaranteeing its connection is always closed."""
    connection = _hive_connection(config)
    cursor = None
    try:
        cursor = connection.cursor()
        yield cursor
    finally:
        try:
            if cursor is not None:
                close_cursor = getattr(cursor, "close", None)
                if callable(close_cursor):
                    close_cursor()
        finally:
            connection.close()


def _hive_column_names(description: Any) -> list[str]:
    """Strip qualifiers, validate identifiers, and reject collisions."""
    names: list[str] = []
    for column in description or []:
        raw_name = column[0]
        if not isinstance(raw_name, str):
            raise ValueError(f"Invalid Hive result column name: {raw_name!r}")
        name = validate_sql_identifier(raw_name.split(".")[-1])
        if name in names:
            raise ValueError(
                f"duplicate Hive result column after removing qualifier: {name!r}"
            )
        names.append(name)
    return names


def _hive_operation_decimal_qualifiers(cursor: Any) -> dict[int, tuple[int, int]]:
    """Recover DECIMAL qualifiers discarded by PyHive ``cursor.description``.

    PyHive fetches the complete Thrift operation schema but intentionally emits
    ``None`` for DB-API precision and scale. Querying the same operation metadata
    exposes Hive's required ``precision``/``scale`` type qualifiers.
    """
    try:
        from TCLIService import ttypes

        connection = cursor._connection
        operation_handle = cursor._operationHandle
        response = connection.client.GetResultSetMetadata(
            ttypes.TGetResultSetMetadataReq(operation_handle)
        )
        columns = response.schema.columns
    except Exception:
        return {}

    result: dict[int, tuple[int, int]] = {}
    for index, column in enumerate(columns):
        try:
            primitive = column.typeDesc.types[0].primitiveEntry
            raw_qualifiers = primitive.typeQualifiers.qualifiers
        except (AttributeError, IndexError, TypeError):
            continue
        qualifiers: dict[str, int] = {}
        for raw_name, raw_value in raw_qualifiers.items():
            name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            value = getattr(raw_value, "i32Value", raw_value)
            if name in {"precision", "scale"} and value is not None:
                qualifiers[name] = int(value)
        if "precision" in qualifiers and "scale" in qualifiers:
            result[index] = (qualifiers["precision"], qualifiers["scale"])
    return result


def _is_hive_decimal(description: Any) -> bool:
    import decimal

    type_code = description[1] if len(description) > 1 else None
    return (
        type_code is decimal.Decimal
        or type_code == 15
        or "DECIMAL" in str(type_code).upper()
        or "NUMERIC" in str(type_code).upper()
    )


def _hive_arrow_type(
    description: Any,
    decimal_qualifiers: tuple[int, int] | None = None,
) -> Any:
    """Map DB-API/PyHive type metadata to a stable Arrow type."""
    import datetime
    import decimal

    import pyarrow as pa

    type_code = description[1] if len(description) > 1 else None
    python_types = {
        bool: pa.bool_(),
        int: pa.int64(),
        float: pa.float64(),
        str: pa.string(),
        bytes: pa.binary(),
        datetime.date: pa.date32(),
        datetime.datetime: pa.timestamp("us"),
        decimal.Decimal: pa.decimal128(38, 18),
    }
    mapped = python_types.get(type_code) if isinstance(type_code, type) else None

    thrift_types = {
        0: pa.bool_(),
        1: pa.int64(),
        2: pa.int64(),
        3: pa.int64(),
        4: pa.int64(),
        5: pa.float64(),
        6: pa.float64(),
        7: pa.string(),
        8: pa.timestamp("us"),
        9: pa.binary(),
        15: pa.decimal128(38, 18),
        17: pa.date32(),
        18: pa.string(),
        19: pa.string(),
    }
    if mapped is not None:
        pass
    elif isinstance(type_code, int) and type_code in thrift_types:
        mapped = thrift_types[type_code]
    else:
        normalized = str(type_code).upper()
        if "INTERVAL" in normalized:
            mapped = None
        elif "BOOLEAN" in normalized or normalized == "BOOL":
            mapped = pa.bool_()
        elif any(
            token in normalized
            for token in ("TINYINT", "SMALLINT", "BIGINT", "INT_TYPE", "INTEGER")
        ):
            mapped = pa.int64()
        elif any(token in normalized for token in ("FLOAT", "DOUBLE", "REAL")):
            mapped = pa.float64()
        elif "DECIMAL" in normalized or "NUMERIC" in normalized:
            mapped = pa.decimal128(38, 18)
        elif "TIMESTAMP" in normalized:
            mapped = pa.timestamp("us")
        elif "DATE" in normalized:
            mapped = pa.date32()
        elif "BINARY" in normalized or "BLOB" in normalized:
            mapped = pa.binary()
        elif any(token in normalized for token in ("STRING", "VARCHAR", "CHAR")):
            mapped = pa.string()
        else:
            mapped = None

    if mapped is None:
        raise ValueError(f"Unsupported Hive result type metadata: {type_code!r}")
    if pa.types.is_decimal(mapped):
        precision = description[4] if len(description) > 4 else None
        scale = description[5] if len(description) > 5 else None
        if decimal_qualifiers is not None:
            qualifier_precision, qualifier_scale = decimal_qualifiers
            precision = precision if precision is not None else qualifier_precision
            scale = scale if scale is not None else qualifier_scale
        if precision is None or scale is None:
            raise ValueError(
                "Hive DECIMAL metadata must include precision and scale; "
                "PyHive cursor.description omits them and the Thrift operation "
                "schema did not expose reliable qualifiers"
            )
        precision = int(precision)
        scale = int(scale)
        if precision < 1 or precision > 38 or scale < 0 or scale > precision:
            raise ValueError(
                f"Invalid Hive decimal metadata: precision={precision}, scale={scale}"
            )
        return pa.decimal128(precision, scale)
    return mapped


def _hive_schema(cursor: Any) -> Any:
    import pyarrow as pa

    columns = list(cursor.description or [])
    names = _hive_column_names(columns)
    needs_decimal_qualifiers = any(
        _is_hive_decimal(column)
        and (len(column) <= 5 or column[4] is None or column[5] is None)
        for column in columns
    )
    decimal_qualifiers = (
        _hive_operation_decimal_qualifiers(cursor) if needs_decimal_qualifiers else {}
    )
    return pa.schema(
        [
            pa.field(
                name,
                _hive_arrow_type(column, decimal_qualifiers.get(index)),
            )
            for index, (name, column) in enumerate(zip(names, columns, strict=True))
        ]
    )


def _hash_bucket_expression(column: str, shard_count: int) -> str:
    quoted = quote_sql_identifier(column)
    return (
        "pmod(coalesce(hash(coalesce(cast("
        f"{quoted} as string), '__tributo_null__')), 0), {shard_count})"
    )


class HiveDatasource(Datasource):
    """Ray datasource whose tasks stream one HiveServer2 shard at a time."""

    def __init__(
        self,
        config: HiveReadConfig,
        total_rows: int,
        schema: pa.Schema,
    ) -> None:
        self._config = config
        self._total_rows = total_rows
        self._schema = schema

    def estimate_inmemory_data_size(self) -> int | None:
        return self._total_rows * schema_row_width(self._schema)

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: Any | None = None,
    ) -> list[ReadTask]:
        del per_task_row_limit, data_context
        if self._config.shard_mode == "offset":
            return self._offset_tasks(parallelism)
        return self._hash_tasks(parallelism)

    def _metadata(self, rows: int) -> BlockMetadata:
        return BlockMetadata(
            num_rows=rows,
            size_bytes=rows * schema_row_width(self._schema),
            exec_stats=None,
        )

    def _hash_tasks(self, parallelism: int) -> list[ReadTask]:
        task_count = min(
            max(1, parallelism),
            self._config.hash_shards,
            self._total_rows,
        )
        column = self._config.hash_column or self._schema.names[0]
        validate_sql_identifier(column)
        if column not in self._schema.names:
            raise ValueError(
                f"hash_column {column!r} not found in query result columns "
                f"{self._schema.names!r}"
            )
        expression = _hash_bucket_expression(column, task_count)
        estimated_rows = max(1, ceil(self._total_rows / task_count))
        tasks: list[ReadTask] = []
        for bucket in range(task_count):

            def read(bucket: int = bucket) -> Iterator[pa.Table]:
                yield from self._stream_hash_bucket(expression, bucket)

            tasks.append(
                ReadTask(read, self._metadata(estimated_rows), schema=self._schema)
            )
        return tasks

    def _offset_tasks(self, parallelism: int) -> list[ReadTask]:
        task_count = min(max(1, parallelism), self._total_rows)
        rows_per_task = ceil(self._total_rows / task_count)
        tasks: list[ReadTask] = []
        for offset in range(0, self._total_rows, rows_per_task):
            limit = min(rows_per_task, self._total_rows - offset)

            def read(offset: int = offset, limit: int = limit) -> Iterator[pa.Table]:
                yield from self._stream_offset(offset, limit)

            tasks.append(ReadTask(read, self._metadata(limit), schema=self._schema))
        return tasks

    def _stream_hash_bucket(self, expression: str, bucket: int) -> Iterator[pa.Table]:
        wrapped = (
            f"SELECT * FROM ({_clean_query(self._config.sql)}) AS t "
            f"WHERE {expression} = {bucket}"
        )
        yield from self._stream_query(wrapped)

    def _stream_offset(self, offset: int, limit: int) -> Iterator[pa.Table]:
        wrapped = (
            f"SELECT * FROM ({_clean_query(self._config.sql)}) AS t"
            f"{deterministic_order_by(self._schema)}"
            f" LIMIT {offset}, {limit}"
        )
        yield from self._stream_query(wrapped)

    def _stream_query(self, sql: str) -> Iterator[pa.Table]:
        import pyarrow as pa

        with _hive_cursor(self._config) as cursor:
            cursor.execute(sql, parameters=self._config.params or None)
            columns = _hive_column_names(cursor.description)
            while True:
                rows = cursor.fetchmany(self._config.batch_size)
                if not rows:
                    break
                yield pa.Table.from_pylist(
                    [dict(zip(columns, row, strict=True)) for row in rows],
                    schema=self._schema,
                )


class HiveDataConnector:
    """Read-only training connector backed by optional ``pyhive``."""

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        config = HiveReadConfig(**kwargs)
        if not _clean_query(config.sql):
            raise ValueError("missing sql in hive data config")
        total_rows = self._count_rows(config)
        if total_rows <= 0:
            raise EmptyDatasetError("Hive query returned empty result")
        schema = self._probe_schema(config)

        import ray.data

        return cast(
            "ray.data.Dataset",
            ray.data.read_datasource(
                HiveDatasource(config, total_rows, schema),
                parallelism=config.parallelism,
            ),
        )

    def _count_rows(self, config: HiveReadConfig) -> int:
        with _hive_cursor(config) as cursor:
            cursor.execute(
                f"SELECT count(*) FROM ({_clean_query(config.sql)}) AS t",
                parameters=config.params or None,
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def _probe_schema(self, config: HiveReadConfig) -> pa.Schema:
        with _hive_cursor(config) as cursor:
            cursor.execute(
                f"SELECT * FROM ({_clean_query(config.sql)}) AS t LIMIT 0",
                parameters=config.params or None,
            )
            return _hive_schema(cursor)

    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        raise NotImplementedError("Hive writes are outside the training reader API")
