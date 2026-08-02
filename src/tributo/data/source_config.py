"""Typed source configuration with discriminated union and legacy normalizer.

Replaces the hardcoded ``if data_type == "s3": ... elif data_type == "csv": ...``
dispatch in ``training/data_loader.py`` and ``inference/pipeline.py``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from tributo._common.config import StrictConfigModel
from tributo.data.base import S3Config
from tributo.util.annotations import PublicAPI

# ---------------------------------------------------------------------------
# Individual source configs
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class ParquetSourceConfig(StrictConfigModel):
    """Parquet source (local filesystem or S3).

    Attributes:
        type: Discriminator value.
        path: File or directory path (local absolute, or S3 URI).
        columns: Column names to read (``None`` = all columns).
        s3: S3 authentication (``None`` = no S3 / env fallback).
    """

    type: Literal["parquet"] = "parquet"  # noqa: A003
    path: str = Field(min_length=1)
    columns: list[str] | None = None
    s3: S3Config | None = None


@PublicAPI(stability="beta")
class CsvSourceConfig(StrictConfigModel):
    """CSV source (local filesystem or S3).

    Attributes:
        type: Discriminator value.
        path: File path.
        s3: S3 authentication (``None`` = no S3 / env fallback).
        columns: Column names to read (``None`` = all columns).
    """

    type: Literal["csv"] = "csv"  # noqa: A003
    path: str = Field(min_length=1)
    s3: S3Config | None = None
    columns: list[str] | None = None


@PublicAPI(stability="beta")
class SqlSourceConfig(StrictConfigModel):
    """Unified SQL data source for ClickHouse, Doris, PostgreSQL, and MySQL.

    The ``dialect`` field determines the runtime client:

    * ``clickhouse`` / ``doris`` — native client (``clickhouse_connect`` / MySQL protocol).
    * ``postgresql`` / ``mysql`` — ConnectorX (when validated compatible).

    Attributes:
        type: Discriminator value.
        dialect: Database dialect.
        host: Hostname (``None`` = env fallback).
        port: Port (``None`` = dialect default).
        database: Database name (``None`` = env fallback).
        user: Username (``None`` = env fallback).
        password: Password (``None`` = env fallback).
        sql: SQL query string.
        params: Query parameter dict for parameterized SQL (clickhouse_connect
            ``parameters``).  ``None`` = no parameters.
    """

    type: Literal["sql"] = "sql"  # noqa: A003
    dialect: Literal["clickhouse", "doris", "postgresql", "mysql"]
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None
    sql: str = ""
    params: dict[str, Any] | None = None
    partitioning: SqlPartitioning | None = None

    @model_validator(mode="after")
    def _require_sql(self) -> "SqlSourceConfig":
        if not self.sql.strip():
            raise ValueError("sql field must be non-empty for SqlSourceConfig")
        return self

    @model_validator(mode="after")
    def _normalize_empty_params(self) -> "SqlSourceConfig":
        """Normalize empty params dict to None.

        ``params={}`` is semantically identical to ``params=None``
        (no parameterization).  Downstream routing checks ``is not None``
        to decide whether a query is parameterized.
        """
        if self.params == {}:
            object.__setattr__(self, "params", None)
        return self


# ---------------------------------------------------------------------------
# SQL partitioning hint — used by Daft Provider to split large query results
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class SqlPartitioning(BaseModel):
    """Performance hint for SQL result partitioning.

    Only consumed by Daft Provider; Legacy Provider ignores it.
    """

    column: str
    num_partitions: int | None = Field(default=None, ge=1)
    bound_strategy: Literal["min-max", "percentile"] = "min-max"


@PublicAPI(stability="beta")
class IcebergSourceConfig(StrictConfigModel):
    """Iceberg table source.

    Attributes:
        type: Discriminator value.
        catalog: Catalog name (e.g. ``"gravitino"``).
        table: Fully qualified table name.
        catalog_properties: PyIceberg catalog connection properties.
        s3: S3 authentication config for object storage access.
        snapshot_id: Specific snapshot to read (``None`` = current).
    """

    type: Literal["iceberg"] = "iceberg"  # noqa: A003
    catalog: str = Field(min_length=1)
    table: str = Field(min_length=1)
    catalog_properties: dict[str, str] = Field(default_factory=dict)
    s3: dict[str, str] | None = None
    snapshot_id: int | None = None


# ---------------------------------------------------------------------------
# Two-level source input
# ---------------------------------------------------------------------------

# Built-in types keep the Pydantic discriminated union (all discriminators are Literal).
BuiltinSourceConfig = Annotated[
    Union[ParquetSourceConfig, CsvSourceConfig, SqlSourceConfig, IcebergSourceConfig],
    Field(discriminator="type"),
]

# Keep SourceConfig alias for backward compatibility.
SourceConfig = BuiltinSourceConfig


@PublicAPI(stability="beta")
class RawSourceConfig(BaseModel):
    """Passthrough for third-party / unknown source types.

    ``type`` is a free-form string (not Literal), so this class must NOT be
    placed inside a Pydantic ``Field(discriminator="type")`` union.
    """

    type: str  # noqa: A003
    raw: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def _reject_builtin_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("type") in _RESERVED_BUILTIN_TYPES:
            raise ValueError(
                f"Built-in type {data['type']!r} cannot be constructed as RawSourceConfig"
            )
        return data


# Top-level union: no discriminator — validated by Pydantic TypeAdapter.
SourceInput = BuiltinSourceConfig | RawSourceConfig

# Reserved built-in type names — RawSourceConfig with these types is rejected.
_RESERVED_BUILTIN_TYPES: frozenset[str] = frozenset(
    cls.model_fields["type"].default
    for cls in (
        ParquetSourceConfig,
        CsvSourceConfig,
        SqlSourceConfig,
        IcebergSourceConfig,
    )
    if hasattr(cls.model_fields["type"], "default")
)

# ---------------------------------------------------------------------------
# DIALECT DEFAULTS
# ---------------------------------------------------------------------------

_DIALECT_DEFAULTS: dict[str, dict[str, int | str]] = {
    "clickhouse": {"port": 8123, "user": "default"},
    "doris": {"port": 9030, "user": "root"},
    "postgresql": {"port": 5432, "user": "postgres"},
    "mysql": {"port": 3306, "user": "root"},
}

from typing import get_args as _get_args  # noqa: E402

# Extract the Literal values so we can type-check the normalizer parameter.
_SqlDialect = _get_args(SqlSourceConfig.model_fields["dialect"].annotation)

# Legacy ``type`` keys that map to SQL dialects.
_SQL_DIALECT_TYPES: dict[str, str] = {
    "clickhouse": "clickhouse",
    "doris": "doris",
    "postgresql": "postgresql",
    "mysql": "mysql",
}


# ---------------------------------------------------------------------------
# Legacy config normalizer
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class LegacyConfigNormalizer:
    """Normalise legacy ``data_config`` dicts into typed ``SourceConfig`` objects.

    Preserves all historical behaviour documented in the original
    ``data_loader.py``:

    * ``type: csv`` without an explicit ``format`` field → reads Parquet
      (the legacy default was ``format=""``, and the code tested
      ``if fmt == "csv"`` — an empty string never matches).
    * ``type: s3`` routes to ``ParquetSourceConfig`` or ``CsvSourceConfig``
      based on the ``format`` field.
    * ``type: clickhouse`` / ``doris`` / ``mysql`` / ``postgresql`` →
      ``SqlSourceConfig(dialect=...)``.
      Fields set to ``None`` signal that the connection layer should perform
      environment-variable fallback.
    """

    # -- public API ----------------------------------------------------------------

    @staticmethod
    def normalize(
        data_config: dict,
    ) -> SourceInput:
        """Convert a legacy data_config dict to a typed SourceInput.

        Unknown types become ``RawSourceConfig`` for plugin passthrough.
        Malformed built-in types still raise ``ValueError``.
        """
        data_type = data_config.get("type", "csv")

        if data_type == "s3":
            return LegacyConfigNormalizer._normalize_s3(data_config)
        if data_type == "csv":
            return LegacyConfigNormalizer._normalize_csv(data_config)
        if data_type == "parquet":
            return LegacyConfigNormalizer._normalize_parquet(data_config)
        if data_type == "iceberg":
            return LegacyConfigNormalizer._normalize_iceberg(data_config)

        if data_type in _SQL_DIALECT_TYPES:
            dialect: _SqlDialect = _SQL_DIALECT_TYPES[data_type]  # type: ignore[valid-type]
            return LegacyConfigNormalizer._normalize_sql(data_config, dialect)

        # Unknown type → passthrough for plugin providers.
        # Reserved built-in types are rejected above; only genuinely unknown
        # types reach here.
        return RawSourceConfig(type=data_type, raw=data_config)

    # -- private normalizers -------------------------------------------------------

    @staticmethod
    def _normalize_s3(data_config: dict) -> ParquetSourceConfig | CsvSourceConfig:
        fmt = data_config.get("format", "parquet")
        if fmt not in {"parquet", "csv"}:
            raise ValueError(f"unsupported s3 format: {fmt!r}")
        uri = data_config.get("uri", "")
        if not uri:
            raise ValueError("missing s3 uri in data config")
        s3_cfg = data_config.get("s3")
        columns = data_config.get("columns")
        if fmt == "csv":
            return CsvSourceConfig(path=uri, s3=s3_cfg, columns=columns)
        return ParquetSourceConfig(path=uri, s3=s3_cfg, columns=columns)

    @staticmethod
    def _normalize_parquet(data_config: dict) -> ParquetSourceConfig:
        path = data_config.get("path", "")
        s3_cfg = data_config.get("s3")
        columns = data_config.get("columns")
        return ParquetSourceConfig(path=path, s3=s3_cfg, columns=columns)

    @staticmethod
    def _normalize_csv(data_config: dict) -> ParquetSourceConfig | CsvSourceConfig:
        # ⚠️ Historical behaviour: ``type=csv`` only reads actual CSV when
        # ``format`` is explicitly set to ``"csv"``.  Otherwise it reads
        # Parquet (the legacy default ``format=""`` never matched ``"csv"``).
        fmt = data_config.get("format", "")
        path = data_config.get("path", "")
        if fmt == "csv":
            return CsvSourceConfig(path=path)
        return ParquetSourceConfig(path=path)

    @staticmethod
    def _normalize_sql(data_config: dict, dialect: _SqlDialect) -> SqlSourceConfig:  # type: ignore[valid-type]
        """Normalise a legacy SQL-dialect config (clickhouse, doris, etc.)."""
        # Fields set to None signal "apply env fallback at connection time".
        # Prefix mapping: ch_ for clickhouse; port/database/user/password/sql
        # are shared keys.  doris/mysql/postgresql use non-prefixed keys.
        if dialect == "clickhouse":
            return SqlSourceConfig(
                dialect=dialect,
                host=data_config.get("ch_host"),
                port=(
                    int(data_config["ch_port"])
                    if data_config.get("ch_port") is not None
                    else None
                ),
                database=data_config.get("ch_database"),
                user=data_config.get("ch_user"),
                password=data_config.get("ch_password"),
                sql=data_config.get("ch_sql", ""),
                params=data_config.get("ch_sql_params"),
            )
        return SqlSourceConfig(
            dialect=dialect,
            host=data_config.get("host"),
            port=(
                int(data_config["port"])
                if data_config.get("port") is not None
                else None
            ),
            database=data_config.get("database"),
            user=data_config.get("user"),
            password=data_config.get("password"),
            sql=data_config.get("sql", ""),
            params=data_config.get("params"),
        )

    @staticmethod
    def _normalize_iceberg(data_config: dict) -> IcebergSourceConfig:
        return IcebergSourceConfig(
            catalog=data_config.get("catalog", ""),
            table=data_config.get("table", ""),
            catalog_properties=data_config.get("catalog_properties", {}),
            s3=data_config.get("s3"),
            snapshot_id=data_config.get("snapshot_id"),
        )

    # -- env fallback helper -------------------------------------------------------

    @staticmethod
    def resolve_env(
        config: SqlSourceConfig,
    ) -> SqlSourceConfig:
        """Fill ``None`` fields from environment variables.

        Priority: explicit value > env var > dialect default.

        Uses ``is None`` (not ``or``) so that explicit falsy values
        (empty string, port 0) are preserved when explicitly set.

        Environment variables follow the pattern:
        ``TRIBUTO_{DIALECT}_{FIELD}`` (e.g. ``TRIBUTO_CLICKHOUSE_HOST``).
        """
        import os

        dialect = config.dialect
        defaults = _DIALECT_DEFAULTS.get(dialect, {})
        prefix = f"TRIBUTO_{dialect.upper()}"

        # host: explicit None → env var → "localhost"
        host = config.host
        if host is None:
            host = os.getenv(f"{prefix}_HOST", "localhost")

        # port: explicit None → env var → dialect default
        port = config.port
        if port is None:
            port_env = os.getenv(f"{prefix}_PORT", "")
            port = int(port_env) if port_env else int(defaults.get("port", 8123))

        # user: explicit None → env var → dialect default
        user = config.user
        if user is None:
            user = os.getenv(f"{prefix}_USER", str(defaults.get("user", "")))

        # password: explicit None → env var → ""
        password = config.password
        if password is None:
            password = os.getenv(f"{prefix}_PASSWORD", "")

        # database: explicit None → env var → ""
        database = config.database
        if database is None:
            database = os.getenv(f"{prefix}_DB", "")

        return config.model_copy(
            update={
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database,
            }
        )
