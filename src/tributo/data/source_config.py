"""Typed source configuration with discriminated union and legacy normalizer.

Replaces the hardcoded ``if data_type == "s3": ... elif data_type == "csv": ...``
dispatch in ``training/data_loader.py`` and ``inference/pipeline.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from tributo.util.annotations import PublicAPI

# ---------------------------------------------------------------------------
# Individual source configs
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class ParquetSourceConfig(BaseModel):
    """Parquet source (local filesystem or S3).

    Attributes:
        type: Discriminator value.
        path: File or directory path (local absolute, or S3 URI).
        columns: Column names to read (``None`` = all columns).
        s3: S3 authentication dict (passed through to ``S3Config``).
    """

    type: Literal["parquet"] = "parquet"  # noqa: A003
    path: str = Field(min_length=1)
    columns: list[str] | None = None
    s3: dict[str, str] | None = None


@PublicAPI(stability="beta")
class CsvSourceConfig(BaseModel):
    """CSV source (local filesystem or S3).

    Attributes:
        type: Discriminator value.
        path: File path.
        s3: S3 authentication dict (only relevant for S3 paths).
        columns: Column names to read (``None`` = all columns).
    """

    type: Literal["csv"] = "csv"  # noqa: A003
    path: str = Field(min_length=1)
    s3: dict[str, str] | None = None
    columns: list[str] | None = None


@PublicAPI(stability="beta")
class SqlSourceConfig(BaseModel):
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
    params: dict[str, str] | None = None


@PublicAPI(stability="beta")
class IcebergSourceConfig(BaseModel):
    """Iceberg table source.

    Attributes:
        type: Discriminator value.
        catalog: Catalog name (e.g. ``"gravitino"``).
        table: Fully qualified table name.
    """

    type: Literal["iceberg"] = "iceberg"  # noqa: A003
    catalog: str = Field(min_length=1)
    table: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

SourceConfig = Annotated[
    Union[ParquetSourceConfig, CsvSourceConfig, SqlSourceConfig, IcebergSourceConfig],
    Field(discriminator="type"),
]

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
    ) -> ParquetSourceConfig | CsvSourceConfig | SqlSourceConfig | IcebergSourceConfig:
        """Convert a legacy data_config dict to a typed SourceConfig.

        Raises:
            ValueError: If ``data_config["type"]`` is unrecognised.
        """
        data_type = data_config.get("type", "csv")

        if data_type == "s3":
            return LegacyConfigNormalizer._normalize_s3(data_config)
        if data_type == "csv":
            return LegacyConfigNormalizer._normalize_csv(data_config)
        if data_type == "iceberg":
            return LegacyConfigNormalizer._normalize_iceberg(data_config)

        if data_type in _SQL_DIALECT_TYPES:
            dialect: _SqlDialect = _SQL_DIALECT_TYPES[data_type]  # type: ignore[valid-type]
            return LegacyConfigNormalizer._normalize_sql(data_config, dialect)

        raise ValueError(f"unsupported data type: {data_type!r}")

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
