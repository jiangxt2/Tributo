"""Typed source configuration with discriminated union and legacy normalizer.

Replaces the hardcoded ``if data_type == "s3": ... elif data_type == "csv": ...``
dispatch in ``training/data_loader.py`` and ``inference/pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from tributo._common.config import StrictConfigModel
from tributo.data.base import S3Config
from tributo.util.annotations import PublicAPI

# ---------------------------------------------------------------------------
# Individual source configs
# ---------------------------------------------------------------------------

_SqlDialect = Literal["clickhouse", "doris", "postgresql", "mysql", "hive"]


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
    s3: S3Config | None = Field(default=None, repr=False)


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
    s3: S3Config | None = Field(default=None, repr=False)
    columns: list[str] | None = None


@PublicAPI(stability="beta")
class SqlSourceConfig(StrictConfigModel):
    """Unified SQL data source for ClickHouse, Doris, PostgreSQL, and MySQL.

    The ``dialect`` field selects a logical Provider. Execution is delegated
    to a Ray Data, Daft, or installed third-party Binding.

    Attributes:
        type: Discriminator value.
        dialect: Database dialect.
        host: Hostname (``None`` = env fallback).
        port: Port (``None`` = dialect default).
        database: Database name (``None`` = env fallback).
        database_schema: Optional schema for structured table reads.
        user: Username (``None`` = env fallback).
        password: Password (``None`` = env fallback).
        sql: Compatibility raw-query input. New ingestion Bindings accept
            structured table reads unless they explicitly advertise a safe
            parameterized-query capability.
        params: Query parameters for the compatibility raw-query input.
            ``None`` means no parameters.
        columns: Column names to project in the SQL reader (``None`` = all
            columns).
    """

    type: Literal["sql"] = "sql"  # noqa: A003
    dialect: _SqlDialect
    host: str | None = None
    port: int | None = None
    http_port: int | None = None
    flight_port: int | None = None
    database: str | None = None
    database_schema: str | None = None
    user: str | None = Field(default=None, repr=False)
    password: str | None = Field(default=None, repr=False)
    sql: str = Field(default="", repr=False)
    table: str | None = None
    protocol: Literal["mysql", "flight"] | None = None
    params: dict[str, Any] | None = Field(default=None, repr=False)
    columns: list[str] | None = None
    partitioning: SqlPartitioning | None = None
    auth: Literal["NONE", "NOSASL", "LDAP", "CUSTOM"] | None = None
    batch_size: int | None = Field(default=None, ge=1)
    shard_mode: Literal["auto", "hash", "offset"] | None = None
    hash_column: str | None = None
    hash_shards: int | None = Field(default=None, ge=1)
    parallelism: int | None = Field(default=None, ge=-1)
    sort_key: str | None = None

    @model_validator(mode="after")
    def _require_one_read_target(self) -> "SqlSourceConfig":
        has_sql = bool(self.sql.strip())
        has_table = bool(self.table and self.table.strip())
        if has_sql == has_table:
            raise ValueError("SqlSourceConfig requires exactly one of sql or table")
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

    @model_validator(mode="after")
    def _validate_protocol(self) -> "SqlSourceConfig":
        if self.dialect == "doris" and self.table and self.protocol is None:
            object.__setattr__(self, "protocol", "mysql")
        if self.dialect != "doris" and self.protocol is not None:
            raise ValueError("protocol is only valid for Doris table reads")
        if self.dialect != "hive" and any(
            value is not None
            for value in (
                self.auth,
                self.batch_size,
                self.shard_mode,
                self.hash_column,
                self.hash_shards,
            )
        ):
            raise ValueError("Hive read options are only valid for Hive sources")
        if self.sort_key is not None and self.dialect != "clickhouse":
            raise ValueError("sort_key is only valid for ClickHouse sources")
        return self


# ---------------------------------------------------------------------------
# SQL partitioning hint — used by Daft Provider to split large query results
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class SqlPartitioning(BaseModel):
    """Performance hint for SQL result partitioning.

    Engine-neutral requirement mapped by the selected Binding.
    """

    mode: Literal["single", "auto", "parallel"] = "parallel"
    column: str | None = None
    num_partitions: int | None = Field(default=None, ge=1)
    bound_strategy: Literal["min-max", "percentile"] = "min-max"

    @model_validator(mode="after")
    def _validate_mode(self) -> "SqlPartitioning":
        if self.mode == "parallel" and not self.column:
            raise ValueError("parallel SQL partitioning requires a column")
        if self.mode != "parallel" and self.column is not None:
            raise ValueError("SQL partitioning column is only valid for parallel mode")
        if self.mode == "single" and self.num_partitions is not None:
            raise ValueError("single SQL partitioning cannot declare num_partitions")
        return self


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
        row_filter: Optional Iceberg row filter expression.
        selected_fields: Optional column projection.
    """

    type: Literal["iceberg"] = "iceberg"  # noqa: A003
    catalog: str = Field(min_length=1)
    table: str = Field(min_length=1)
    catalog_properties: dict[str, str] = Field(default_factory=dict, repr=False)
    s3: dict[str, str] | None = Field(default=None, repr=False)
    snapshot_id: int | None = None
    row_filter: str | None = None
    selected_fields: list[str] | None = None


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
class ProviderSourceConfig(StrictConfigModel):
    """Target ``provider/uri`` canonical source shape.

    Attributes:
        provider: Full provider ID (e.g. ``"tributo.parquet"``). Short
            aliases are resolved by the ProviderRegistry, but the persisted
            config stores the full ID.
        uri: Canonical URI of the bounded data source (``s3://``, local
            path, or a dialect-specific connection reference).
        options: Provider-validated options (format options, table
            references, SQL query digest, etc.). May carry credentials —
            redaction guarantees they never reach ``repr``, logs, errors,
            ``DatasetRef`` or benchmark output.
    """

    provider: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict, repr=False)


# Canonical input union: existing type/path/dialect shapes plus the target
# provider/uri shape. No discriminator — the shape is matched structurally
# (a dict containing "provider" resolves to ProviderSourceConfig; the mixed
# shape carrying both "provider" and "type" fails against both members).
CanonicalSourceInput = BuiltinSourceConfig | ProviderSourceConfig


def _provider_projection_option(source: ProviderSourceConfig) -> str | None:
    """Resolve projection metadata through the Provider SPI."""
    # Local import avoids source_config -> provider -> source_config at module load.
    from tributo.data.provider_registry import resolve_provider

    return resolve_provider(source).projection_option_name


@PublicAPI(stability="beta")
def source_projection(source: CanonicalSourceInput) -> list[str] | None:
    """Return the provider-native projection configured on ``source``.

    The helper deliberately only handles providers with a documented
    projection option.  Unknown providers must define their own projection
    contract instead of receiving a silently ignored option.
    """
    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig, SqlSourceConfig)):
        return list(source.columns) if source.columns else None
    if isinstance(source, IcebergSourceConfig):
        return list(source.selected_fields) if source.selected_fields else None
    option_name = _provider_projection_option(source)
    if option_name is None:
        return None
    value = source.options.get(option_name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(column, str) for column in value
    ):
        raise ValueError(f"source option {option_name!r} must be a list of strings")
    return list(value) if value else None


@PublicAPI(stability="beta")
def apply_source_projection(
    source: CanonicalSourceInput, columns: list[str]
) -> CanonicalSourceInput:
    """Apply ``columns`` using the source's native projection option.

    An existing projection may be narrowed only when every requested column
    is already present.  Requesting a column outside the existing projection
    fails before the provider opens a dataset, preventing a partial read.
    """
    if not columns:
        return source
    if not all(isinstance(column, str) and column for column in columns):
        raise ValueError("source projection columns must be non-empty strings")

    existing = source_projection(source)
    if existing is not None and any(column not in existing for column in columns):
        missing = [column for column in columns if column not in existing]
        raise ValueError(
            "requested projection contains columns outside the configured "
            f"source projection: {missing!r}"
        )

    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig, SqlSourceConfig)):
        return source.model_copy(update={"columns": list(columns)})
    if isinstance(source, IcebergSourceConfig):
        return source.model_copy(update={"selected_fields": list(columns)})

    option_name = _provider_projection_option(source)
    if option_name is None:
        raise ValueError(
            f"provider {source.provider!r} does not declare a projection option"
        )
    options = dict(source.options)
    options[option_name] = list(columns)
    return source.model_copy(update={"options": options})


@PublicAPI(stability="beta")
class RawSourceConfig(BaseModel):
    """Passthrough for third-party / unknown source types.

    ``type`` is a free-form string (not Literal), so this class must NOT be
    placed inside a Pydantic ``Field(discriminator="type")`` union.
    """

    type: str  # noqa: A003
    raw: dict[str, Any] = Field(repr=False)

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
    "hive": {"port": 10000, "user": "default"},
}

# Legacy ``type`` keys that map to SQL dialects.
_SQL_DIALECT_TYPES: dict[str, _SqlDialect] = {
    "clickhouse": "clickhouse",
    "doris": "doris",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "hive": "hive",
}


# ---------------------------------------------------------------------------
# Legacy config normalizer
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class LegacySourceInput:
    """Explicit wrapper for legacy ``data_config`` dicts.

    Carries the historical semantics of the old JSON shapes — ``type=csv``
    without an explicit ``format`` reads Parquet, ``type=s3`` routes by
    ``format``, etc. The ProviderRegistry never guesses these semantics from
    a bare dict; only this typed wrapper enters the legacy resolution path.
    """

    raw: dict[str, Any] = field(repr=False)
    mode: Literal["legacy"] = "legacy"


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
        if "provider" in data_config:
            raise ValueError(
                "canonical provider/uri input must use "
                "ProviderSourceConfig; it cannot enter the legacy normalizer"
            )
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
            dialect = _SQL_DIALECT_TYPES[data_type]
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
    def _normalize_sql(data_config: dict, dialect: _SqlDialect) -> SqlSourceConfig:
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
                sort_key=data_config.get("ch_sort_key"),
                parallelism=data_config.get("ch_parallelism"),
            )
        if dialect == "hive":
            return SqlSourceConfig(
                dialect=dialect,
                host=data_config.get("hive_host"),
                port=(
                    int(data_config["hive_port"])
                    if data_config.get("hive_port") is not None
                    else None
                ),
                database=data_config.get("hive_database"),
                user=data_config.get("hive_user"),
                password=data_config.get("hive_password"),
                auth=data_config.get("hive_auth", data_config.get("auth")),
                sql=data_config.get("hive_sql", ""),
                params=data_config.get("hive_sql_params"),
                batch_size=data_config.get("hive_batch_size"),
                shard_mode=data_config.get("hive_shard_mode"),
                hash_column=data_config.get("hive_hash_column"),
                hash_shards=data_config.get("hive_hash_shards"),
                parallelism=data_config.get("hive_parallelism"),
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
            row_filter=data_config.get("row_filter"),
            selected_fields=data_config.get("selected_fields"),
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
