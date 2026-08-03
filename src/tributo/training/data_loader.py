"""Ray Dataset distributed data loading.

Canonical entry-point (ProviderRegistry path)::

    from tributo.training.data_loader import load_ray_dataset_from_source
    ds = load_ray_dataset_from_source({"type": "parquet", "path": "s3://..."})

    # or the target provider/uri shape:
    ds = load_ray_dataset_from_source(
        {"provider": "tributo.parquet", "uri": "s3://...", "options": {}}
    )

Legacy entry-point (deprecated, kept for old plugins)::

    from tributo.training.data_loader import load_ray_dataset_from_config
    ds = load_ray_dataset_from_config({"type": "s3", "uri": "s3://...", ...})

``TRIBUTO_DATA_BACKEND=legacy`` (read once at import) rolls back to the
pre-registry direct dispatch.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import TypeAdapter

from tributo._common import find_project_root
from tributo.data.provider_registry import resolve_provider
from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    LegacySourceInput,
    ParquetSourceConfig,
    ProviderSourceConfig,
    RawSourceConfig,
    SourceConfig,
    SqlSourceConfig,
)
from tributo.exceptions import JobConfigurationError

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)

# Rollback switch (migration-safety.md): flags are read once at import time —
# changing the env var at runtime has no effect, by design.
DATA_BACKEND = os.getenv("TRIBUTO_DATA_BACKEND", "provider")

# ---------------------------------------------------------------------------
# Canonical loader
# ---------------------------------------------------------------------------


def load_ray_dataset_from_source(
    source: dict[str, Any],
    *,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a Ray Dataset from a canonical source dict (provider path).

    Validates *source* against ``CanonicalSourceInput`` (both the
    ``type/path/dialect`` and the ``provider/uri`` shapes) and routes it
    through the ProviderRegistry: ``resolve → normalize → open →
    to_ray_dataset``.  Unknown fields are rejected (``extra="forbid"``).

    With ``TRIBUTO_DATA_BACKEND=legacy`` the old direct dispatch is used
    instead (rollback switch); a ``provider/uri`` input then fails loudly
    rather than being silently guessed.

    Args:
        source: Canonical source dict, e.g.
            ``{"type": "parquet", "path": "s3://bucket/data"[, "s3": {...}]}``
            or ``{"provider": "tributo.parquet", "uri": ..., "options": {...}}``.
        project_root_path: Project root for resolving relative paths.

    Returns:
        Ray Dataset (lazy — data does not land in driver memory).

    Raises:
        ValidationError: If *source* fails Pydantic validation.
        JobConfigurationError: Unknown provider/type, or provider/uri input
            under the legacy backend.
    """
    if DATA_BACKEND == "legacy":
        if isinstance(source, ProviderSourceConfig) or (
            isinstance(source, dict) and "provider" in source
        ):
            raise JobConfigurationError(
                "provider/uri sources require TRIBUTO_DATA_BACKEND=provider; "
                "the legacy backend cannot route them."
            )
        return _legacy_load_from_canonical_dict(source, project_root_path)

    adapter: TypeAdapter[Any] = TypeAdapter(CanonicalSourceInput)
    cfg = adapter.validate_python(source)
    cfg = _resolve_relative_path(cfg, project_root_path)
    return _load_via_provider(cfg)


# ---------------------------------------------------------------------------
# Legacy loader (deprecated wrapper)
# ---------------------------------------------------------------------------


def load_ray_dataset_from_config(
    data_config: dict,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a Ray Dataset from a legacy flat ``data`` dict.

    .. deprecated::
        Use ``load_ray_dataset_from_source()`` with a canonical
        ``{"type": "...", "path": "..."}`` dict instead.  This wrapper
        normalises the legacy dict via ``LegacyConfigNormalizer`` and
        delegates to the canonical loader.

    Args:
        data_config: Legacy ``data`` dict (``type: s3``, ``type: csv``, etc.).
        project_root_path: Project root for resolving relative paths.

    Returns:
        Ray Dataset (lazy).
    """
    warnings.warn(
        "load_ray_dataset_from_config() is deprecated. "
        "Use load_ray_dataset_from_source() with a canonical source dict. "
        "See https://github.com/jiangxt2/Tributo/blob/master/docs/architecture/"
        "migration-safety.md.",
        FutureWarning,
        stacklevel=2,
    )
    if "provider" in data_config:
        raise JobConfigurationError(
            "provider/uri sources require load_ray_dataset_from_source(); "
            "the legacy config entrypoint cannot route canonical ProviderSourceConfig"
        )
    if DATA_BACKEND == "legacy":
        cfg = LegacyConfigNormalizer.normalize(data_config)
        return _legacy_load_from_canonical_dict(
            cfg.model_dump(mode="python"),
            project_root_path=project_root_path,
        )
    # Historical semantics (type=csv → Parquet default, type=s3
    # routing, ...) live only in the LegacySourceInput branch.
    return _load_via_provider(
        LegacySourceInput(raw=dict(data_config)),
        project_root_path=project_root_path,
    )


# ---------------------------------------------------------------------------
# Canonical provider path
# ---------------------------------------------------------------------------


def _load_via_provider(
    source: CanonicalSourceInput | LegacySourceInput,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Route a typed canonical/legacy source through the ProviderRegistry.

    The registry resolves the provider (canonical route or legacy type
    route); legacy semantics are materialized by ``LegacyConfigNormalizer``
    into a builtin config before ``normalize()``.  Relative file paths are
    resolved against *project_root_path* (same behaviour as the legacy
    dispatch).
    """
    provider = resolve_provider(source)
    if isinstance(source, LegacySourceInput):
        normalized = LegacyConfigNormalizer.normalize(source.raw)
        if isinstance(normalized, RawSourceConfig):
            raise JobConfigurationError(
                f"Unknown legacy source type: {normalized.type!r}"
            )
        resolved_input = _resolve_relative_path(normalized, project_root_path)
        resolved = provider.normalize(resolved_input)
    else:
        resolved = provider.normalize(source)
    handle = provider.open(resolved)
    try:
        return handle.to_ray_dataset()
    finally:
        handle.close()


def _resolve_relative_path(
    source: CanonicalSourceInput, project_root_path: Path | None
) -> CanonicalSourceInput:
    """Resolve relative paths of builtin file sources against the project root.

    Preserves the historical local-path behaviour. Relative local paths in
    both builtin ``path`` and file-provider ``uri`` shapes are resolved
    against *project_root_path*; explicit URI schemes and absolute paths are
    left unchanged.
    """
    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig)):
        parsed = urlsplit(source.path)
        if parsed.scheme:
            return source
        path = Path(source.path)
        if not path.is_absolute():
            root = project_root_path or find_project_root()
            return source.model_copy(update={"path": str(root / path)})
    if isinstance(source, ProviderSourceConfig):
        # Canonical file URIs may be supplied as relative local paths.  Resolve
        # them at the loader boundary so the actual read and ref_id use the
        # same project-root semantics as the typed source shape.
        if source.provider in {"tributo.parquet", "parquet", "tributo.csv", "csv"}:
            parsed = urlsplit(source.uri)
            path = Path(source.uri)
            if not parsed.scheme and not path.is_absolute():
                root = project_root_path or find_project_root()
                return source.model_copy(update={"uri": str(root / path)})
    return source


def _legacy_load_from_canonical_dict(
    source: dict[str, Any],
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Legacy direct dispatch — the ``TRIBUTO_DATA_BACKEND=legacy`` path.

    Frozen copy of the pre-registry loader: validates against the
    ``SourceConfig`` union and dispatches by ``isinstance``.  Kept intact
    for the rollback switch; not evolved (the provider path fixes the
    S3-CSV and Iceberg connector issues).
    """
    adapter: TypeAdapter[Any] = TypeAdapter(SourceConfig)
    cfg = adapter.validate_python(source)

    if isinstance(cfg, SqlSourceConfig):
        return _load_sql_dataset(cfg)
    if isinstance(cfg, (ParquetSourceConfig, CsvSourceConfig)):
        return _load_file_dataset(cfg, project_root_path)
    if isinstance(cfg, IcebergSourceConfig):
        from tributo.data import get_connector

        connector = get_connector("iceberg")
        return connector.read(catalog=cfg.catalog, table=cfg.table)

    raise ValueError(f"unsupported source type: {type(cfg).__name__}")


# ---------------------------------------------------------------------------
# File-based loading (CSV / Parquet — local or S3)
# ---------------------------------------------------------------------------


def _load_file_dataset(
    source: ParquetSourceConfig | CsvSourceConfig,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a file-based dataset, routing S3 paths through connectors."""
    import ray.data

    path = source.path

    if path.startswith("s3://"):
        from tributo.data import get_connector

        fmt = "csv" if isinstance(source, CsvSourceConfig) else "parquet"
        connector = get_connector(fmt)
        kwargs: dict[str, Any] = {"path": path, "s3": source.s3}
        if source.columns:
            kwargs["columns"] = source.columns
        return connector.read(**kwargs)

    # -- local path ---------------------------------------------------------
    root = project_root_path or find_project_root()
    full_path = Path(path)
    if not full_path.is_absolute():
        full_path = root / path
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {full_path}")

    if isinstance(source, CsvSourceConfig):
        return ray.data.read_csv(str(full_path))
    return ray.data.read_parquet(
        str(full_path), columns=source.columns if source.columns else None
    )


# ---------------------------------------------------------------------------
# SQL loading (ClickHouse native client)
# ---------------------------------------------------------------------------


def _load_sql_dataset(source: SqlSourceConfig) -> "ray.data.Dataset":
    """Execute a SQL query and return the result as a Ray Dataset.

    Uses the dialect-appropriate client:

    * ``clickhouse`` — ``clickhouse_connect`` native client.
    * ``doris`` / ``mysql`` — MySQL protocol (via PyMySQL if available).
    * ``postgresql`` — reserved for ConnectorX integration.

    For large result sets, consider pre-exporting to Parquet/S3 instead.
    """
    if source.dialect == "clickhouse":
        return _load_clickhouse(source)
    if source.dialect == "doris":
        return _load_doris_mysql(source)
    if source.dialect in ("postgresql", "mysql"):
        return _load_connectorx(source)

    raise ValueError(f"unsupported SQL dialect: {source.dialect!r}")


def _load_clickhouse(source: SqlSourceConfig) -> "ray.data.Dataset":
    """Execute a ClickHouse query via the native ``clickhouse_connect`` client."""
    import ray.data

    if not source.sql:
        raise ValueError("missing sql in clickhouse source config")

    resolved = LegacyConfigNormalizer.resolve_env(source)

    import clickhouse_connect  # type: ignore[import-untyped]

    client = clickhouse_connect.get_client(
        host=resolved.host,
        port=resolved.port,
        username=resolved.user,
        password=resolved.password or "",
        database=resolved.database,
    )
    try:
        table = client.query_arrow(
            resolved.sql,
            parameters=resolved.params,
        )
    finally:
        client.close()

    if table is None or table.num_rows == 0:
        raise ValueError("ClickHouse query returned empty result")

    num_blocks = max(1, table.num_rows // 10000)
    return ray.data.from_arrow(table, override_num_blocks=num_blocks)


def _load_doris_mysql(source: SqlSourceConfig) -> "ray.data.Dataset":
    """Execute a Doris query via MySQL protocol."""
    import ray.data

    if not source.sql:
        raise ValueError("missing sql in doris source config")

    resolved = LegacyConfigNormalizer.resolve_env(source)

    import pyarrow as pa

    try:
        import pymysql
    except ImportError as exc:
        raise ImportError(
            "The 'mysql' extra is required for Doris sources. "
            "Install it with: pip install 'tributo[mysql]'"
        ) from exc

    conn = pymysql.connect(
        host=resolved.host,
        port=resolved.port or 9030,
        user=resolved.user,
        password=resolved.password,
        database=resolved.database,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(resolved.sql)
            if cursor.description is None:
                raise ValueError(
                    "Doris query returned no result set "
                    "(did you run a non-SELECT statement?)."
                )
            rows = cursor.fetchall()
            if not rows:
                raise ValueError("Doris query returned empty result")
            columns = [desc[0] for desc in cursor.description]
            # Build Arrow table column-wise (single pass over rows).
            table = pa.table(
                {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            )
    finally:
        conn.close()

    return ray.data.from_arrow(table)


def _load_connectorx(source: SqlSourceConfig) -> "ray.data.Dataset":
    """Execute a PostgreSQL/MySQL query via ConnectorX (experimental)."""
    raise NotImplementedError(
        f"ConnectorX path for {source.dialect!r} is not yet implemented. "
        "Use the native client path instead."
    )


# ---------------------------------------------------------------------------
# Legacy helpers (preserved for backward compatibility)
# ---------------------------------------------------------------------------


def _load_clickhouse_dataset(data_config: dict) -> "ray.data.Dataset":
    """Backward-compatible wrapper: directly call the ClickHouse loader from a dict.

    Deprecated: prefer ``load_ray_dataset_from_config`` which normalises via
    ``LegacyConfigNormalizer``.
    """
    source = LegacyConfigNormalizer.normalize(data_config)
    if not isinstance(source, SqlSourceConfig):
        raise ValueError(
            f"Expected clickhouse data config, got {type(source).__name__}"
        )
    return _load_clickhouse(source)


def load_dataframe_from_config(
    data_config: dict,
    project_root_path: Path | None = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Load a DataFrame from a YAML ``data`` section (fully loaded into Driver memory).

    ⚠️ Only suitable for small datasets. Use ``load_ray_dataset_from_config`` for large data.

    Args:
        data_config: The ``data`` dictionary from YAML.
        project_root_path: Project root directory for resolving relative paths.

    Returns:
        pandas DataFrame.
    """
    ds = load_ray_dataset_from_config(data_config, project_root_path)
    return ds.to_pandas()  # type: ignore[no-any-return]
