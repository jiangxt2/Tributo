"""Ray Dataset distributed data loading.

Canonical entry-point (Phase 3)::

    from tributo.training.data_loader import load_ray_dataset_from_source
    ds = load_ray_dataset_from_source({"type": "parquet", "path": "s3://..."})

Legacy entry-point (deprecated, kept for old plugins)::

    from tributo.training.data_loader import load_ray_dataset_from_config
    ds = load_ray_dataset_from_config({"type": "s3", "uri": "s3://...", ...})
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from tributo._common import find_project_root
from tributo.data.source_config import (
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    ParquetSourceConfig,
    SourceConfig,
    SqlSourceConfig,
)

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical loader (Phase 3)
# ---------------------------------------------------------------------------


def load_ray_dataset_from_source(
    source: dict[str, Any],
    *,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a Ray Dataset from a canonical ``SourceConfig`` dict.

    Validates *source* against the ``SourceConfig`` discriminated union
    before dispatching to the appropriate loader.  Unknown fields are
    rejected because all four source types inherit ``StrictConfigModel``.

    Args:
        source: Canonical source dict, e.g.
            ``{"type": "parquet", "path": "s3://bucket/data"[, "s3": {...}]}``.
        project_root_path: Project root for resolving relative paths.

    Returns:
        Ray Dataset (lazy — data does not land in driver memory).

    Raises:
        ValidationError: If *source* fails Pydantic validation.
        ValueError: Unsupported source type (should not happen after validation).
    """
    adapter = TypeAdapter(SourceConfig)
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
        "Use load_ray_dataset_from_source() with a canonical source dict.",
        FutureWarning,
        stacklevel=2,
    )
    cfg = LegacyConfigNormalizer.normalize(data_config)
    # Dump back to dict so the canonical loader re-validates via TypeAdapter.
    return load_ray_dataset_from_source(
        cfg.model_dump(mode="python"),
        project_root_path=project_root_path,
    )


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
    import pymysql

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
) -> "pd.DataFrame":  # noqa: F821
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
