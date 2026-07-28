"""Iceberg data connector — S3 + Iceberg (Parquet + ZSTD) read and write."""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray.data
from pydantic import BaseModel, Field
from pyiceberg.catalog import load_catalog as _pyiceberg_load_catalog
from pyiceberg.exceptions import NoSuchTableError

from tributo.data._s3 import to_iceberg_properties
from tributo.data.base import DataConnector, S3Config, WriteMode
from tributo.data.registry import register_connector
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class IcebergReadConfig(BaseModel):
    """Iceberg read configuration."""

    table_identifier: str = Field(
        min_length=1, description="Table identifier, e.g. 'db.table_name'"
    )
    catalog_name: str = Field(default="default", description="PyIceberg catalog name")
    catalog_properties: dict[str, str] = Field(
        default_factory=dict,
        description="PyIceberg catalog properties (type, uri, warehouse, etc.)",
    )
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")
    snapshot_id: Optional[int] = Field(
        default=None, description="Snapshot ID for time-travel queries"
    )
    row_filter: Optional[str] = Field(
        default=None, description="Row-level filter expression (PyIceberg row_filter)"
    )
    selected_fields: Optional[list[str]] = Field(
        default=None, description="Column projection — read only the specified columns"
    )


class IcebergWriteConfig(BaseModel):
    """Iceberg write configuration."""

    table_identifier: str = Field(
        min_length=1, description="Table identifier, e.g. 'db.table_name'"
    )
    catalog_name: str = Field(default="default", description="PyIceberg catalog name")
    catalog_properties: dict[str, str] = Field(
        default_factory=dict,
        description="PyIceberg catalog properties (type, uri, warehouse, etc.)",
    )
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")
    location: Optional[str] = Field(
        default=None, description="Table data location (required for REST/S3 catalogs)"
    )
    mode: WriteMode = Field(default=WriteMode.OVERWRITE, description="Write mode")


@PublicAPI(stability="alpha")
class IcebergDataConnector(DataConnector):
    """Iceberg data connector.

    Reads and writes Iceberg tables via the PyIceberg catalog API.
    - ``read()``: Gets file list via ``plan_files()``, then Ray Data reads
      the Parquet files in distributed mode — data never touches the driver.
    - ``write()``: Collects the Dataset to the driver and writes through
      PyIceberg (auto-creates table if needed).
    """

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Read an Iceberg table and return a Ray Dataset (distributed read).

        Uses PyIceberg ``plan_files()`` to get the file list (with partition
        pruning), then Ray Data reads the Parquet files in distributed mode
        — driver memory is never a bottleneck.

        Args:
            **kwargs: ``IcebergReadConfig`` fields.

        Returns:
            A lazy ``ray.data.Dataset``.
        """
        cfg = IcebergReadConfig(**kwargs)

        catalog = _load_catalog(cfg.catalog_name, cfg.catalog_properties, cfg.s3)
        table = catalog.load_table(cfg.table_identifier)

        scan_kwargs: dict[str, Any] = {}
        if cfg.row_filter:
            scan_kwargs["row_filter"] = cfg.row_filter
        if cfg.selected_fields:
            scan_kwargs["selected_fields"] = tuple(cfg.selected_fields)
        if cfg.snapshot_id is not None:
            scan_kwargs["snapshot_id"] = cfg.snapshot_id

        scan = table.scan(**scan_kwargs)
        file_paths = [task.file.file_path for task in scan.plan_files()]

        if not file_paths:
            logger.warning("Iceberg table '%s' has no data files", cfg.table_identifier)
            # Construct an empty Arrow table from the catalog schema (metadata
            # only — no I/O) instead of scanning all empty data files.
            schema = table.schema().as_arrow()
            return ray.data.from_arrow(schema.empty_table())

        # Build S3 filesystem if configured
        read_kwargs: dict[str, Any] = {}
        if cfg.selected_fields:
            read_kwargs["columns"] = list(cfg.selected_fields)

        if file_paths[0].startswith("s3://"):
            import pyarrow.fs as pafs

            from tributo.data._s3 import to_pyarrow_s3_kwargs

            read_kwargs["filesystem"] = pafs.S3FileSystem(
                **to_pyarrow_s3_kwargs(cfg.s3)
            )
            # Strip s3:// prefix
            file_paths = [p.removeprefix("s3://") for p in file_paths]

        logger.info(
            "Iceberg table '%s': %d files to read",
            cfg.table_identifier,
            len(file_paths),
        )
        return ray.data.read_parquet(file_paths, **read_kwargs)

    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        """Write a Ray Dataset to an Iceberg table.

        Automatically creates the table (schema inferred from the Dataset)
        if it does not exist.

        Note:
            Collects the entire Dataset into driver memory before writing
            via PyIceberg — unsuitable for datasets larger than available RAM.

        Args:
            dataset: The dataset to write.
            **kwargs: ``IcebergWriteConfig`` fields.
        """
        cfg = IcebergWriteConfig(**kwargs)

        catalog = _load_catalog(cfg.catalog_name, cfg.catalog_properties, cfg.s3)

        arrow_table = dataset.to_arrow()

        try:
            table = catalog.load_table(cfg.table_identifier)
        except NoSuchTableError:
            logger.info("Table '%s' not found, creating...", cfg.table_identifier)
            table = catalog.create_table(
                identifier=cfg.table_identifier,
                schema=arrow_table.schema,
                location=cfg.location,
            )

        if cfg.mode == WriteMode.OVERWRITE:
            table.overwrite(arrow_table)
        else:
            table.append(arrow_table)
        logger.info(
            "Iceberg table '%s' %s: %d rows",
            cfg.table_identifier,
            "overwritten" if cfg.mode == WriteMode.OVERWRITE else "appended",
            arrow_table.num_rows,
        )

    def exists(self, **kwargs: Any) -> bool:
        """Check whether an Iceberg table exists.

        Args:
            **kwargs: ``IcebergReadConfig`` fields.  Extraneous fields
                such as ``snapshot_id`` / ``row_filter`` are ignored.

        Returns:
            ``True`` if the table exists.
        """
        cfg = IcebergReadConfig(**kwargs)
        catalog = _load_catalog(cfg.catalog_name, cfg.catalog_properties, cfg.s3)
        try:
            catalog.load_table(cfg.table_identifier)
            return True
        except NoSuchTableError:
            return False


def _load_catalog(
    catalog_name: str,
    catalog_properties: dict[str, str],
    s3_config: S3Config | None,
) -> Any:
    """Load a PyIceberg catalog, merging S3 auth properties."""
    merged = {**catalog_properties, **to_iceberg_properties(s3_config)}
    return _pyiceberg_load_catalog(catalog_name, **merged)


# ── Built-in registration ──

register_connector("iceberg", IcebergDataConnector)
