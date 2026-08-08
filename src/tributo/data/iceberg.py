"""Iceberg data connector — S3 + Iceberg (Parquet + ZSTD) read and write."""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray.data
from pydantic import BaseModel, Field
from pyiceberg.catalog import load_catalog as _pyiceberg_load_catalog
from pyiceberg.exceptions import NoSuchTableError

from tributo.data._s3 import merge_iceberg_properties
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


@PublicAPI(stability="beta")
class IcebergDataConnector(DataConnector):
    """Iceberg data connector.

    - ``read()``: compatibility adapter over the native Ray Iceberg Binding.
    - ``write()``: Collects the Dataset to the driver and writes through
      PyIceberg (auto-creates table if needed).
    """

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Delegate Iceberg reads to the native Ray Data Binding.

        Args:
            **kwargs: ``IcebergReadConfig`` fields.

        Returns:
            A lazy ``ray.data.Dataset``.
        """
        cfg = IcebergReadConfig(**kwargs)
        from tributo.data._compat_read import open_ray_compat
        from tributo.data.source_config import IcebergSourceConfig

        return open_ray_compat(
            IcebergSourceConfig(
                catalog=cfg.catalog_name,
                table=cfg.table_identifier,
                catalog_properties=cfg.catalog_properties,
                s3=cfg.s3.model_dump(exclude_none=True) if cfg.s3 else None,
                snapshot_id=cfg.snapshot_id,
                row_filter=cfg.row_filter,
                selected_fields=cfg.selected_fields,
            )
        )

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
    merged = merge_iceberg_properties(catalog_properties, source=s3_config)
    return _pyiceberg_load_catalog(catalog_name, **merged)


# ── Built-in registration ──

register_connector("iceberg", IcebergDataConnector)
