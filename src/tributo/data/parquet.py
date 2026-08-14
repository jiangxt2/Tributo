"""Parquet data connector — local files and S3."""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray.data
from pydantic import BaseModel, Field

from tributo.data.base import DataConnector, S3Config, WriteMode
from tributo.data.registry import register_connector
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class ParquetReadConfig(BaseModel):
    """Parquet read configuration."""

    path: str = Field(description="Local path or s3:// URI")
    columns: Optional[list[str]] = Field(default=None, description="Columns to read")
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")


class ParquetWriteConfig(BaseModel):
    """Parquet write configuration."""

    path: str = Field(description="Local path or s3:// URI")
    mode: WriteMode = Field(default=WriteMode.OVERWRITE, description="Write mode")
    compression: str = Field(default="zstd", description="Compression codec")
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")


@PublicAPI(stability="alpha")
class ParquetDataConnector(DataConnector):
    """Parquet data connector.

    Supports local files and S3 paths (including glob patterns).
    S3 authentication is resolved from ``S3Config`` or environment
    variables.
    """

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Delegate Parquet reads to the explicit Ray ingestion Binding.

        Args:
            **kwargs: ``ParquetReadConfig`` fields.

        Returns:
            A lazy ``ray.data.Dataset``.
        """
        cfg = ParquetReadConfig(**kwargs)
        if not cfg.path:
            raise ValueError("path must not be empty")
        from tributo.data._compat_read import open_ray_compat
        from tributo.data.source_config import ParquetSourceConfig

        return open_ray_compat(
            ParquetSourceConfig(path=cfg.path, columns=cfg.columns, s3=cfg.s3)
        )

    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        """Write a Parquet file.

        Args:
            dataset: The dataset to write.
            **kwargs: ``ParquetWriteConfig`` fields.

        Raises:
            ValueError: ``APPEND`` mode is not supported for Parquet.
        """
        cfg = ParquetWriteConfig(**kwargs)

        if cfg.mode == WriteMode.APPEND:
            raise ValueError(
                "Parquet does not support APPEND mode. "
                "Use OVERWRITE or write to a new path."
            )

        from tributo.data.writing.compatibility import execute_ray_connector_write

        execute_ray_connector_write(
            dataset=dataset,
            target_kind="parquet",
            target=cfg.path,
            options={"compression": cfg.compression},
            runtime_options={"s3": cfg.s3} if cfg.s3 is not None else {},
            mode=cfg.mode,
        )

        logger.info("Parquet written to %s", cfg.path)


# ── Built-in registration ──

register_connector("parquet", ParquetDataConnector)
