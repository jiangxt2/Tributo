"""Parquet data connector — local files and S3."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pyarrow.fs as pafs
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

        if cfg.path.startswith("s3://"):
            self._write_s3(dataset, cfg.path, cfg.compression, cfg.s3)
        else:
            from pathlib import Path

            Path(cfg.path).parent.mkdir(parents=True, exist_ok=True)
            dataset.write_parquet(cfg.path, compression=cfg.compression)

        logger.info("Parquet written to %s", cfg.path)

    def _write_s3(
        self,
        dataset: ray.data.Dataset,
        s3_path: str,
        compression: str,
        s3_config: S3Config | None,
    ) -> None:
        """Write Parquet to S3."""
        from tributo.data._s3 import to_pyarrow_s3_kwargs

        fs = pafs.S3FileSystem(**to_pyarrow_s3_kwargs(s3_config))
        path = s3_path.removeprefix("s3://")
        dataset.write_parquet(path, filesystem=fs, compression=compression)


# ── Built-in registration ──

register_connector("parquet", ParquetDataConnector)
