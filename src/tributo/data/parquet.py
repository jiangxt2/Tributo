"""Parquet data connector — local files and S3."""

from __future__ import annotations

import fnmatch
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
        """Read Parquet files.

        Args:
            **kwargs: ``ParquetReadConfig`` fields.

        Returns:
            A lazy ``ray.data.Dataset``.
        """
        cfg = ParquetReadConfig(**kwargs)
        if not cfg.path:
            raise ValueError("path must not be empty")
        path = cfg.path

        if path.startswith("s3://"):
            return self._read_s3(path, cfg.columns, cfg.s3)

        # Local file
        read_kwargs: dict[str, Any] = {}
        if cfg.columns:
            read_kwargs["columns"] = cfg.columns
        return ray.data.read_parquet(path, **read_kwargs)

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

    def _read_s3(
        self,
        s3_path: str,
        columns: list[str] | None,
        s3_config: S3Config | None,
    ) -> ray.data.Dataset:
        """Read Parquet from S3 (supports glob patterns)."""
        from tributo.data._s3 import to_pyarrow_s3_kwargs

        fs = pafs.S3FileSystem(**to_pyarrow_s3_kwargs(s3_config))
        path = s3_path.removeprefix("s3://")

        read_kwargs: dict[str, Any] = {"filesystem": fs}
        if columns:
            read_kwargs["columns"] = columns

        if any(c in path for c in ("*", "?", "[")):
            return self._read_s3_glob(fs, path, read_kwargs)

        return ray.data.read_parquet(path, **read_kwargs)

    def _read_s3_glob(
        self,
        fs: Any,
        path: str,
        read_kwargs: dict[str, Any],
    ) -> ray.data.Dataset:
        """S3 glob pattern read."""
        # Find the first glob character and use its parent directory as the
        # FileSelector base.  Unlike the previous implementation we do NOT
        # strip one extra directory level — that caused the selector to scan
        # the entire parent prefix (e.g. "bucket" instead of
        # "bucket/data/2024") for single-level globs.
        glob_idx = path.index("*" if "*" in path else "?" if "?" in path else "[")
        base_dir = path[:glob_idx].rstrip("/")
        if not base_dir:
            base_dir = path.split("/")[0]
        selector = pafs.FileSelector(base_dir, recursive=True)
        file_infos = fs.get_file_info(selector)
        matched = [
            fi.path
            for fi in file_infos
            if fi.is_file and fnmatch.fnmatchcase(fi.path, path)
        ]
        if not matched:
            raise FileNotFoundError(f"No files matched glob pattern: s3://{path}")
        return ray.data.read_parquet(matched, **read_kwargs)

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
