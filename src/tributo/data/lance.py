"""Lance data connector backed by the shared distributed Lance writer."""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray.data
from pydantic import BaseModel, Field, model_validator

from tributo._common.lance_write import write_lance_dataset
from tributo.data.base import DataConnector, S3Config, WriteMode
from tributo.data.registry import register_connector
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class LanceReadConfig(BaseModel):
    """Lance read configuration."""

    path: str = Field(min_length=1, description="Lance dataset path or s3:// URI")
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")


class LanceWriteConfig(BaseModel):
    """Lance write configuration."""

    path: str = Field(min_length=1, description="Lance dataset path or s3:// URI")
    mode: WriteMode = Field(default=WriteMode.OVERWRITE, description="Write mode")
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")
    min_rows_per_file: int = Field(default=1024 * 1024, ge=1)
    max_rows_per_file: int = Field(default=64 * 1024 * 1024, ge=1)
    data_storage_version: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_file_bounds(self) -> "LanceWriteConfig":
        if self.max_rows_per_file < self.min_rows_per_file:
            raise ValueError("max_rows_per_file must be >= min_rows_per_file")
        return self


@PublicAPI(stability="beta")
class LanceDataConnector(DataConnector):
    """Lance data connector.

    The compatibility write path retains the historical connector shape while
    sharing Tributo's distributed Ray Data/Lance transaction writer and strict
    save-mode semantics with inference.
    """

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Delegate Lance reads to the native Ray Data Binding.

        Args:
            **kwargs: ``LanceReadConfig`` fields.

        Returns:
            A ``ray.data.Dataset``.
        """
        cfg = LanceReadConfig(**kwargs)
        from tributo.data._compat_read import open_ray_compat
        from tributo.data.source_config import ProviderSourceConfig

        options: dict[str, Any] = {}
        if cfg.s3 is not None:
            options["s3"] = cfg.s3.model_dump(exclude_none=True)
        return open_ray_compat(
            ProviderSourceConfig(
                provider="tributo.lance",
                uri=cfg.path,
                options=options,
            )
        )

    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        """Write a Lance dataset.

        Always write Lance.  Output format selection belongs to the caller;
        this compatibility connector no longer guesses based on a column type.

        Args:
            dataset: The dataset to write.
            **kwargs: ``LanceWriteConfig`` fields.
        """
        cfg = LanceWriteConfig(**kwargs)

        logger.info("Writing Lance dataset: %s", cfg.path)
        from tributo.data._s3 import to_lance_storage_options

        schema = dataset.schema()
        arrow_schema = schema.base_schema if hasattr(schema, "base_schema") else schema
        write_lance_dataset(
            dataset,
            uri=cfg.path,
            schema=arrow_schema,
            mode=cfg.mode.value,
            min_rows_per_file=cfg.min_rows_per_file,
            max_rows_per_file=cfg.max_rows_per_file,
            data_storage_version=cfg.data_storage_version,
            storage_options=to_lance_storage_options(cfg.s3),
        )


# ── Built-in registration ──

register_connector("lance", LanceDataConnector)
