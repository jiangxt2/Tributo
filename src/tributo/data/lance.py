"""Lance data connector with native-engine write delegation."""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray.data
from pydantic import BaseModel, Field

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


@PublicAPI(stability="beta")
class LanceDataConnector(DataConnector):
    """Lance data connector.

    Writes delegate to the native Ray Lance writer.  Format selection for
    scalar compatibility outputs is kept in the compatibility facade and
    never reimplements Lance fragments or commits.
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

        Auto-detects vector columns: writes Lance when vectors are present,
        otherwise falls back to Parquet + ZSTD.

        Args:
            dataset: The dataset to write.
            **kwargs: ``LanceWriteConfig`` fields.
        """
        cfg = LanceWriteConfig(**kwargs)

        schema = dataset.schema()
        arrow_schema = schema.base_schema if hasattr(schema, "base_schema") else schema
        target_kind = "lance"
        if not _has_vector_column(arrow_schema):
            logger.info(
                "No vector columns detected; using Parquet compatibility target"
            )
            target_kind = "parquet"

        from tributo.data.writing.compatibility import execute_ray_connector_write

        execute_ray_connector_write(
            dataset=dataset,
            target_kind=target_kind,
            target=cfg.path,
            options={"compression": "zstd"} if target_kind == "parquet" else {},
            runtime_options={"s3": cfg.s3} if cfg.s3 is not None else {},
            mode=cfg.mode,
        )


def _has_vector_column(schema: Any) -> bool:
    """Backward-compatible alias for the shared format-selection policy."""
    from tributo.data.writing.policy import has_vector_column

    return has_vector_column(schema)


# ── Built-in registration ──

register_connector("lance", LanceDataConnector)
