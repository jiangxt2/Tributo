"""Lance data connector with distributed two-phase commit writes."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import ray.data
from pydantic import BaseModel, Field

from tributo.data.base import DataConnector, S3Config, WriteMode
from tributo.data.registry import register_connector
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


def _import_lance() -> Any:
    """Lazy-import pylance with a friendly error when not installed."""
    try:
        import lance

        return lance
    except ImportError as e:
        raise ImportError(
            "pylance is required for Lance read/write. "
            "Install with: uv sync --extra data"
        ) from e


class LanceReadConfig(BaseModel):
    """Lance read configuration."""

    path: str = Field(min_length=1, description="Lance dataset path or s3:// URI")
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")


class LanceWriteConfig(BaseModel):
    """Lance write configuration."""

    path: str = Field(min_length=1, description="Lance dataset path or s3:// URI")
    mode: WriteMode = Field(default=WriteMode.OVERWRITE, description="Write mode")
    s3: Optional[S3Config] = Field(default=None, description="S3 connection config")


@PublicAPI(stability="alpha")
class LanceDataConnector(DataConnector):
    """Lance data connector.

    Write path uses distributed two-phase commit:
    Phase 1 — each worker writes its batch as a fragment.
    Phase 2 — the driver collects fragment metadata and commits atomically.
    """

    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Read a Lance dataset and return a Ray Dataset.

        Note:
            Currently loads the entire dataset into driver memory, so it
            is unsuitable for datasets larger than available RAM.

        Args:
            **kwargs: ``LanceReadConfig`` fields.

        Returns:
            A ``ray.data.Dataset``.
        """
        lance = _import_lance()

        cfg = LanceReadConfig(**kwargs)

        from tributo.data._s3 import to_lance_storage_options

        storage_options = to_lance_storage_options(cfg.s3)
        ds = lance.dataset(cfg.path, storage_options=storage_options)
        arrow_table = ds.to_table()
        logger.info(
            "Lance dataset '%s' read: %d rows, %d columns",
            cfg.path,
            arrow_table.num_rows,
            arrow_table.num_columns,
        )
        return ray.data.from_arrow(arrow_table)

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

        if _has_vector_column(arrow_schema):
            logger.info("Vector columns detected → writing Lance: %s", cfg.path)
            from tributo.data._s3 import to_lance_storage_options

            storage_options = to_lance_storage_options(cfg.s3)
            _write_lance_distributed(dataset, cfg.path, storage_options, cfg.mode)
        else:
            logger.info("No vector columns → writing Parquet + ZSTD: %s", cfg.path)
            # Duplicates the S3 write logic from ParquetDataConnector._write_s3
            # (parquet.py).  Keep the two in sync if S3 parameters change.
            write_kwargs: dict[str, Any] = {"compression": "zstd"}
            if cfg.path.startswith("s3://"):
                import pyarrow.fs as pafs

                from tributo.data._s3 import to_pyarrow_s3_kwargs

                write_kwargs["filesystem"] = pafs.S3FileSystem(
                    **to_pyarrow_s3_kwargs(cfg.s3)
                )
                output_path = cfg.path.removeprefix("s3://")
            else:
                output_path = cfg.path
            dataset.write_parquet(output_path, **write_kwargs)


def _has_vector_column(schema: Any) -> bool:
    """Check if the schema contains a floating-point list (vector) column."""
    import pyarrow as pa

    for field in schema:
        t = field.type
        if (
            pa.types.is_fixed_size_list(t)
            or pa.types.is_large_list(t)
            or pa.types.is_list(t)
        ):
            value_type = t.value_type if hasattr(t, "value_type") else t.field(0).type
            if pa.types.is_floating(value_type):
                return True
        if isinstance(t, pa.ExtensionType):
            storage = t.storage_type
            if (
                pa.types.is_fixed_size_list(storage)
                or pa.types.is_large_list(storage)
                or pa.types.is_list(storage)
            ):
                value_type = (
                    storage.value_type
                    if hasattr(storage, "value_type")
                    else storage.field(0).type
                )
                if pa.types.is_floating(value_type):
                    return True
    return False


def _write_lance_distributed(
    ds: ray.data.Dataset,
    uri: str,
    storage_options: dict[str, str] | None,
    mode: WriteMode,
) -> None:
    """Distributed Lance write (two-phase commit)."""
    lance = _import_lance()

    schema = ds.schema()
    arrow_schema = schema.base_schema if hasattr(schema, "base_schema") else schema

    # Phase 1 — distributed fragment writes
    fragment_records = ds.map_batches(
        _WriteLanceFragment,
        batch_format="pyarrow",
        fn_constructor_args=(uri, storage_options),
    ).take_all()

    # Phase 2 — driver-side atomic commit
    all_fragments = [
        lance.FragmentMetadata.from_json(r["json"]) for r in fragment_records
    ]

    try:
        existing = lance.dataset(uri, storage_options=storage_options)
    except Exception:
        existing = None

    if existing is not None and mode == WriteMode.APPEND:
        read_version = existing.version
        op = lance.LanceOperation.Append(all_fragments)
        lance.LanceDataset.commit(
            uri, op, read_version=read_version, storage_options=storage_options
        )
        logger.info("Lance append committed: %s (version %d)", uri, read_version + 1)
    else:
        op = lance.LanceOperation.Overwrite(arrow_schema, all_fragments)
        lance.LanceDataset.commit(
            uri, op, read_version=0, storage_options=storage_options
        )
        logger.info("Lance dataset created: %s", uri)


class _WriteLanceFragment:
    """Ray ``map_batches`` actor — writes a single PyArrow batch as a Lance fragment."""

    def __init__(self, uri: str, storage_options: dict[str, str] | None) -> None:
        self.uri = uri
        self.storage_options = storage_options

    def __call__(self, batch: Any) -> Any:
        lance = _import_lance()
        import pyarrow as pa

        fragments = lance.fragment.write_fragments(
            batch, self.uri, schema=batch.schema, storage_options=self.storage_options
        )
        jsons = [json.dumps(f.to_json()) for f in fragments]
        return pa.table({"json": jsons})


# ── Built-in registration ──

register_connector("lance", LanceDataConnector)
