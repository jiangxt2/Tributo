"""Automatic output format detection and distributed writing.

Routes datasets containing vector columns to Lance (distributed
two-phase commit) and scalar datasets to Parquet + ZSTD.

Lance utility functions (``_has_vector_column``, ``_write_lance_distributed``,
etc.) reuse the implementations in ``data/lance.py`` to avoid code duplication.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tributo.data._s3 import to_lance_storage_options
from tributo.data.base import WriteMode
from tributo.data.lance import _has_vector_column, _write_lance_distributed

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)


def write_dataset(
    ds: "ray.data.Dataset",
    output_path: str,
    mode: WriteMode = WriteMode.APPEND,
) -> None:
    """Write a Ray Dataset, automatically choosing the best format.

    - Contains vector columns → **Lance** (distributed two-phase commit)
    - No vector columns → **Parquet + ZSTD**

    Args:
        ds: The dataset to write.
        output_path: Destination URI or path. For Lance, should end with
            ``.lance`` or be a directory URI.
        mode: Lance write mode. Default APPEND (appends if exists, creates if not).

    Raises:
        ImportError: If ``pylance`` is required but not installed.
    """
    schema = ds.schema()
    arrow_schema = schema.base_schema if hasattr(schema, "base_schema") else schema
    if _has_vector_column(arrow_schema):
        logger.info("Vector columns detected → writing Lance: %s", output_path)
        storage_options = to_lance_storage_options(None)
        _write_lance_distributed(ds, output_path, storage_options, mode)
    else:
        logger.info("No vector columns → writing Parquet + ZSTD: %s", output_path)
        ds.write_parquet(output_path, compression="zstd")
