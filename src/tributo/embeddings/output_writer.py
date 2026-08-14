"""Automatic output format selection through the native write Gateway.

Routes datasets containing vector columns to the native Ray Lance writer and
scalar datasets to the native Ray Parquet writer with ZSTD compression.  The
format-selection policy is business-specific, while execution is shared with
all other bounded data writers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tributo.data.base import WriteMode
from tributo.data.contracts.handles import RayDataHandle
from tributo.data.writing.builtins import default_write_gateway
from tributo.data.writing.contracts import WriteRequest
from tributo.data.writing.policy import has_vector_column

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)


def write_dataset(
    ds: "ray.data.Dataset",
    output_path: str,
    mode: WriteMode = WriteMode.APPEND,
) -> None:
    """Write a Ray Dataset, automatically choosing the best format.

    - Contains vector columns → native **Lance** writer
    - No vector columns → **Parquet + ZSTD**

    Args:
        ds: The dataset to write.
        output_path: Destination URI or path. For Lance, should end with
            ``.lance`` or be a directory URI.
        mode: Lance write mode. Default APPEND.

    Raises:
        WriteCapabilityError: If the selected native writer is unavailable or
            does not support the requested mode.
    """
    schema = ds.schema()
    arrow_schema = schema.base_schema if hasattr(schema, "base_schema") else schema
    if has_vector_column(arrow_schema):
        logger.info("Vector columns detected → writing Lance: %s", output_path)
        target_kind = "lance"
    else:
        logger.info("No vector columns → writing Parquet + ZSTD: %s", output_path)
        target_kind = "parquet"

    default_write_gateway().execute(
        WriteRequest(
            engine="ray",
            target_kind=target_kind,
            target=output_path,
            mode=mode,
            options={"compression": "zstd"} if target_kind == "parquet" else {},
        ),
        RayDataHandle(ds),
    )


_has_vector_column = has_vector_column
