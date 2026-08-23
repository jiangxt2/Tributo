"""Bounded inline and Parquet delivery for Arrow vector-search results."""

from __future__ import annotations

import base64
import json
import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Mapping

import pyarrow as pa

from tributo.data.persistence import write_parquet_table
from tributo.vector_index.contracts import (
    ResultDeliveryMode,
    SearchResultOutput,
)
from tributo.vector_index.errors import VectorResultDeliveryError


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VectorResultDeliveryError(
                "inline results cannot contain non-finite floating-point values"
            )
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return "base64:" + base64.b64encode(value).decode("ascii")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise VectorResultDeliveryError(
        f"inline result contains unsupported value type {type(value).__name__}"
    )


def inline_rows(
    table: pa.Table,
    *,
    limit: int,
    max_bytes: int,
) -> tuple[dict[str, Any], ...]:
    """Convert an Arrow result to bounded JSON-safe row dictionaries."""
    if table.num_rows > limit:
        raise VectorResultDeliveryError(
            "search result exceeds inline_max_rows; use materialized delivery"
        )
    if table.nbytes > max_bytes:
        raise VectorResultDeliveryError(
            "search result exceeds inline_max_bytes; use materialized delivery"
        )
    rows = tuple(
        {str(key): _json_value(value) for key, value in row.items()}
        for row in table.to_pylist()
    )
    serialized = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(serialized) > max_bytes:
        raise VectorResultDeliveryError(
            "search result exceeds inline_max_bytes; use materialized delivery"
        )
    return rows


def _is_unsupported_conditional_put(exc: Exception) -> bool:
    try:
        from botocore.exceptions import ParamValidationError
    except ImportError:
        return False
    return isinstance(exc, ParamValidationError) and "IfNoneMatch" in str(exc)


def write_parquet_result(
    table: pa.Table,
    *,
    output: SearchResultOutput,
    storage_profile: str | None,
) -> str:
    """Write one Arrow result table to the exact requested Parquet URI."""
    if output.mode is not ResultDeliveryMode.MATERIALIZED or output.output_uri is None:
        raise VectorResultDeliveryError(
            "Parquet materialization requires materialized delivery and output_uri"
        )
    uri = output.output_uri
    try:
        write_parquet_table(
            table,
            uri,
            storage_profile=storage_profile,
            exclusive=True,
        )
    except FileExistsError:
        raise VectorResultDeliveryError("output Parquet file already exists") from None
    except Exception as exc:
        if _is_unsupported_conditional_put(exc):
            raise VectorResultDeliveryError(
                "installed S3 client does not support conditional PutObject; "
                "install the supported Tributo vector-index dependencies"
            ) from None
        raise VectorResultDeliveryError(
            f"Parquet result delivery failed ({type(exc).__name__})"
        ) from None
    return uri
