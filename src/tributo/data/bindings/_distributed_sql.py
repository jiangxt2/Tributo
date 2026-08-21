"""Small SQL-safety helpers shared by distributed Ray SQL bindings."""

from __future__ import annotations

import re
from typing import Any

_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_sql_identifier(name: str) -> str:
    if not isinstance(name, str) or _SQL_IDENTIFIER.fullmatch(name) is None:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def quote_sql_identifier(name: str) -> str:
    return f"`{validate_sql_identifier(name)}`"


def deterministic_order_by(schema: Any) -> str:
    names = getattr(schema, "names", None)
    if not names:
        raise ValueError("At least one result column is required for sharding")
    return " ORDER BY " + ", ".join(quote_sql_identifier(name) for name in names)


def schema_row_width(schema: Any) -> int:
    if schema is None:
        return 200
    import pyarrow as pa

    total = 0
    for field in schema:
        field_type = field.type
        if pa.types.is_boolean(field_type):
            total += 1
        elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
            total += max(1, field_type.bit_width // 8)
        elif (
            pa.types.is_timestamp(field_type)
            or pa.types.is_date(field_type)
            or pa.types.is_time(field_type)
        ):
            total += 8
        else:
            total += 64
    return max(1, total)
