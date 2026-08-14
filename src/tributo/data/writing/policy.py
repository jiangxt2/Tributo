"""Format-selection policy shared by compatibility and business writers."""

from __future__ import annotations

from typing import Any


def has_vector_column(schema: Any) -> bool:
    """Return whether an Arrow schema contains a floating-point list column."""
    import pyarrow as pa

    for field in schema:
        value_type = _list_value_type(field.type)
        if value_type is not None and pa.types.is_floating(value_type):
            return True
        if isinstance(field.type, pa.ExtensionType):
            value_type = _list_value_type(field.type.storage_type)
            if value_type is not None and pa.types.is_floating(value_type):
                return True
    return False


def _list_value_type(data_type: Any) -> Any | None:
    import pyarrow as pa

    if not (
        pa.types.is_fixed_size_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_list(data_type)
    ):
        return None
    value_type = getattr(data_type, "value_type", None)
    return value_type if value_type is not None else data_type.field(0).type
