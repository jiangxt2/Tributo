"""Pure output-contract checks shared by the official algorithm Gate and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutputExpectation:
    """Expected materialized value for one manifest output tensor."""

    tensor_name: str
    column: str
    dtype: str
    trailing_shape: tuple[int | str, ...]


def build_output_expectation(
    *,
    tensor_name: str,
    column: str,
    dtype: str,
    shape: tuple[int | str, ...],
) -> OutputExpectation:
    """Build a row-level expectation from a batch-oriented manifest field."""
    if not shape or shape[0] != "batch":
        raise AssertionError(
            f"model output {tensor_name!r} must declare dynamic batch first"
        )
    try:
        canonical_dtype = np.dtype(dtype).name
    except TypeError as exc:
        raise AssertionError(
            f"model output {tensor_name!r} has unsupported dtype {dtype!r}"
        ) from exc
    return OutputExpectation(
        tensor_name=tensor_name,
        column=column,
        dtype=canonical_dtype,
        trailing_shape=shape[1:],
    )


def validate_output_value(expectation: OutputExpectation, value: object) -> None:
    """Validate one materialized result value's shape and finite-value contract."""
    array = np.asarray(value)
    actual_shape = tuple(int(dimension) for dimension in array.shape)
    expected_shape = expectation.trailing_shape
    if len(actual_shape) != len(expected_shape):
        raise AssertionError(
            f"model output {expectation.tensor_name!r} has row shape {actual_shape}, "
            f"but its manifest declares trailing shape {expected_shape}"
        )
    for axis, (actual, expected) in enumerate(
        zip(actual_shape, expected_shape, strict=True)
    ):
        if isinstance(expected, int) and actual != expected:
            raise AssertionError(
                f"model output {expectation.tensor_name!r} has row shape "
                f"{actual_shape}, but manifest axis {axis + 1} requires {expected}"
            )
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise AssertionError(
            f"model output {expectation.tensor_name!r} contains non-finite values"
        )


def validate_output_dtype(
    expectation: OutputExpectation, persisted_type: object
) -> None:
    """Validate the persisted Arrow element dtype against the manifest field."""
    import pyarrow as pa

    if not isinstance(persisted_type, pa.DataType):
        raise AssertionError(
            f"model output {expectation.tensor_name!r} has unsupported persisted "
            f"type {persisted_type!r}"
        )
    element_type = persisted_type
    value_type = getattr(element_type, "value_type", None)
    if isinstance(value_type, pa.DataType):
        element_type = value_type
    elif isinstance(element_type, pa.ExtensionType):
        element_type = element_type.storage_type
    while (
        pa.types.is_list(element_type)
        or pa.types.is_large_list(element_type)
        or pa.types.is_fixed_size_list(element_type)
    ):
        element_type = element_type.value_type
    try:
        actual_dtype = np.dtype(element_type.to_pandas_dtype()).name
    except (NotImplementedError, TypeError) as exc:
        raise AssertionError(
            f"model output {expectation.tensor_name!r} has unsupported persisted "
            f"type {persisted_type!r}"
        ) from exc
    if actual_dtype != expectation.dtype:
        raise AssertionError(
            f"model output {expectation.tensor_name!r} has persisted dtype "
            f"{actual_dtype!r}, but its manifest declares {expectation.dtype!r}"
        )


__all__ = [
    "OutputExpectation",
    "build_output_expectation",
    "validate_output_dtype",
    "validate_output_value",
]
