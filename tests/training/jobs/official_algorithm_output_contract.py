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
    """Validate one materialized result value against its manifest field."""
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
    if array.dtype.name != expectation.dtype:
        raise AssertionError(
            f"model output {expectation.tensor_name!r} has dtype "
            f"{array.dtype.name!r}, but its manifest declares "
            f"{expectation.dtype!r}"
        )
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise AssertionError(
            f"model output {expectation.tensor_name!r} contains non-finite values"
        )


__all__ = [
    "OutputExpectation",
    "build_output_expectation",
    "validate_output_value",
]
