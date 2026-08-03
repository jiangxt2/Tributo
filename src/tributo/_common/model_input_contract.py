"""Shared model input contract validation.

The serving protocols and the bundle runtime must agree on the same
name/dtype/shape contract.  Keeping this small validator outside either
transport prevents HTTP, gRPC, and batch inference from drifting apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_ONNX_DTYPE_TO_CANONICAL: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(bool)": "bool",
    "tensor(float16)": "float16",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
}


def canonical_onnx_dtype(value: Any) -> str | None:
    """Return a canonical dtype for an ONNX Runtime type string.

    Unknown or malformed values return ``None`` so compatibility adapters
    can still operate when a test double does not expose ONNX metadata.
    """
    if not isinstance(value, str):
        return None
    return _ONNX_DTYPE_TO_CANONICAL.get(value, value if value else None)


def normalize_model_shape(value: Any) -> tuple[int | None, ...] | None:
    """Normalize an ONNX shape, treating symbolic dimensions as dynamic."""
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(dim if isinstance(dim, int) else None for dim in value)


def validate_named_inputs(
    inputs: Mapping[str, np.ndarray],
    *,
    expected_names: Sequence[str],
    expected_dtypes: Sequence[str | None] | None = None,
    expected_shapes: Sequence[Sequence[int | None] | None] | None = None,
) -> None:
    """Validate named arrays against a model's input signature.

    Dynamic dimensions are represented by ``None`` and accept any extent.
    Names, dtypes, rank, and fixed dimensions are strict.  The function
    raises ``ValueError`` because all violations are client input errors at
    the serving boundary.
    """
    expected_names_tuple = tuple(expected_names)
    actual_names = tuple(inputs)
    if set(actual_names) != set(expected_names_tuple):
        raise ValueError(
            f"Input names {sorted(actual_names)} do not match the model's "
            f"inputs {sorted(expected_names_tuple)}"
        )

    if expected_dtypes is not None and len(expected_dtypes) != len(
        expected_names_tuple
    ):
        raise ValueError(
            "Model input signature is invalid: dtype cardinality does not "
            "match input names"
        )
    if expected_shapes is not None and len(expected_shapes) != len(
        expected_names_tuple
    ):
        raise ValueError(
            "Model input signature is invalid: shape cardinality does not "
            "match input names"
        )

    for index, name in enumerate(expected_names_tuple):
        array = inputs[name]
        if expected_dtypes is not None:
            expected_dtype = expected_dtypes[index]
            if expected_dtype is not None:
                actual_dtype = np.dtype(array.dtype).name
                if actual_dtype != expected_dtype:
                    raise ValueError(
                        f"Input {name!r} has dtype {actual_dtype!r}, but the "
                        f"model expects {expected_dtype!r}"
                    )

        if expected_shapes is None:
            continue
        expected_shape = expected_shapes[index]
        if expected_shape is None:
            continue
        actual_shape = tuple(int(dim) for dim in array.shape)
        expected_shape_tuple = tuple(expected_shape)
        if len(actual_shape) != len(expected_shape_tuple):
            raise ValueError(
                f"Input {name!r} has rank {len(actual_shape)}, but the model "
                f"expects rank {len(expected_shape_tuple)}"
            )
        for axis, (actual_dim, expected_dim) in enumerate(
            zip(actual_shape, expected_shape_tuple)
        ):
            if expected_dim is not None and actual_dim != expected_dim:
                raise ValueError(
                    f"Input {name!r} has shape {actual_shape}, but the model "
                    f"expects fixed dimension {expected_dim} at axis {axis}"
                )


def is_onnx_invalid_argument(exc: BaseException) -> bool:
    """Return whether *exc* is ONNX Runtime's client-input exception."""
    return (
        exc.__class__.__name__ == "InvalidArgument"
        and exc.__class__.__module__.startswith("onnxruntime")
    )
