"""Request/response data models for ONNX online inference service.

E3 introduces the versioned input protocol ``name/shape/datatype/data``
so that HTTP, gRPC, and Batch inference share one feature semantics.
The legacy flat ``features`` matrix remains as a compat adapter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tributo.util.annotations import PublicAPI

#: Canonical dtype names → numpy dtype mapping for versioned inputs.
_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "bool": np.dtype(np.bool_),
    "float16": np.dtype(np.float16),
    "uint8": np.dtype(np.uint8),
    "int8": np.dtype(np.int8),
}
_INTEGER_DATATYPES = frozenset({"int8", "int32", "int64", "uint8"})


@PublicAPI(stability="beta")
class PredictInput(BaseModel):
    """A single versioned input tensor: name/shape/datatype/data.

    Attributes:
        name: Input name, matching the model signature field.
        shape: Shape; dynamic dimensions are omitted or 0.
        datatype: Canonical dtype name (``float32``, ``int64``, ...).
        data: Flat numeric payload. Integer datatypes require JSON integer
            values; floating-point carriers are rejected to prevent silent
            truncation and loss above the JSON exact-integer range.
    """

    name: str = Field(..., min_length=1)
    shape: list[int] = Field(default_factory=list)
    datatype: str = Field(default="float32", min_length=1)
    data: list[Any] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_shape(self) -> "PredictInput":
        if any(dim < 0 for dim in self.shape):
            raise ValueError(
                f"shape dimensions must be non-negative, got {self.shape!r}"
            )
        return self

    def to_numpy(self, *, reject_integer_floats: bool = False) -> np.ndarray:
        """Convert this input to numpy, optionally rejecting integer floats.

        The HTTP JSON adapter sets ``reject_integer_floats=True`` because a
        JSON floating-point carrier can already have lost integer precision.
        The gRPC adapter leaves it disabled for legacy int32 data carried by
        protobuf ``double`` values; int64 gRPC inputs use ``int64_data``.
        """
        if self.datatype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported datatype {self.datatype!r}. "
                f"Supported: {sorted(_DTYPE_MAP)}"
            )
        if (
            reject_integer_floats
            and self.datatype in _INTEGER_DATATYPES
            and any(isinstance(value, (float, np.floating)) for value in self.data)
        ):
            raise ValueError(
                f"datatype {self.datatype!r} requires integer JSON values; "
                "floating-point values are rejected to prevent precision loss"
            )
        arr = np.asarray(self.data, dtype=_DTYPE_MAP[self.datatype])
        if 0 in self.shape:
            # 0 marks a dynamic dimension — infer its extent from the
            # payload size, then reshape with concrete values.  More than
            # one dynamic dimension has no unique interpretation, so it
            # fails fast instead of guessing.
            zero_count = sum(1 for dim in self.shape if dim == 0)
            if zero_count > 1:
                raise ValueError(
                    f"shape {self.shape!r} has {zero_count} dynamic "
                    "dimensions; at most one (the batch axis) is supported"
                )
            fixed_count = math.prod(dim for dim in self.shape if dim != 0)
            if len(arr) % fixed_count != 0:
                raise ValueError(
                    f"cannot infer the dynamic dimension: {len(arr)} elements "
                    f"are not divisible by the fixed dims (product {fixed_count}) "
                    f"of shape {self.shape!r}"
                )
            batch = len(arr) // fixed_count
            target = [batch if dim == 0 else dim for dim in self.shape]
            arr = arr.reshape(target)
        elif self.shape:
            arr = arr.reshape(self.shape)
        return arr


@PublicAPI(stability="beta")
class PredictRequest(BaseModel):
    """Inference request body.

    Either the versioned ``inputs`` list or the legacy flat ``features``
    matrix must be provided — never both.

    Attributes:
        schema_version: Input protocol version (1 = name/shape/datatype/data).
        inputs: Versioned input tensors (E3 protocol).
        features: Legacy input feature matrix, one sample per row.
        return_probs: Whether to return probabilities (classification model),
            default True.  When False, only the label is returned.
    """

    schema_version: int = Field(default=1, ge=1, le=1)
    inputs: list[PredictInput] | None = None
    features: list[list[float]] | None = None
    return_probs: bool = Field(
        default=True,
        description="Whether classification model returns probabilities; False returns only the label",
    )

    @model_validator(mode="after")
    def _check_input_mode(self) -> "PredictRequest":
        if self.inputs is not None and self.features is not None:
            raise ValueError(
                "inputs (versioned) and features (legacy) are mutually "
                "exclusive — provide exactly one"
            )
        if self.inputs is None and self.features is None:
            raise ValueError(
                "either inputs (versioned) or features (legacy) must be provided"
            )
        if self.features is not None and not self.features:
            raise ValueError("features must not be empty")
        if self.inputs is not None and not self.inputs:
            raise ValueError(
                "inputs must not be empty — provide at least one named tensor"
            )
        return self


@PublicAPI(stability="beta")
class PredictResponse(BaseModel):
    """Inference response body.

    Attributes:
        predictions: Prediction results.
            Classification + return_probs=True: list[list[float]] (per-class probabilities);
            Classification + return_probs=False: list[int] (predicted labels);
            Regression: list[float].
        model_path: ONNX model path used for this inference.
        inference_time_ms: Inference time in milliseconds.
        request_id: Request correlation identifier.
        trace_id: W3C trace identifier.
        traceparent: W3C trace context returned for downstream propagation.
        bundle_id: Immutable bundle identifier, when serving a bundle.
        model_version: E3 model version; manifest-v1 uses bundle_id.
    """

    predictions: list[Any]
    model_path: str
    inference_time_ms: float
    request_id: str | None = None
    trace_id: str | None = None
    traceparent: str | None = None
    bundle_id: str | None = None
    model_version: str | None = None


def request_to_inputs(
    request: PredictRequest, input_name: str
) -> dict[str, np.ndarray]:
    """Convert a request to named numpy inputs for the given input.

    Versioned inputs are used as-is; the legacy flat ``features`` matrix
    is mapped onto *input_name* (the model's first input).
    """
    if request.inputs is not None:
        result: dict[str, np.ndarray] = {}
        for inp in request.inputs:
            if inp.name in result:
                raise ValueError(
                    f"Duplicate input name {inp.name!r} in versioned inputs"
                )
            result[inp.name] = inp.to_numpy(reject_integer_floats=True)
        return result
    features = request.features
    assert features is not None  # guarded by the model validator
    return {input_name: np.asarray(features, dtype=np.float32)}
