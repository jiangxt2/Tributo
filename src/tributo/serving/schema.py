"""Request/response data models for ONNX online inference service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class PredictRequest(BaseModel):
    """Inference request body.

    Attributes:
        features: Input feature matrix, one sample per row.
            Example: ``[[0.1, 0.2, ...], [0.3, 0.4, ...]]``
        return_probs: Whether to return probabilities (classification model), default True.
            When False, only returns the label.
    """

    features: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Input feature matrix, one sample per row",
    )
    return_probs: bool = Field(
        default=True,
        description="Whether classification model returns probabilities; False returns only the label",
    )


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
    """

    predictions: list[Any]
    model_path: str
    inference_time_ms: float
