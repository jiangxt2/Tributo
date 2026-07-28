"""Tributo online inference service module.

Based on Ray Serve, provides ONNX model inference, LLM streaming inference,
and gRPC inference capabilities. gRPC features require ``grpcio`` — install
with ``pip install tributo[grpc]``.
"""

from __future__ import annotations

from typing import Any

from tributo.serving.model_deployment import ONNXModel
from tributo.serving.schema import PredictRequest, PredictResponse
from tributo.serving.serve_runner import (
    get_serving_status,
    start_serving,
    stop_serving,
)
from tributo.serving.streaming_deployment import (
    LLMStreamingService,
    StreamingInferenceService,
)
from tributo.serving.streaming_runner import (
    get_streaming_serving_status,
    start_streaming_serving,
    stop_streaming_serving,
)

__all__ = [
    "LLMStreamingService",
    "ONNXModel",
    "PredictRequest",
    "PredictResponse",
    "StreamingInferenceService",
    "gRPCInferenceService",
    "get_grpc_serving_status",
    "get_serving_status",
    "get_streaming_serving_status",
    "start_grpc_serving",
    "start_serving",
    "start_streaming_serving",
    "stop_grpc_serving",
    "stop_serving",
    "stop_streaming_serving",
]

_GRPC_ATTRS: set[str] = {
    "gRPCInferenceService",
    "get_grpc_serving_status",
    "start_grpc_serving",
    "stop_grpc_serving",
}


def __getattr__(name: str) -> Any:  # type: ignore[misc]
    """Lazy-import gRPC symbols so that ``import tributo.serving`` succeeds
    even when ``grpcio`` is not installed."""

    if name not in _GRPC_ATTRS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    try:
        if name == "gRPCInferenceService":
            from tributo.serving.grpc_deployment import (
                gRPCInferenceService as _obj,  # type: ignore[assignment]
            )
        else:
            _mod = __import__("tributo.serving.grpc_runner", fromlist=[name])
            _obj = getattr(_mod, name)  # type: ignore[misc]
    except ImportError as exc:
        msg = f"{name} requires grpcio. Install with: pip install tributo[grpc]"
        raise ImportError(msg) from exc

    # Cache on the module so the lookup runs only once.  # type: ignore[misc]
    globals()[name] = _obj
    return _obj


def __dir__() -> list[str]:
    return sorted(__all__)
