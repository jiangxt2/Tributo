"""gRPC inference service Deployment.

Based on Ray Serve's gRPC support, provides Unary, Server streaming,
and Client streaming RPC modes.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np

from tributo.serving.proto.generated import inference_pb2

if TYPE_CHECKING:
    from ray.serve.grpc_util import RayServegRPCContext, gRPCInputStream

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


def _validate_features(request: inference_pb2.PredictRequest) -> np.ndarray | None:
    """Validate and convert request features.

    Args:
        request: gRPC predict request.

    Returns:
        Converted feature array; returns ``None`` if validation fails.
    """
    if not request.features:
        return None
    return np.array(request.features, dtype=np.float32).reshape(1, -1)


def _set_invalid_argument(context: RayServegRPCContext, message: str) -> None:
    """Set gRPC INVALID_ARGUMENT status code and details."""
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(message)


@PublicAPI(stability="beta")
class gRPCInferenceService:
    """gRPC inference service Deployment.

    Based on Ray Serve's gRPC support, provides Unary, Server streaming,
    and Client streaming RPC modes.

    Note: Do not apply @serve.deployment decorator directly on this class.
    The decoration is handled by deploy_serve_app() to support parameter
    overrides like num_replicas.
    """

    def __init__(self, model_path: str):
        """Initialize gRPC inference service.

        Args:
            model_path: ONNX model file path.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for gRPC serving. Install with: uv sync"
            ) from e

        self._session = ort.InferenceSession(model_path)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "gRPC model loaded from %s, input_name=%s, inputs=%s",
            model_path,
            self._input_name,
            [inp.name for inp in self._session.get_inputs()],
        )

    async def Predict(
        self,
        request: inference_pb2.PredictRequest,
        grpc_context: RayServegRPCContext,
    ) -> inference_pb2.PredictResponse:
        """Unary RPC inference.

        Args:
            request: Predict request.
            grpc_context: gRPC context (Ray Serve convention parameter name).

        Returns:
            Predict result.
        """
        features = _validate_features(request)
        if features is None:
            _set_invalid_argument(grpc_context, "features cannot be empty")
            return inference_pb2.PredictResponse()

        start = time.perf_counter()
        result = self._session.run(None, {self._input_name: features})
        predictions = result[0].flatten().tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "gRPC Predict: features_shape=%s, predictions_len=%d, time=%.2fms",
            features.shape,
            len(predictions),
            elapsed_ms,
        )

        return inference_pb2.PredictResponse(
            predictions=predictions,
            confidence=max(predictions) if predictions else 0.0,
        )

    async def StreamPredict(
        self,
        request: inference_pb2.PredictRequest,
        grpc_context: RayServegRPCContext,
    ):
        """Server streaming inference (batched response).

        Args:
            request: Predict request.
            grpc_context: gRPC context (Ray Serve convention parameter name).

        Yields:
            Batched predict results.
        """
        features = _validate_features(request)
        if features is None:
            _set_invalid_argument(grpc_context, "features cannot be empty")
            return

        result = self._session.run(None, {self._input_name: features})
        predictions = result[0].flatten().tolist()

        # Server streaming: return predictions one by one, suitable for real-time intermediate result display
        for pred in predictions:
            yield inference_pb2.PredictResponse(
                predictions=[pred],
                confidence=pred,
            )

    async def BatchPredict(
        self,
        request_stream: gRPCInputStream,
        grpc_context: RayServegRPCContext,
    ) -> inference_pb2.PredictResponse:
        """Client streaming batch inference.

        Args:
            request_stream: Request stream.
            grpc_context: gRPC context (Ray Serve convention parameter name).

        Returns:
            Merged predict results.
        """
        features_list = []
        async for request in request_stream:
            features = _validate_features(request)
            if features is None:
                _set_invalid_argument(grpc_context, "features cannot be empty")
                return inference_pb2.PredictResponse()
            features_list.append(features)

        if not features_list:
            _set_invalid_argument(grpc_context, "request stream cannot be empty")
            return inference_pb2.PredictResponse()

        batch = np.concatenate(features_list, axis=0)
        result = self._session.run(None, {self._input_name: batch})
        all_predictions = result[0].flatten().tolist()

        return inference_pb2.PredictResponse(
            predictions=all_predictions,
            confidence=max(all_predictions) if all_predictions else 0.0,
        )

    async def health(self) -> dict[str, Any]:
        """Health check.

        Returns:
            Health status information.
        """
        return {
            "status": "healthy",
            "model_loaded": self._session is not None,
            "input_names": [inp.name for inp in self._session.get_inputs()],
        }
