"""Ray Serve ONNX model Deployment implementation."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from tributo.serving.schema import PredictRequest, PredictResponse

if TYPE_CHECKING:
    from starlette.requests import Request

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class ONNXModel:
    """ONNX Runtime inference Deployment.

    Loads the specified ONNX model into memory at startup; all subsequent
    inference requests are completed in memory without accessing disk again.

    Args:
        model_path: ONNX model file path (in-container path, must be mounted or exist).
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for serving. Install with: uv sync"
            ) from e

        logger.info("Loading ONNX model from %s", model_path)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        logger.info(
            "ONNX model loaded. Inputs: %s, Outputs: %s",
            [i.name for i in self.session.get_inputs()],
            [o.name for o in self.session.get_outputs()],
        )

    async def __call__(self, request: Request) -> dict[str, Any]:
        """Handle HTTP inference request."""
        body = await request.json()
        req = PredictRequest.model_validate(body)
        return self._predict(req).model_dump()

    def _predict(self, request: PredictRequest) -> PredictResponse:
        """Execute inference and construct response."""
        start = time.perf_counter()
        x = np.array(request.features, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: x})
        elapsed_ms = (time.perf_counter() - start) * 1000

        # session.run may return list or numpy array, unify to array
        outputs = [np.asarray(o) for o in outputs]

        # Classification models typically have two outputs: [labels, probabilities]
        # Decide which to return based on return_probs
        if len(outputs) >= 2 and request.return_probs:
            predictions = outputs[1].tolist()
        else:
            predictions = outputs[0].tolist()

        logger.debug(
            "Inference batch_size=%d, output_shapes=%s, time=%.2fms",
            len(request.features),
            [o.shape for o in outputs],
            elapsed_ms,
        )

        return PredictResponse(
            predictions=predictions,
            model_path=self.model_path,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def predict_numpy(
        self,
        inputs: dict[str, np.ndarray],
        output_index: int = 0,
    ) -> np.ndarray:
        """Perform inference directly with numpy arrays.

        Used by internal components like IdentityPredictor, bypassing the HTTP request layer.

        Args:
            inputs: Input dictionary, key is input name, value is numpy array.
            output_index: Output index, default 0.

        Returns:
            Inference result as numpy array.
        """
        start = time.perf_counter()
        outputs = self.session.run(None, inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "Inference output_shapes=%s, time=%.2fms",
            [o.shape for o in outputs],
            elapsed_ms,
        )

        return np.asarray(outputs[output_index])

    def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_path": self.model_path,
            "input_name": self.input_name,
        }
