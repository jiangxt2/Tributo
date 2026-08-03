"""Ray Serve ONNX model Deployment implementation.

The primary loading path is a ``bundle_uri`` plus an explicit ``role``,
routed through the shared ``BundleModelLoader``; a raw ``model_path``
remains as a compatibility adapter.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from tributo.serving.schema import PredictRequest, PredictResponse, request_to_inputs

if TYPE_CHECKING:
    from starlette.requests import Request

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class ONNXModel:
    """ONNX Runtime inference Deployment.

    Loads the model into memory at startup; all subsequent inference
    requests are completed in memory without accessing disk again.

    Args:
        model_path: ONNX model file path (legacy compat adapter).
        bundle_uri: Published bundle URI (stable serving entry point).
        role: Artifact role to serve; defaults to ``inference``.
        unsafe: Permit loading bundles without typed signatures or
            flavors that are not safe.
        storage_profile: Storage profile name for S3 bundles.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        bundle_uri: str | None = None,
        role: str = "inference",
        unsafe: bool = False,
        storage_profile: str | None = None,
    ) -> None:
        if (model_path is None) == (bundle_uri is None):
            raise ValueError(
                "exactly one of 'model_path' (legacy) or 'bundle_uri' must be provided"
            )

        self.model_path: str
        self._runtime: Any = None
        self._session: Any = None
        self.input_name = ""

        if bundle_uri is not None:
            self._open_bundle(
                bundle_uri, role=role, unsafe=unsafe, storage_profile=storage_profile
            )
            self.model_path = bundle_uri
            return

        assert model_path is not None  # guarded by the exclusivity check above
        self._open_legacy(model_path)
        self.model_path = model_path

    def _open_bundle(
        self,
        bundle_uri: str,
        *,
        role: str,
        unsafe: bool,
        storage_profile: str | None,
    ) -> None:
        """Load the model through the shared BundleModelLoader."""
        from tributo.exporting.runtime import BundleModelLoader

        loader = BundleModelLoader()
        self._runtime = loader.open(
            bundle_uri, role=role, unsafe=unsafe, storage_profile=storage_profile
        )
        self.input_name = self._runtime.model.input_names[0]
        logger.info(
            "Loaded bundle %r (role=%r). Inputs: %s, Outputs: %s",
            bundle_uri,
            role,
            self._runtime.model.input_names,
            self._runtime.model.output_names,
        )

    def _open_legacy(self, model_path: str) -> None:
        """Legacy compat path: raw ONNX file, no bundle manifest."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for serving. Install with: uv sync"
            ) from e

        logger.info("Loading ONNX model from %s (legacy path)", model_path)
        self._session = ort.InferenceSession(model_path)
        self.input_name = self._session.get_inputs()[0].name
        logger.info(
            "ONNX model loaded. Inputs: %s, Outputs: %s",
            [i.name for i in self._session.get_inputs()],
            [o.name for o in self._session.get_outputs()],
        )

    async def __call__(self, request: Request) -> dict[str, Any]:
        """Handle HTTP inference request."""
        body = await request.json()
        req = PredictRequest.model_validate(body)
        return self._predict(req).model_dump()

    def _predict(self, request: PredictRequest) -> PredictResponse:
        """Execute inference and construct response."""
        start = time.perf_counter()
        outputs = self._run_outputs(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Classification models typically have two outputs: [labels, probabilities]
        # Decide which to return based on return_probs
        if len(outputs) >= 2 and request.return_probs:
            predictions = outputs[1].tolist()
        else:
            predictions = outputs[0].tolist()

        logger.debug(
            "Inference batch_size=%d, output_shapes=%s, time=%.2fms",
            len(request.features or request.inputs or []),
            [o.shape for o in outputs],
            elapsed_ms,
        )

        return PredictResponse(
            predictions=predictions,
            model_path=self.model_path,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def _run_outputs(self, request: PredictRequest) -> list[np.ndarray]:
        """Run the model and return outputs as a list of arrays.

        Both bundle and legacy paths normalise the request through
        ``request_to_inputs`` — the versioned ``inputs`` protocol works
        on raw model paths too (it is never silently dropped).
        """
        inputs = request_to_inputs(request, self.input_name)
        if self._runtime is not None:
            result = self._runtime.predict(inputs)
            return [
                np.asarray(result[name]) for name in self._runtime.model.output_names
            ]
        return [np.asarray(o) for o in self._session.run(None, inputs)]

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
        if self._runtime is not None:
            result = self._runtime.predict(inputs)
            outputs = [
                np.asarray(result[name]) for name in self._runtime.model.output_names
            ]
        else:
            outputs = [np.asarray(o) for o in self._session.run(None, inputs)]
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "Inference output_shapes=%s, time=%.2fms",
            [o.shape for o in outputs],
            elapsed_ms,
        )

        return np.asarray(outputs[output_index])

    def close(self) -> None:
        """Release bundle resources (idempotent).

        Call when the deployment is torn down (e.g. Ray Serve replica
        shutdown, or the embedding actor's lifetime ends).  Prediction
        keeps working after close — the model is in memory; close only
        releases the bundle's temp files.  No-op on the legacy
        ``model_path`` path, which owns no runtime resources.
        """
        if self._runtime is not None:
            self._runtime.close()

    def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_path": self.model_path,
            "input_name": self.input_name,
        }
