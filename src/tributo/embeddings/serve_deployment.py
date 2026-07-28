"""Ray Serve Deployment for online text embedding inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tributo.embeddings.local_encoder import LocalEncoder
from tributo.embeddings.registry import ModelSpec, get_spec
from tributo.embeddings.schema import EmbedRequest, EmbedResponse

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)


class TextEmbeddingService:
    """Ray Serve deployment that serves text embedding over HTTP.

    Loads the ONNX model once at startup and reuses it across requests.

    Args:
        model_path: Local directory containing ``model.onnx`` and tokenizer.
        model_name: Short registered model name. Required if path name is
            unrecognised.
    """

    def __init__(self, model_path: str, model_name: str | None = None) -> None:
        spec = self._resolve_spec(model_path, model_name)
        self.encoder = LocalEncoder(Path(model_path), spec)
        self.spec = spec
        logger.info(
            "TextEmbeddingService ready: model=%s dim=%d",
            spec.name,
            spec.dim,
        )

    async def __call__(self, request: Request) -> dict[str, Any]:
        """Handle HTTP embedding request."""
        body = await request.json()
        req = EmbedRequest.model_validate(body)
        return self._embed(req).model_dump()

    def _embed(self, request: EmbedRequest) -> EmbedResponse:
        """Run inference and construct response."""
        start = time.perf_counter()
        embeddings = self.encoder.encode(request.texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "Embedded batch_size=%d dim=%d time=%.2fms",
            len(request.texts),
            self.spec.dim,
            elapsed_ms,
        )

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=self.spec.name,
            dim=self.spec.dim,
        )

    def health(self) -> dict[str, Any]:
        """Health-check endpoint."""
        return {
            "status": "healthy",
            "model": self.spec.name,
            "dim": self.spec.dim,
        }

    @staticmethod
    def _resolve_spec(model_path: str, model_name: str | None) -> ModelSpec:
        from pathlib import Path

        if model_name:
            return get_spec(model_name)
        try:
            return get_spec(Path(model_path).name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve spec for model path '{model_path}'. "
                f"Pass model_name explicitly or use a registered model name."
            ) from exc
