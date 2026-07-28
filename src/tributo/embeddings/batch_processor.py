"""Ray Dataset map_batches operator for distributed text embedding.

Provides a stateful Actor class that loads the ONNX model once and
reuses it across many batches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tributo.embeddings.local_encoder import LocalEncoder
from tributo.embeddings.registry import ModelSpec, get_spec

logger = logging.getLogger(__name__)


class Embedder:
    """Stateful Ray Actor for batch text embedding.

    Each actor instance downloads the model once in ``__init__`` and
    holds it in memory for the lifetime of the actor. This amortizes
    model loading cost across many batches.

    Args:
        model_uri: Path or URI to the exported model directory.
            Supports ``s3://`` or local ``file://`` / plain path.
        text_column: Name of the column containing raw text.
    """

    def __init__(self, model_uri: str, text_column: str) -> None:
        self.text_column = text_column
        local_dir = _resolve_model_path(model_uri)
        spec = _get_spec_from_path(local_dir)
        self.encoder = LocalEncoder(local_dir, spec)
        logger.info(
            "Embedder actor ready: model=%s dim=%d",
            spec.name,
            spec.dim,
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Embed a batch of texts and append an ``embedding`` column.

        Args:
            batch: Dictionary of numpy arrays from Ray Dataset.

        Returns:
            Batch with original columns preserved plus ``embedding``.
        """
        texts = batch[self.text_column].tolist()

        # Sanitize nulls / non-strings
        texts = [str(t) if t is not None else "" for t in texts]

        embeddings = self.encoder.encode(texts)
        batch["embedding"] = embeddings
        return batch


def _resolve_model_path(model_uri: str) -> Path:
    """Resolve a model URI to a local directory path.

    Supports:
    - ``s3://bucket/path/`` → download to local temp
    - ``file:///absolute/path`` → strip prefix
    - ``/absolute/path`` → use directly

    Args:
        model_uri: URI or local path.

    Returns:
        Local directory containing ``model.onnx``.
    """
    if model_uri.startswith("s3://"):
        return _download_from_s3(model_uri)

    if model_uri.startswith("file://"):
        return Path(model_uri[7:])

    local = Path(model_uri)
    if local.exists():
        return local

    raise FileNotFoundError(f"Model path not found: {model_uri}")


def _download_from_s3(s3_uri: str) -> Path:
    """Download model artifacts from S3 to a local temp directory.

    Args:
        s3_uri: S3 directory URI, e.g. ``s3://bucket/models/bge-small-zh/``.

    Returns:
        Local directory path.
    """
    import tempfile
    import uuid

    try:
        import s3fs
    except ImportError as e:
        raise ImportError(
            "s3fs is required for S3 model download. "
            "Install with: uv sync --extra embeddings"
        ) from e

    cache_dir = Path(tempfile.gettempdir()) / "tributo_models" / str(uuid.uuid4())
    cache_dir.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem()
    # s3fs expects bucket/prefix paths without s3://
    s3_path = s3_uri.replace("s3://", "")

    logger.info("Downloading model from %s to %s", s3_uri, cache_dir)
    fs.get(s3_path, str(cache_dir), recursive=True)
    logger.info("Model download complete")
    return cache_dir


def _get_spec_from_path(model_dir: Path) -> ModelSpec:
    """Derive ModelSpec from a local model directory.

    First tries to match by directory name against the registry.
    Raises ``RuntimeError`` if the name is unrecognised.
    """
    name = model_dir.name
    try:
        return get_spec(name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve spec for model directory '{name}'. "
            f"Register the model in embeddings/registry.py or use a registered name."
        ) from exc
