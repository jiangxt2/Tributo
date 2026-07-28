"""ONNX Runtime local encoder with pooling and L2 normalization.

Wraps an exported ONNX model and its tokenizer to provide a simple
``encode(texts) -> np.ndarray`` interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tributo.embeddings.registry import ModelSpec

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment,misc]


class LocalEncoder:
    """Encode texts to dense vectors using ONNX Runtime.

    Loads the ONNX model and tokenizer once during construction.
    Thread-safe for inference (read-only after init).

    Args:
        model_dir: Directory containing ``model.onnx`` and tokenizer artifacts.
        spec: Model specification controlling pooling and normalization.
    """

    def __init__(self, model_dir: Path, spec: ModelSpec) -> None:
        self.spec = spec
        model_dir = Path(model_dir)

        if ort is None or AutoTokenizer is None:
            raise ImportError(
                "onnxruntime and transformers are required. "
                "Install with: uv sync --extra embeddings"
            )

        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info("Loading encoder from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(str(onnx_path))

        inputs = [i.name for i in self.session.get_inputs()]
        logger.info(
            "Encoder loaded. inputs=%s outputs=%s dim=%d",
            inputs,
            [o.name for o in self.session.get_outputs()],
            spec.dim,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts to normalized vectors.

        Args:
            texts: List of raw text strings. Empty strings are tolerated.

        Returns:
            Array of shape ``[len(texts), dim]`` and dtype ``float32``.
        """
        if not texts:
            return np.empty((0, self.spec.dim), dtype=np.float32)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="np",
        )

        # ONNX Runtime expects plain dict of numpy arrays
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in [i.name for i in self.session.get_inputs()]:
            onnx_inputs["token_type_ids"] = inputs.get(
                "token_type_ids",
                np.zeros_like(inputs["input_ids"]),
            )

        outputs = self.session.run(None, onnx_inputs)
        hidden = outputs[0]  # [batch, seq_len, hidden_dim]

        if self.spec.pooling == "cls":
            vec = hidden[:, 0, :]
        else:
            # Mean pooling with attention mask
            mask = inputs["attention_mask"][:, :, None]  # [batch, seq_len, 1]
            vec = (hidden * mask).sum(axis=1) / mask.sum(axis=1)

        vec = vec.astype(np.float32)

        if self.spec.normalize:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            # Avoid division by zero
            vec = np.divide(vec, norms, out=np.zeros_like(vec), where=norms != 0)

        return vec  # type: ignore[no-any-return]
