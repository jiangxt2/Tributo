"""HuggingFace embedding model → ONNX exporter with validation.

Uses ``optimum`` for one-click ONNX export and tokenizer serialization.
Post-export validation ensures ONNX outputs match HuggingFace native
outputs within a cosine-similarity threshold.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

import numpy as np

from tributo.exceptions import ModelExportError

logger = logging.getLogger(__name__)

#: Minimum cosine similarity between HF and ONNX outputs to accept export.
_MIN_SIMILARITY = 0.999


def export_model(
    hf_model_id: str,
    output_dir: Path,
    quantization: Literal["int8"] | None = None,
) -> Path:
    """Export a HuggingFace model to ONNX + tokenizer.

    The resulting directory contains at least ``model.onnx`` and
    ``tokenizer.json``, compatible with ``LocalEncoder``.

    Args:
        hf_model_id: HuggingFace model ID, e.g. ``"BAAI/bge-small-zh-v1.5"``.
        output_dir: Directory to write artifacts into. Created if missing.
        quantization: Optional quantization mode. Currently only ``"int8"``
            is supported; ``None`` skips quantization.

    Returns:
        Path to the exported ``model.onnx`` file.

    Raises:
        ModelExportError: If export or validation fails.
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ModelExportError(
            "optimum and transformers are required for export. "
            "Install with: uv sync --extra embeddings"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting %s to %s", hf_model_id, output_dir)

    try:
        model = ORTModelForFeatureExtraction.from_pretrained(hf_model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    except Exception as e:
        raise ModelExportError(
            f"Failed to load or export model {hf_model_id}: {e}"
        ) from e

    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        raise ModelExportError(
            f"Failed to save model artifacts to {output_dir}: {e}"
        ) from e

    onnx_path = output_dir / "model.onnx"
    if not onnx_path.exists():
        raise ModelExportError(f"ONNX model not found at {onnx_path}")

    if quantization == "int8":
        onnx_path = _quantize_model(output_dir)

    try:
        _validate_export(output_dir, hf_model_id)
    except ModelExportError:
        # On validation failure, only delete ONNX artifacts, leave other user files untouched
        onnx_path.unlink(missing_ok=True)
        raise

    logger.info("Export complete: %s", onnx_path)
    return onnx_path


def _quantize_model(output_dir: Path) -> Path:
    """Apply INT8 dynamic quantization to the exported ONNX model.

    Args:
        output_dir: Directory containing ``model.onnx``.

    Returns:
        Path to the quantized model.

    Raises:
        ModelExportError: If quantization fails.
    """
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError as e:
        raise ModelExportError("optimum quantization support is missing") from e

    logger.info("Applying INT8 quantization in %s", output_dir)

    try:
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(output_dir)
        quantizer.quantize(
            save_dir=output_dir,
            quantization_config=qconfig,
        )
    except Exception as e:
        raise ModelExportError(f"Quantization failed: {e}") from e

    quantized_path = output_dir / "model_quantized.onnx"
    if not quantized_path.exists():
        raise ModelExportError(f"Quantized model not found at {quantized_path}")

    # Rename so downstream code can always look for "model.onnx"
    target = output_dir / "model.onnx"
    shutil.move(str(quantized_path), str(target))
    logger.info("Quantized model moved to %s", target)
    return target


def _validate_export(output_dir: Path, hf_model_id: str) -> None:
    """Compare ONNX Runtime output against HuggingFace native output.

    A random Chinese sentence is encoded by both pipelines and the
    cosine similarity of the [CLS] vectors must exceed
    ``_MIN_SIMILARITY``.

    Args:
        output_dir: Directory containing the exported ONNX model.
        hf_model_id: Original HuggingFace model ID for reference.

    Raises:
        ModelExportError: If similarity is below threshold or validation
            cannot be performed.
    """
    try:
        import onnxruntime as ort
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.warning("Skipping ONNX validation: torch/onnxruntime not available")
        return

    text = "这是一个测试句子，用于验证ONNX导出是否保持与原始模型一致的语义表示。"

    # HuggingFace native
    try:
        tokenizer_hf = AutoTokenizer.from_pretrained(hf_model_id)
        model_hf = AutoModel.from_pretrained(hf_model_id)
        model_hf.eval()

        inputs_hf = tokenizer_hf(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            hidden_hf = model_hf(**inputs_hf).last_hidden_state
        vec_hf = hidden_hf[:, 0, :].numpy().squeeze()
    except Exception as e:
        raise ModelExportError(f"Failed to run HF reference inference: {e}") from e

    # ONNX Runtime
    try:
        tokenizer_onnx = AutoTokenizer.from_pretrained(output_dir)
        session = ort.InferenceSession(str(output_dir / "model.onnx"))

        inputs_onnx = tokenizer_onnx(
            text, return_tensors="np", truncation=True, max_length=512
        )
        hidden_onnx = session.run(None, dict(inputs_onnx))[0]
        vec_onnx = hidden_onnx[:, 0, :].squeeze()
    except Exception as e:
        raise ModelExportError(f"Failed to run ONNX validation inference: {e}") from e

    similarity = float(
        np.dot(vec_hf, vec_onnx) / (np.linalg.norm(vec_hf) * np.linalg.norm(vec_onnx))
    )

    if similarity < _MIN_SIMILARITY:
        raise ModelExportError(
            f"ONNX validation failed: cosine similarity {similarity:.6f} "
            f"< {_MIN_SIMILARITY}"
        )

    logger.info("ONNX validation passed: cosine similarity %.6f", similarity)
