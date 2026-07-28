"""Tests for embeddings/local_encoder.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo.embeddings.local_encoder import LocalEncoder
from tributo.embeddings.registry import ModelSpec


@pytest.fixture
def mock_model_dir(tmp_path: Path):
    """Create a fake model directory with model.onnx."""
    (tmp_path / "model.onnx").touch()
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    return tmp_path


def test_encode_batch(mock_model_dir: Path):
    spec = ModelSpec(
        name="test",
        hf_model_id="hf/test",
        dim=4,
        pooling="cls",
        normalize=True,
    )

    mock_session = MagicMock()
    # Simulate ONNX output [batch=2, seq_len=3, hidden=4]
    mock_session.run.return_value = [np.random.randn(2, 3, 4).astype(np.float32)]
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]

    mock_tokenizer_cls = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": np.array([[1, 1, 1], [1, 1, 0]]),
    }
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    with patch("tributo.embeddings.local_encoder.ort") as mock_ort:
        mock_ort.InferenceSession = MagicMock(return_value=mock_session)
        with patch(
            "tributo.embeddings.local_encoder.AutoTokenizer",
            mock_tokenizer_cls,
        ):
            encoder = LocalEncoder(mock_model_dir, spec)
            texts = ["hello", "world"]
            result = encoder.encode(texts)

    assert result.shape == (2, 4)
    assert result.dtype == np.float32
    # L2 normalized → each row norm ≈ 1
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)


def test_encode_empty_list(mock_model_dir: Path):
    spec = ModelSpec(name="test", hf_model_id="hf/test", dim=4)

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input_ids")]
    mock_tokenizer_cls = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

    with patch("tributo.embeddings.local_encoder.ort") as mock_ort:
        mock_ort.InferenceSession = MagicMock(return_value=mock_session)
        with patch(
            "tributo.embeddings.local_encoder.AutoTokenizer",
            mock_tokenizer_cls,
        ):
            encoder = LocalEncoder(mock_model_dir, spec)
            result = encoder.encode([])

    assert result.shape == (0, 4)


def test_encode_mean_pooling(mock_model_dir: Path):
    spec = ModelSpec(
        name="test",
        hf_model_id="hf/test",
        dim=2,
        pooling="mean",
        normalize=False,
    )

    mock_session = MagicMock()
    # [batch=1, seq=2, hidden=2]
    mock_session.run.return_value = [
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    ]
    mock_session.get_inputs.return_value = [MagicMock(name="input_ids")]

    mock_tokenizer_cls = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2]]),
        "attention_mask": np.array([[1, 1]]),
    }
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    with patch("tributo.embeddings.local_encoder.ort") as mock_ort:
        mock_ort.InferenceSession = MagicMock(return_value=mock_session)
        with patch(
            "tributo.embeddings.local_encoder.AutoTokenizer",
            mock_tokenizer_cls,
        ):
            encoder = LocalEncoder(mock_model_dir, spec)
            result = encoder.encode(["x"])

    # mean of [1,2] and [3,4] = [2,3]
    expected = np.array([[2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_import_error_when_dependencies_missing():
    """依赖未安装时应抛出 ImportError，而非 ModelExportError。"""
    spec = ModelSpec(name="test", hf_model_id="hf/test", dim=4)

    with patch("tributo.embeddings.local_encoder.ort", None):
        with patch("tributo.embeddings.local_encoder.AutoTokenizer", None):
            with pytest.raises(ImportError, match="onnxruntime and transformers"):
                LocalEncoder(Path("/fake/model"), spec)


def test_model_not_found_raises_file_not_found():
    """模型文件不存在时应抛出 FileNotFoundError，而非 ModelExportError。"""
    spec = ModelSpec(name="test", hf_model_id="hf/test", dim=4)

    mock_ort = MagicMock()
    mock_tokenizer_cls = MagicMock()

    with patch("tributo.embeddings.local_encoder.ort", mock_ort):
        with patch(
            "tributo.embeddings.local_encoder.AutoTokenizer",
            mock_tokenizer_cls,
        ):
            with pytest.raises(FileNotFoundError, match="ONNX model not found"):
                LocalEncoder(Path("/nonexistent/model"), spec)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
