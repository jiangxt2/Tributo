"""Tests for embeddings/batch_processor.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo.embeddings.batch_processor import (
    Embedder,
    _get_spec_from_path,
    _resolve_model_path,
)


def test_embedder_call_appends_embedding():
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )
    mock_encoder.spec.dim = 3

    with patch(
        "tributo.embeddings.batch_processor.LocalEncoder",
        return_value=mock_encoder,
    ):
        with patch(
            "tributo.embeddings.batch_processor._resolve_model_path",
            return_value=Path("/fake/model"),
        ):
            with patch(
                "tributo.embeddings.batch_processor._get_spec_from_path",
                return_value=MagicMock(name="test", dim=3),
            ):
                embedder = Embedder("/fake/model", "content")
                batch = {
                    "content": np.array(["text one", "text two"]),
                }
                result = embedder(batch)

    assert "embedding" in result
    assert result["embedding"].shape == (2, 3)
    assert "content" in result
    np.testing.assert_array_equal(result["content"], np.array(["text one", "text two"]))


def test_embedder_call_sanitizes_nulls():
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.array([[0.1]], dtype=np.float32)
    mock_encoder.spec.dim = 1

    with patch(
        "tributo.embeddings.batch_processor.LocalEncoder",
        return_value=mock_encoder,
    ):
        with patch(
            "tributo.embeddings.batch_processor._resolve_model_path",
            return_value=Path("/fake/model"),
        ):
            with patch(
                "tributo.embeddings.batch_processor._get_spec_from_path",
                return_value=MagicMock(name="test", dim=1),
            ):
                embedder = Embedder("/fake/model", "text")
                batch = {"text": np.array(["hello", None, 42])}
                _ = embedder(batch)

    passed_texts = mock_encoder.encode.call_args[0][0]
    assert passed_texts == ["hello", "", "42"]


def test_resolve_local_path():
    with tempfile.TemporaryDirectory() as tmp:
        assert _resolve_model_path(tmp) == Path(tmp)


def test_resolve_file_scheme():
    assert _resolve_model_path("file:///tmp/model") == Path("/tmp/model")


def test_resolve_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        _resolve_model_path("/nonexistent/path/12345")


def test_s3fs_import_error():
    """s3fs 未安装时应抛出 ImportError。"""
    with patch.dict("sys.modules", {"s3fs": None}):
        with pytest.raises(ImportError, match="s3fs is required"):
            from tributo.embeddings.batch_processor import _download_from_s3

            _download_from_s3("s3://bucket/models/test/")


def test_get_spec_runtime_error():
    """模型 spec 解析失败时应抛出 RuntimeError。"""
    from unittest.mock import patch

    with patch(
        "tributo.embeddings.batch_processor.get_spec",
        side_effect=ValueError("unknown model"),
    ):
        with pytest.raises(RuntimeError, match="Could not resolve spec"):
            _get_spec_from_path(Path("/fake/unregistered-model"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
