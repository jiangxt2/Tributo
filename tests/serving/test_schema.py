"""serving.schema 单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tributo.serving.schema import PredictRequest, PredictResponse


def test_predict_request_valid():
    """正常构造 PredictRequest。"""
    req = PredictRequest(features=[[0.1, 0.2], [0.3, 0.4]])
    assert req.features == [[0.1, 0.2], [0.3, 0.4]]
    assert req.return_probs is True


def test_predict_request_return_probs_false():
    """显式设置 return_probs=False。"""
    req = PredictRequest(features=[[1.0]], return_probs=False)
    assert req.return_probs is False


def test_predict_request_empty_features_raises():
    """空 features 应触发 ValidationError。"""
    with pytest.raises(ValidationError):
        PredictRequest(features=[])


def test_predict_response_serialization():
    """PredictResponse 可正确序列化。"""
    resp = PredictResponse(
        predictions=[[0.8, 0.2], [0.3, 0.7]],
        model_path="/workspace/onnx/model.onnx",
        inference_time_ms=12.34,
    )
    dumped = resp.model_dump()
    assert dumped["model_path"] == "/workspace/onnx/model.onnx"
    assert dumped["inference_time_ms"] == 12.34


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
