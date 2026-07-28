"""serving.model_deployment 单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo.serving.model_deployment import ONNXModel
from tributo.serving.schema import PredictRequest


def _make_dummy_onnx(tmp_path: Path) -> str:
    """生成一个最小可用的 ONNX 分类模型文件，用于测试。"""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("skl2onnx or sklearn not installed")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    clf = LogisticRegression().fit(X, y)

    initial_types = [("float_input", FloatTensorType([None, 2]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_types)

    path = str(tmp_path / "dummy.onnx")
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def test_onnx_model_loads_and_predicts(tmp_path: Path):
    """ONNXModel 能加载模型并返回正确 shape 的预测结果。"""
    model_path = _make_dummy_onnx(tmp_path)
    deployment = ONNXModel(model_path=model_path)

    req = PredictRequest(features=[[0.5, 0.5], [0.1, 0.9]])
    resp = deployment._predict(req)

    assert resp.model_path == model_path
    assert len(resp.predictions) == 2
    assert resp.inference_time_ms >= 0


def test_onnx_model_return_probs_false(tmp_path: Path):
    """return_probs=False 时返回 label 而非概率。"""
    model_path = _make_dummy_onnx(tmp_path)
    deployment = ONNXModel(model_path=model_path)

    req = PredictRequest(features=[[0.5, 0.5]], return_probs=False)
    resp = deployment._predict(req)

    # label 应为 int 类型
    assert isinstance(resp.predictions[0], (int, np.integer))


def test_onnx_model_health():
    """health() 返回预期字段。"""
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.return_value.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.return_value.get_outputs.return_value = []

        deployment = ONNXModel(model_path="/fake/model.onnx")
        health = deployment.health()

    assert health["status"] == "healthy"
    assert health["model_path"] == "/fake/model.onnx"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
