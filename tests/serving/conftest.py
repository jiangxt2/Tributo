"""serving 模块共享 fixtures。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_grpc_context():
    """Ray 提供的 FakeGrpcContext mock。"""
    from ray.serve._private.test_utils import FakeGrpcContext

    return FakeGrpcContext()


@pytest.fixture
def mock_onnx_session():
    """mock ONNX InferenceSession，返回固定预测结果。"""
    with patch("onnxruntime.InferenceSession") as mock:
        session = MagicMock()
        session.run.return_value = [np.array([[0.8, 0.2]])]
        mock_input = MagicMock()
        mock_input.name = "float_input"
        session.get_inputs.return_value = [mock_input]
        mock.return_value = session
        yield session


@pytest.fixture
def dummy_onnx_path(tmp_path):
    """生成临时 ONNX 模型文件。"""
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
