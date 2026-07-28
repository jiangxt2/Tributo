"""serving.grpc_deployment 单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tributo.serving.proto.generated import inference_pb2


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


def test_grpc_deployment_import():
    """gRPCInferenceService 可以正常导入。"""
    from tributo.serving.grpc_deployment import gRPCInferenceService

    assert gRPCInferenceService is not None


def test_grpc_runner_import():
    """gRPC runner 函数可以正常导入。"""
    from tributo.serving.grpc_runner import (
        get_grpc_serving_status,
        start_grpc_serving,
        stop_grpc_serving,
    )

    assert start_grpc_serving is not None
    assert stop_grpc_serving is not None
    assert get_grpc_serving_status is not None


def test_grpc_deployment_class_definition():
    """gRPCInferenceService 是普通类（不由 @serve.decoration 装饰）。"""
    from tributo.serving.grpc_deployment import gRPCInferenceService

    # 不应有 bind 属性（由 deploy_serve_app 统一处理装饰）
    assert not hasattr(gRPCInferenceService, "bind")
    # 应有 __init__ 和推理方法
    assert hasattr(gRPCInferenceService, "__init__")
    assert hasattr(gRPCInferenceService, "Predict")
    assert hasattr(gRPCInferenceService, "StreamPredict")
    assert hasattr(gRPCInferenceService, "BatchPredict")
    assert hasattr(gRPCInferenceService, "health")


def test_inference_pb2_import():
    """inference_pb2 可以正常导入并包含正确的消息类型。"""
    from tributo.serving.proto.generated import inference_pb2

    # 检查消息类型存在
    assert hasattr(inference_pb2, "PredictRequest")
    assert hasattr(inference_pb2, "PredictResponse")


def test_inference_pb2_grpc_import():
    """inference_pb2_grpc 可以正常导入并包含正确的服务类型。"""
    from tributo.serving.proto.generated import inference_pb2_grpc

    # 检查服务类型存在
    assert hasattr(inference_pb2_grpc, "InferenceServiceStub")
    assert hasattr(inference_pb2_grpc, "InferenceServiceServicer")


def test_predict_request_creation():
    """PredictRequest 可以正确创建。"""
    request = inference_pb2.PredictRequest(
        features=[0.5, 0.5],
        model_name="test",
    )

    assert request.features == [0.5, 0.5]
    assert request.model_name == "test"


def test_predict_response_creation():
    """PredictResponse 可以正确创建。"""
    response = inference_pb2.PredictResponse(
        predictions=[0.8, 0.2],
        confidence=0.8,
    )

    # protobuf float 字段使用 float32，会有精度损失
    assert list(response.predictions) == pytest.approx([0.8, 0.2], abs=1e-6)
    assert response.confidence == pytest.approx(0.8, abs=1e-6)


def test_serve_utils_supports_grpc():
    """serve_utils.py 的 deploy_serve_app 支持 gRPC 相关参数。"""
    import inspect

    from tributo._common.serve_utils import deploy_serve_app

    sig = inspect.signature(deploy_serve_app)
    assert "grpc_port" in sig.parameters
    assert sig.parameters["grpc_port"].default is None
    assert "grpc_servicer_functions" in sig.parameters
    assert sig.parameters["grpc_servicer_functions"].default is None
    assert "enable_http" in sig.parameters
    assert sig.parameters["enable_http"].default is True


def test_cli_grpc_commands():
    """CLI 中包含 grpc 命令组。"""
    from click.testing import CliRunner

    from tributo.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "grpc", "--help"])

    assert result.exit_code == 0
    assert "gRPC inference service management" in result.output


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
