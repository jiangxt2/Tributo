"""serving.grpc_runner 单元测试（mock Ray Serve）。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tributo.serving.grpc_runner import (
    DEFAULT_APP_NAME,
    DEFAULT_GRPC_PORT,
    get_grpc_serving_status,
    start_grpc_serving,
    stop_grpc_serving,
)


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_start_grpc_serving(mock_serve, mock_ray):
    """start_grpc_serving 正确调用 deploy_serve_app 并传入 grpc_port。"""
    mock_ray.is_initialized.return_value = True
    mock_deployment = mock_serve.deployment.return_value
    mock_deployment.bind.return_value = mock_deployment

    app_name = start_grpc_serving(
        "/workspace/onnx/model.onnx",
        app_name="test-grpc",
        grpc_port=8001,
    )

    assert app_name == "test-grpc"
    mock_serve.start.assert_called_once()
    call_kwargs = mock_serve.start.call_args
    assert call_kwargs.kwargs["grpc_options"]["port"] == 8001
    assert call_kwargs.kwargs["grpc_options"]["grpc_servicer_functions"] == [
        "tributo.serving.proto.inference_pb2_grpc"
        ".add_InferenceServiceServicer_to_server"
    ]


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_start_grpc_serving_default_values(mock_serve, mock_ray):
    """start_grpc_serving 默认参数正确。"""
    mock_ray.is_initialized.return_value = True
    mock_deployment = mock_serve.deployment.return_value
    mock_deployment.bind.return_value = mock_deployment

    app_name = start_grpc_serving("/workspace/onnx/model.onnx")

    assert app_name == DEFAULT_APP_NAME
    call_kwargs = mock_serve.start.call_args
    assert call_kwargs.kwargs["grpc_options"]["port"] == DEFAULT_GRPC_PORT
    assert call_kwargs.kwargs["grpc_options"]["grpc_servicer_functions"] == [
        "tributo.serving.proto.inference_pb2_grpc"
        ".add_InferenceServiceServicer_to_server"
    ]


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_start_grpc_serving_auto_route_prefix(mock_serve, mock_ray):
    """gRPC 服务自动生成 route_prefix（Ray Serve ROUTE_TABLE 要求非空）。"""
    mock_ray.is_initialized.return_value = True
    mock_deployment = mock_serve.deployment.return_value
    mock_deployment.bind.return_value = mock_deployment

    start_grpc_serving("/workspace/onnx/model.onnx")

    call_kwargs = mock_serve.run.call_args
    # gRPC 部署自动设置 route_prefix = f"/{app_name}"
    assert call_kwargs.kwargs["route_prefix"] == f"/{DEFAULT_APP_NAME}"


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_stop_grpc_serving(mock_serve, mock_ray):
    """stop_grpc_serving 正确调用 serve.delete。"""
    mock_ray.is_initialized.return_value = True

    result = stop_grpc_serving("test-grpc")

    assert result is True
    mock_serve.delete.assert_called_once_with("test-grpc")


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_stop_grpc_serving_default_app_name(mock_serve, mock_ray):
    """stop_grpc_serving 默认使用 DEFAULT_APP_NAME。"""
    mock_ray.is_initialized.return_value = True

    stop_grpc_serving()

    mock_serve.delete.assert_called_once_with(DEFAULT_APP_NAME)


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_get_grpc_serving_status_running(mock_serve, mock_ray):
    """get_grpc_serving_status 返回运行中状态。"""
    from unittest.mock import MagicMock

    mock_ray.is_initialized.return_value = True

    mock_app = MagicMock()
    mock_app.route_prefix = None
    mock_app.status = "RUNNING"
    mock_app.deployments = {"gRPCInferenceService": {}}

    mock_status = MagicMock()
    mock_status.applications = {DEFAULT_APP_NAME: mock_app}
    mock_serve.status.return_value = mock_status

    status = get_grpc_serving_status(DEFAULT_APP_NAME)

    assert status["running"] is True
    assert status["app_name"] == DEFAULT_APP_NAME
    assert status["status"] == "RUNNING"


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_get_grpc_serving_status_not_found(mock_serve, mock_ray):
    """应用不存在时返回 NOT_FOUND。"""
    mock_ray.is_initialized.return_value = True

    mock_status = mock_serve.status.return_value
    mock_status.applications = {}

    status = get_grpc_serving_status("nonexistent")

    assert status["running"] is False
    assert status["status"] == "NOT_FOUND"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
