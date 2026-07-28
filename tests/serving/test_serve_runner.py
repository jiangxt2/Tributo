"""serving.serve_runner 单元测试（mock Ray Serve）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.serving.serve_runner import (
    DEFAULT_APP_NAME,
    get_serving_status,
    start_serving,
    stop_serving,
)


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_start_serving(mock_serve, mock_ray):
    """start_serving 正确调用 serve.run。"""
    mock_ray.is_initialized.return_value = True
    mock_deployment = MagicMock()
    mock_serve.deployment.return_value = lambda cls: mock_deployment
    mock_serve.run.return_value = MagicMock()

    app_name = start_serving("/workspace/onnx/model.onnx", app_name="test-app")

    assert app_name == "test-app"
    mock_serve.start.assert_called_once()
    mock_serve.run.assert_called_once()


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_stop_serving(mock_serve, mock_ray):
    """stop_serving 正确调用 serve.delete。"""
    mock_ray.is_initialized.return_value = True

    result = stop_serving("test-app")

    assert result is True
    mock_serve.delete.assert_called_once_with("test-app")


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_get_serving_status_running(mock_serve, mock_ray):
    """get_serving_status 返回运行中状态。"""
    mock_ray.is_initialized.return_value = True

    mock_app = {
        "route_prefix": "/predict",
        "status": "RUNNING",
        "deployments": {"ONNXModel": {}},
    }
    mock_status = MagicMock()
    mock_status.applications = {DEFAULT_APP_NAME: mock_app}
    mock_serve.status.return_value = mock_status

    status = get_serving_status(DEFAULT_APP_NAME)

    assert status["running"] is True
    assert status["app_name"] == DEFAULT_APP_NAME


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_get_serving_status_not_found(mock_serve, mock_ray):
    """应用不存在时返回 NOT_FOUND。"""
    mock_ray.is_initialized.return_value = True

    mock_status = MagicMock()
    mock_status.applications = {}
    mock_serve.status.return_value = mock_status

    status = get_serving_status("nonexistent")

    assert status["running"] is False
    assert status["status"] == "NOT_FOUND"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
