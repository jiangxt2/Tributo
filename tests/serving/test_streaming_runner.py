"""serving.streaming_runner 单元测试（mock Ray Serve）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tributo.serving.streaming_runner import (
    DEFAULT_APP_NAME,
    get_streaming_serving_status,
    start_streaming_serving,
    stop_streaming_serving,
)


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_start_streaming_serving(mock_serve, mock_ray):
    """U-07: start_streaming_serving 正确调用 serve.run 并传递 bind 参数。"""
    mock_ray.is_initialized.return_value = True
    mock_deployment = MagicMock()
    mock_serve.deployment.return_value = lambda cls: mock_deployment
    mock_serve.run.return_value = MagicMock()

    app_name = start_streaming_serving(
        model_path="/fake/model",
        tokenizer_path="/fake/tokenizer",
        app_name="test-streaming",
    )

    assert app_name == "test-streaming"
    mock_serve.start.assert_called_once()
    mock_deployment.bind.assert_called_once_with(
        model_path="/fake/model",
        tokenizer_path="/fake/tokenizer",
        max_tokens=512,
        max_workers=4,
    )
    mock_serve.run.assert_called_once()


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_stop_streaming_serving(mock_serve, mock_ray):
    """U-08: stop_streaming_serving 正确调用 serve.delete。"""
    mock_ray.is_initialized.return_value = True

    result = stop_streaming_serving("test-streaming")

    assert result is True
    mock_serve.delete.assert_called_once_with("test-streaming")


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_get_streaming_serving_status_running(mock_serve, mock_ray):
    """U-09: get_streaming_serving_status 返回运行中状态。"""
    mock_ray.is_initialized.return_value = True

    mock_app = {
        "route_prefix": "/v1/chat/completions",
        "status": "RUNNING",
        "deployments": {"LLMStreamingService": {}},
    }
    mock_status = MagicMock()
    mock_status.applications = {DEFAULT_APP_NAME: mock_app}
    mock_serve.status.return_value = mock_status

    status = get_streaming_serving_status(DEFAULT_APP_NAME)

    assert status["running"] is True
    assert status["app_name"] == DEFAULT_APP_NAME


@patch("tributo._common.serve_utils.ray")
@patch("tributo._common.serve_utils.serve")
def test_get_streaming_serving_status_not_found(mock_serve, mock_ray):
    """U-10: 应用不存在时返回 NOT_FOUND。"""
    mock_ray.is_initialized.return_value = True

    mock_status = MagicMock()
    mock_status.applications = {}
    mock_serve.status.return_value = mock_status

    status = get_streaming_serving_status("nonexistent")

    assert status["running"] is False
    assert status["status"] == "NOT_FOUND"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
