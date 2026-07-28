"""serving.streaming_deployment HTTP/SSE 层测试（ASGI 直连）。

使用 httpx.AsyncClient + Starlette ASGI 直连，不启动 Ray Serve。
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

from tributo.serving.streaming_deployment import StreamingInferenceService

# ---------------------------------------------------------------------------
# 测试用子类
# ---------------------------------------------------------------------------


class DummyStreamingService(StreamingInferenceService):
    """返回固定 token 流的轻量子类。"""

    def __init__(self) -> None:
        super().__init__()
        self._model_loaded = True

    async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        yield json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
        yield json.dumps({"choices": [{"delta": {"content": " World"}}]})


class ErrorStreamingService(StreamingInferenceService):
    """生成过程中抛异常的子类。"""

    def __init__(self) -> None:
        super().__init__()
        self._model_loaded = True

    async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        raise RuntimeError("Model crashed")
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_app() -> Starlette:
    service = DummyStreamingService()

    async def endpoint(request: Request):
        return await service(request)

    return Starlette(routes=[Route("/v1/chat/completions", endpoint, methods=["POST"])])


@pytest.fixture()
def error_app() -> Starlette:
    service = ErrorStreamingService()

    async def endpoint(request: Request):
        return await service(request)

    return Starlette(routes=[Route("/v1/chat/completions", endpoint, methods=["POST"])])


# ---------------------------------------------------------------------------
# H-01 ~ H-04
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_event_stream_format(dummy_app: Starlette):
    """H-01: SSE 事件流格式 — data: 前缀 + [DONE] 结束标记。"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        lines = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"prompt": "test"},
        ) as response:
            async for line in response.aiter_lines():
                lines.append(line)

    # 过滤空行
    data_lines = [ln for ln in lines if ln.strip()]
    assert len(data_lines) == 3
    assert data_lines[0].startswith("data: ")
    assert data_lines[1].startswith("data: ")
    assert data_lines[2] == "data: [DONE]"


@pytest.mark.asyncio
async def test_http_status_200(dummy_app: Starlette):
    """H-02a: 正常请求返回 200。"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"prompt": "test"},
        ) as response:
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_http_status_400_invalid_json(dummy_app: Starlette):
    """H-02b: Invalid JSON 请求返回 400。"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            content="not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_content_type_header(dummy_app: Starlette):
    """H-03: Content-Type 为 text/event-stream。"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"prompt": "test"},
        ) as response:
            assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_generator_error_returns_error_chunk(error_app: Starlette):
    """H-04: 生成器异常返回 data: {"error": ...}。"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=error_app), base_url="http://test"
    ) as client:
        lines = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"prompt": "test"},
        ) as response:
            async for line in response.aiter_lines():
                lines.append(line)

    data_lines = [ln for ln in lines if ln.strip()]
    # 应包含错误信息行
    error_lines = [ln for ln in data_lines if "error" in ln and ln.startswith("data: ")]
    assert len(error_lines) >= 1
    error_data = json.loads(error_lines[0].removeprefix("data: "))
    assert "Model crashed" in error_data["error"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
