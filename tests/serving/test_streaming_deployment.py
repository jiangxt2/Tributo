"""serving.streaming_deployment 单元测试。"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tributo.serving.streaming_deployment import (
    LLMStreamingService,
    StreamingInferenceService,
)

# ---------------------------------------------------------------------------
# 测试用的具体子类
# ---------------------------------------------------------------------------


class DummyStreamingService(StreamingInferenceService):
    """用于测试基类的简单子类。"""

    def __init__(self) -> None:
        super().__init__()
        self._model_loaded = True
        self.received_input: dict | None = None

    async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        """返回固定 token 流。"""
        self.received_input = input_data
        yield json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
        yield json.dumps({"choices": [{"delta": {"content": " World"}}]})


# ---------------------------------------------------------------------------
# StreamingInferenceService 基类测试
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_format():
    """SSE 响应格式正确：data: 前缀 + [DONE] 结束标记。"""
    service = DummyStreamingService()

    # 构造 mock request
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(return_value={"prompt": "test"})

    response = await service(mock_request)

    # 收集所有 SSE 事件
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # 验证格式
    assert len(chunks) == 3  # Hello + World + [DONE]
    assert chunks[0] == 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    assert chunks[1] == 'data: {"choices": [{"delta": {"content": " World"}}]}\n\n'
    assert chunks[2] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_invalid_json():
    """Invalid JSON 请求返回 400 错误。"""
    service = DummyStreamingService()

    mock_request = AsyncMock()
    mock_request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

    response = await service(mock_request)

    assert response.status_code == 400
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    assert "Invalid JSON" in chunks[0]


@pytest.mark.asyncio
async def test_streaming_error():
    """流式生成过程中出错，返回错误信息。"""

    class ErrorStreamingService(StreamingInferenceService):
        def __init__(self) -> None:
            super().__init__()
            self._model_loaded = True

        async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
            raise RuntimeError("Model crashed")
            # 必须包含 yield，否则该函数会被识别为普通协程而非 async generator。
            yield  # pragma: no cover

    service = ErrorStreamingService()
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(return_value={"prompt": "test"})

    response = await service(mock_request)

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # 错误信息 + [DONE]
    assert any("Model crashed" in c for c in chunks)
    assert any("[DONE]" in c for c in chunks)


@pytest.mark.asyncio
async def test_health_model_loaded():
    """模型加载成功时 health 返回 healthy。"""
    service = DummyStreamingService()
    health = await service.health()

    assert health["status"] == "healthy"
    assert health["model_loaded"] is True


@pytest.mark.asyncio
async def test_health_model_not_loaded():
    """模型未加载时 health 返回 unhealthy。"""
    service = DummyStreamingService()
    service._model_loaded = False
    health = await service.health()

    assert health["status"] == "unhealthy"
    assert health["model_loaded"] is False


# ---------------------------------------------------------------------------
# LLMStreamingService 测试
# ---------------------------------------------------------------------------


def test_llm_service_init_does_not_load_model():
    """LLMStreamingService 初始化时不阻塞，模型在 reconfigure 中加载。"""
    with (
        patch("transformers.AutoTokenizer") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM") as mock_model,
    ):
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )

        assert service._model_loaded is False
        mock_tokenizer.from_pretrained.assert_not_called()
        mock_model.from_pretrained.assert_not_called()


def test_llm_service_init_load_failure():
    """构造函数本身不加载模型，不会抛出加载异常。"""
    with patch("transformers.AutoTokenizer") as mock_tokenizer:
        mock_tokenizer.from_pretrained.side_effect = RuntimeError("Load failed")

        # 同步 __init__ 不调用 from_pretrained，因此不会抛异常
        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )
        assert service._model_loaded is False


@pytest.mark.asyncio
async def test_llm_service_reconfigure_loads_model():
    """reconfigure 中异步加载模型并传递正确参数。"""
    import torch as _torch

    with (
        patch("transformers.AutoTokenizer") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM") as mock_model,
        patch.object(_torch.cuda, "is_available", return_value=False),
    ):
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )
        await service.reconfigure({})

        assert service._model_loaded is True
        mock_tokenizer.from_pretrained.assert_called_once_with("/fake/tokenizer")
        mock_model.from_pretrained.assert_called_once_with(
            "/fake/model",
            torch_dtype=_torch.float32,
            device_map=None,
        )
        mock_model_instance.to.assert_called_once_with("cpu")
        service._model.eval.assert_called_once()


@pytest.mark.asyncio
async def test_llm_service_reconfigure_load_failure():
    """reconfigure 中模型加载失败时抛出异常。"""
    with patch("transformers.AutoTokenizer") as mock_tokenizer:
        mock_tokenizer.from_pretrained.side_effect = RuntimeError("Load failed")

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )
        with pytest.raises(RuntimeError, match="Load failed"):
            await service.reconfigure({})


@pytest.mark.asyncio
async def test_llm_service_model_not_loaded():
    """模型未加载时返回错误信息。"""
    with (
        patch("transformers.AutoTokenizer") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM") as mock_model,
    ):
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )
        # 显式模拟未加载状态
        service._model_loaded = False

        chunks = []
        async for chunk in service._generate_stream({"prompt": "test"}):
            chunks.append(json.loads(chunk))

        assert "error" in chunks[0]
        assert "Model not loaded" in chunks[0]["error"]


@pytest.mark.asyncio
async def test_llm_service_model_loaded_generate():
    """模型加载后调用 _generate_stream 返回 token 流。"""
    import torch as _torch

    with (
        patch("transformers.AutoTokenizer") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM") as mock_model,
    ):

        class _MockBatchEncoding(dict):
            """模拟 transformers BatchEncoding 的 .to(device) 行为。"""

            def to(self, device):
                return self

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.decode.return_value = "ok"
        mock_tokenizer_instance.return_value = _MockBatchEncoding(
            {
                "input_ids": _torch.tensor([[1, 2, 3]]),
                "attention_mask": _torch.tensor([[1, 1, 1]]),
            }
        )
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_outputs = MagicMock()
        mock_outputs.past_key_values = None
        mock_outputs.logits = _torch.tensor([[[0.0, 1.0, 0.0]]])
        mock_model_instance = MagicMock()
        mock_model_instance.device = _torch.device("cpu")
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = mock_outputs
        mock_model.from_pretrained.return_value = mock_model_instance

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )
        await service.reconfigure({})

        chunks = []
        async for chunk in service._generate_stream(
            {"prompt": "test", "max_tokens": 1}
        ):
            chunks.append(json.loads(chunk))

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "ok"
        assert chunks[1]["choices"][0]["finish_reason"] == "stop"

        # 验证单步 forward 传入了 attention_mask 与 position_ids
        call_kwargs = mock_model_instance.call_args.kwargs
        assert "attention_mask" in call_kwargs
        assert "position_ids" in call_kwargs
        assert "past_key_values" in call_kwargs


# ---------------------------------------------------------------------------
# U-01 ~ U-06：扩展测试
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_headers():
    """U-01: SSE 响应 Headers 正确性。"""
    service = DummyStreamingService()
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(return_value={"prompt": "test"})

    response = await service(mock_request)

    assert response.media_type == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["Connection"] == "keep-alive"
    assert response.headers["X-Accel-Buffering"] == "no"


@pytest.mark.asyncio
async def test_openai_messages_format():
    """U-02: OpenAI messages 格式透传到 _generate_stream。"""
    service = DummyStreamingService()
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(
        return_value={
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }
    )

    response = await service(mock_request)
    async for _ in response.body_iterator:
        pass

    # 原始 messages 数据透传到 _generate_stream
    assert service.received_input is not None
    assert "messages" in service.received_input
    assert len(service.received_input["messages"]) == 2
    assert service.received_input["messages"][0]["content"] == "You are helpful."
    assert service.received_input["messages"][1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_simple_prompt_format():
    """U-03: 简化 prompt 格式解析。"""
    service = DummyStreamingService()
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(return_value={"prompt": "直接输入"})

    response = await service(mock_request)
    async for _ in response.body_iterator:
        pass

    assert service.received_input is not None
    assert service.received_input["prompt"] == "直接输入"


@pytest.mark.asyncio
async def test_max_tokens_and_temperature_passthrough():
    """U-04: max_tokens / temperature 参数透传。"""

    class ParamCaptureService(StreamingInferenceService):
        def __init__(self) -> None:
            super().__init__()
            self._model_loaded = True
            self.captured: dict | None = None

        async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
            self.captured = input_data
            yield json.dumps({"choices": [{"delta": {"content": "ok"}}]})

    service = ParamCaptureService()
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(
        return_value={"prompt": "test", "max_tokens": 100, "temperature": 0.5}
    )

    response = await service(mock_request)
    async for _ in response.body_iterator:
        pass

    assert service.captured is not None
    assert service.captured["max_tokens"] == 100
    assert service.captured["temperature"] == 0.5


@pytest.mark.asyncio
async def test_chunk_concatenation():
    """U-05: 多 chunk 拼接等于完整结果。"""
    service = DummyStreamingService()
    mock_request = AsyncMock()
    mock_request.json = AsyncMock(return_value={"prompt": "test"})

    response = await service(mock_request)

    contents = []
    async for chunk in response.body_iterator:
        line = chunk.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            data = json.loads(line[6:])
            delta = data["choices"][0]["delta"]
            if "content" in delta:
                contents.append(delta["content"])

    assert "".join(contents) == "Hello World"


def test_llm_service_executor_created():
    """U-06: LLM 线程池 executor 正确创建。"""
    with (
        patch("transformers.AutoTokenizer") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM") as mock_model,
    ):
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
            max_workers=2,
        )

        import concurrent.futures

        assert isinstance(service._executor, concurrent.futures.ThreadPoolExecutor)
        assert service._executor._max_workers == 2
        assert service._model_loaded is False


def test_llm_service_executor_shutdown_on_del():
    """U-07: 实例销毁时 ThreadPoolExecutor 被正确关闭。"""
    with (
        patch("transformers.AutoTokenizer") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM") as mock_model,
    ):
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        service = LLMStreamingService(
            model_path="/fake/model",
            tokenizer_path="/fake/tokenizer",
        )
        executor = service._executor
        with patch.object(executor, "shutdown") as mock_shutdown:
            service.__del__()
            mock_shutdown.assert_called_once_with(wait=False, cancel_futures=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
