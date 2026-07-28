"""测试用流式服务类。

独立模块，避免 Ray Serve 反序列化时导入 pytest 等测试依赖。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from tributo.serving.streaming_deployment import StreamingInferenceService


class DummyStreamingService(StreamingInferenceService):
    """用于集成测试的轻量子类，返回 fake token 流。"""

    def __init__(self, tokens: list[str] | None = None) -> None:
        super().__init__()
        self._tokens = tokens or ["Hello", " World", "!"]
        self._model_loaded = True

    async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        for token in self._tokens:
            yield json.dumps({"choices": [{"delta": {"content": token}}]})
            await asyncio.sleep(0.01)


class SlowStreamingService(StreamingInferenceService):
    """慢速流式服务，每个 token 间有明显延迟，用于验证逐步到达。"""

    def __init__(self) -> None:
        super().__init__()
        self._model_loaded = True

    async def _generate_stream(self, input_data: dict) -> AsyncGenerator[str, None]:
        tokens = ["First", " Second", " Third"]
        for token in tokens:
            yield json.dumps({"choices": [{"delta": {"content": token}}]})
            await asyncio.sleep(0.3)
