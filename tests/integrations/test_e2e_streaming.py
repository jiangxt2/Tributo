"""Streaming 端到端测试。

流程：加载小模型 → 启动 Ray Serve → 流式推理 → 并发测试 → 清理

使用方式（Docker 容器内执行）：
    docker cp tests/integration/test_e2e_streaming.py ray-head:/opt/tributo/tests/integration/
    docker exec ray-head python /opt/tributo/tests/integration/test_e2e_streaming.py

前置条件：
    - Docker Ray 集群运行中（ray-head @ 127.0.0.1）
    - transformers + torch 已安装
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

import httpx
import pytest
import ray
from ray import serve

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# 设置 HuggingFace 缓存目录（避免权限问题）
os.environ["HF_HOME"] = "/tmp/hf_cache"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Ray Serve 配置
SERVE_HTTP = "http://127.0.0.1:8000"
APP_NAME = "e2e-streaming"
ROUTE_PREFIX = "/v1/chat/completions"

# 小模型配置（SmolLM-135M，约 270MB，base 模型无 chat template）
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


# ---------------------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------------------


async def test_streaming_inference():
    """测试流式推理功能。"""
    logger.info("=" * 50)
    logger.info("测试 1：流式推理")
    logger.info("=" * 50)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 使用 prompt 格式（base 模型无 chat template）
        response = await client.post(
            f"{SERVE_HTTP}{ROUTE_PREFIX}",
            json={
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
                "stream": True,
            },
        )

        assert response.status_code == 200, (
            f"HTTP {response.status_code}: {response.text}"
        )
        assert "text/event-stream" in response.headers.get("content-type", "")

        # 解析 SSE 流
        chunks = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    pass

        assert len(chunks) > 0, "没有收到任何 chunk"

        # 拼接所有 token
        tokens = []
        for chunk in chunks:
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    tokens.append(content)

        full_text = "".join(tokens)
        logger.info(f"生成文本: {full_text!r}")
        assert len(full_text) > 0, "生成文本为空"

    logger.info("✅ 测试 1 通过：流式推理")
    return True


async def test_concurrent_requests():
    """测试并发请求。"""
    logger.info("=" * 50)
    logger.info("测试 2：并发请求")
    logger.info("=" * 50)

    async def send_request(client: httpx.AsyncClient, request_id: int) -> str:
        response = await client.post(
            f"{SERVE_HTTP}{ROUTE_PREFIX}",
            json={
                "prompt": f"Say {request_id}",
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert response.status_code == 200, (
            f"Request {request_id} failed: {response.status_code}"
        )

        tokens = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if "choices" in chunk:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            tokens.append(content)
                except json.JSONDecodeError:
                    pass
        return "".join(tokens)

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [send_request(client, i) for i in range(3)]
        results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        logger.info(f"请求 {i}: {result!r}")
        assert len(result) > 0, f"请求 {i} 结果为空"

    logger.info("✅ 测试 2 通过：并发请求")
    return True


async def test_health_check():
    """测试健康检查。"""
    logger.info("=" * 50)
    logger.info("测试 3：健康检查")
    logger.info("=" * 50)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{SERVE_HTTP}/-/healthz")
        logger.info(f"健康检查状态: {response.status_code}")
        assert response.status_code == 200

    logger.info("✅ 测试 3 通过：健康检查")
    return True


def main():
    """Streaming 端到端测试。"""
    logger.info("开始 Streaming 端到端测试")

    # 连接 Ray Cluster
    try:
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            runtime_env={
                "working_dir": "/opt/tributo",
                "py_modules": ["/opt/tributo/src/tributo"],
                "env_vars": {"HF_HOME": "/tmp/hf_cache"},
            },
        )
        logger.info("已连接 Ray Cluster")
    except Exception:
        logger.exception("无法连接 Ray Cluster")
        sys.exit(1)

    # 启动 Ray Serve
    try:
        serve.start(http_options={"host": "0.0.0.0", "port": 8000})
        logger.info("Ray Serve 已启动")
    except Exception:
        logger.exception("无法启动 Ray Serve")
        sys.exit(1)

    # 部署 Streaming 服务（使用 deploy_serve_app 触发 reconfigure）
    try:
        from tributo._common.serve_utils import deploy_serve_app
        from tributo.serving.streaming_deployment import LLMStreamingService

        deploy_serve_app(
            LLMStreamingService,
            app_name=APP_NAME,
            route_prefix=ROUTE_PREFIX,
            num_replicas=1,
            user_config={},
            model_path=MODEL_NAME,
            tokenizer_path=MODEL_NAME,
            max_tokens=10,
        )
        logger.info(f"Streaming 服务已部署: {MODEL_NAME}")
    except Exception:
        logger.exception("无法部署 Streaming 服务")
        serve.shutdown()
        sys.exit(1)

    # 等待模型加载
    logger.info("等待模型加载...")
    time.sleep(45)

    # 运行测试
    results = {}
    tests = [
        ("流式推理", test_streaming_inference),
        ("并发请求", test_concurrent_requests),
        ("健康检查", test_health_check),
    ]

    for test_name, test_func in tests:
        try:
            success = asyncio.run(test_func())
            results[test_name] = success
        except Exception:
            logger.exception(f"测试 '{test_name}' 异常")
            results[test_name] = False

    # 清理
    logger.info("清理服务...")
    try:
        serve.delete(APP_NAME)
    except Exception:
        pass
    serve.shutdown()
    ray.shutdown()

    # 输出测试报告
    logger.info("\n" + "=" * 50)
    logger.info("测试报告")
    logger.info("=" * 50)

    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{status} - {test_name}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    logger.info(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        logger.info("\n🎉 所有测试通过！")
        return 0
    else:
        logger.error("\n💥 部分测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
