"""serving.streaming_deployment 集成测试（真实 Ray 集群）。

需要 Docker Ray 集群运行中（ray-head @ 127.0.0.1:8265）。
标记 @pytest.mark.slow，默认跳过。
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest
import ray
from ray import serve

from tests.serving._streaming_test_services import (
    DummyStreamingService,
    SlowStreamingService,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 容器内使用 auto，本地使用 ray://
RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")
SERVE_HTTP = os.environ.get("SERVE_HTTP", "http://127.0.0.1:8000")
APP_NAME = "test-streaming"
ROUTE_PREFIX = "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_serve_instance():
    """模块级 Ray + Serve 生命周期。"""
    import os

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    tributo_src = os.path.join(project_root, "src", "tributo")
    ray.init(
        address=RAY_ADDRESS,
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": project_root,
            "py_modules": [tributo_src],
        },
    )
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    yield
    serve.shutdown()


@pytest.fixture()
def deploy_dummy(ray_serve_instance):
    """部署 DummyStreamingService，测试结束后清理。"""
    deployment = serve.deployment(num_replicas=1)(DummyStreamingService)
    serve.run(
        deployment.bind(),
        name=APP_NAME,
        route_prefix=ROUTE_PREFIX,
    )
    yield
    serve.delete(APP_NAME)


@pytest.fixture()
def deploy_slow(ray_serve_instance):
    """部署 SlowStreamingService，测试结束后清理。"""
    deployment = serve.deployment(num_replicas=1)(SlowStreamingService)
    serve.run(
        deployment.bind(),
        name=f"{APP_NAME}-slow",
        route_prefix=f"{ROUTE_PREFIX}-slow",
    )
    yield
    serve.delete(f"{APP_NAME}-slow")


# ---------------------------------------------------------------------------
# I-01 ~ I-06
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_streaming(deploy_dummy):
    """I-01: 端到端流式返回，chunk 内容正确。"""
    url = f"{SERVE_HTTP}{ROUTE_PREFIX}"
    lines = []
    with httpx.stream("POST", url, json={"prompt": "test"}, timeout=10) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.strip():
                lines.append(line)

    data_lines = [
        ln for ln in lines if ln.startswith("data: ") and ln != "data: [DONE]"
    ]
    assert len(data_lines) == 3

    contents = []
    for line in data_lines:
        data = json.loads(line.removeprefix("data: "))
        contents.append(data["choices"][0]["delta"]["content"])

    assert "".join(contents) == "Hello World!"
    assert lines[-1] == "data: [DONE]"


@pytest.mark.slow
def test_responses_actually_streamed(deploy_slow):
    """I-02: 流式逐步到达验证 — 每个 chunk 有明显间隔。"""
    url = f"{SERVE_HTTP}{ROUTE_PREFIX}-slow"
    timestamps = []
    with httpx.stream("POST", url, json={"prompt": "test"}, timeout=10) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                timestamps.append(time.monotonic())

    # 至少 3 个 chunk，且每个 chunk 之间有明显间隔（>0.2s）
    assert len(timestamps) == 3
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        assert gap > 0.2, f"Chunk {i} arrived too fast ({gap:.3f}s), not truly streamed"


@pytest.mark.slow
def test_concurrent_streaming(deploy_dummy):
    """I-04: 并发流式请求，各请求独立。"""
    url = f"{SERVE_HTTP}{ROUTE_PREFIX}"

    def collect_chunks() -> list[str]:
        contents = []
        with httpx.stream("POST", url, json={"prompt": "test"}, timeout=10) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line.removeprefix("data: "))
                    delta = data["choices"][0]["delta"]
                    if "content" in delta:
                        contents.append(delta["content"])
        return contents

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(collect_chunks) for _ in range(3)]
        results = [f.result() for f in futures]

    for result in results:
        assert "".join(result) == "Hello World!"


@pytest.mark.slow
def test_client_disconnect_no_leak(ray_serve_instance):
    """I-03: 客户端断开后服务端不崩溃、不泄漏资源。"""
    # 部署慢速服务（每个 token 间隔 0.3s）
    deployment = serve.deployment(num_replicas=1)(SlowStreamingService)
    serve.run(
        deployment.bind(),
        name="disconnect-test",
        route_prefix="/disconnect-test",
    )

    try:
        url = f"{SERVE_HTTP}/disconnect-test"

        # 用极短超时强制客户端在收到第一个 chunk 后断开
        with pytest.raises((httpx.ReadTimeout, httpx.RemoteProtocolError)):
            with httpx.stream("POST", url, json={"prompt": "test"}, timeout=0.15) as r:
                for _ in r.iter_lines():
                    pass

        # 验证服务端仍然可用：发送正常请求应得到完整响应
        for _ in range(2):
            lines = []
            with httpx.stream("POST", url, json={"prompt": "test"}, timeout=10) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if line.strip():
                        lines.append(line)

            data_lines = [
                ln for ln in lines if ln.startswith("data: ") and ln != "data: [DONE]"
            ]
            assert len(data_lines) == 3

        # 验证 Serve 应用仍被控制器正常管理
        status = serve.status()
        assert "disconnect-test" in status.applications
        assert status.applications["disconnect-test"].status in ("RUNNING", "DEPLOYING")
    finally:
        try:
            serve.delete("disconnect-test")
        except Exception:
            pass


@pytest.mark.slow
def test_sse_done_marker(deploy_dummy):
    """I-05: SSE 结束标记。"""
    url = f"{SERVE_HTTP}{ROUTE_PREFIX}"
    lines = []
    with httpx.stream("POST", url, json={"prompt": "test"}, timeout=10) as r:
        for line in r.iter_lines():
            if line.strip():
                lines.append(line)

    assert lines[-1] == "data: [DONE]"


@pytest.mark.slow
def test_cli_streaming_commands(ray_serve_instance):
    """I-06: CLI tributo serve streaming status/stop 命令。"""
    import subprocess

    # 先部署一个 app 供 CLI 测试
    deployment = serve.deployment(num_replicas=1)(DummyStreamingService)
    serve.run(
        deployment.bind(),
        name="cli-test-streaming",
        route_prefix="/cli-test",
    )

    try:
        # status
        result = subprocess.run(
            [
                "tributo",
                "serve",
                "streaming",
                "status",
                "--app-name",
                "cli-test-streaming",
                "--ray-address",
                RAY_ADDRESS,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "cli-test-streaming" in result.stdout

        # stop
        result = subprocess.run(
            [
                "tributo",
                "serve",
                "streaming",
                "stop",
                "--app-name",
                "cli-test-streaming",
                "--ray-address",
                RAY_ADDRESS,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "stopped" in result.stdout.lower()
    finally:
        # 确保清理
        try:
            serve.delete("cli-test-streaming")
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", "-m", "slow", __file__]))
