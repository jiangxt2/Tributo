"""XGBoost 训练 + ONNX 导出集成测试（基于 Jobs API + runtime_env 自动配置）。

运行方式：
    # 提交到 Docker Ray 集群
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run pytest tests/training/ -sv -m slow

    # 跳过 slow（CI 默认）
    uv run pytest tests/training/ -sv -m "not slow"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo.training import submit_training_job, wait_for_job

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def job_client():
    """Ray Jobs API 客户端，连接到集群 Dashboard。"""
    import requests

    # Ray Dashboard 对某些 User-Agent 返回 502，设置为 curl 兼容的头
    session = requests.Session()
    session.headers.update({"User-Agent": "curl/7.64.1"})

    # Monkey-patch requests 默认 session
    original_request = requests.api.request

    def patched_request(method, url, **kwargs):
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].setdefault("User-Agent", "curl/7.64.1")
        return original_request(method, url, **kwargs)

    requests.api.request = patched_request

    client = JobSubmissionClient("http://127.0.0.1:8265")
    yield client

    # 恢复原始 request
    requests.api.request = original_request


@pytest.fixture(autouse=True)
def disable_uv_runtime_env_hook():
    """禁用 uv runtime env hook，防止集群容器内找不到 uv。"""
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


@pytest.mark.slow
def test_xgboost_training_completes(job_client):
    """端到端训练可以跑完，result.json 包含 onnx_path。"""
    test_name = "test_completes"

    job_id = submit_training_job(
        entrypoint="python tests/training/jobs/xgboost_train_job.py",
        env_vars={
            "TEST_NAME": test_name,
            "NUM_ROUNDS": "10",
            "MAX_DEPTH": "3",
            "ETA": "0.3",
            "EARLY_STOPPING_ROUNDS": "5",
            "USE_VAL": "true",
        },
    )

    job_result = wait_for_job(job_client, job_id, timeout=180)
    assert job_result["status"] == JobStatus.SUCCEEDED, (
        f"Job failed:\n{job_result['logs']}"
    )

    result_json = (
        Path.home() / "Docker/ray-cluster/workspace/onnx" / f"{test_name}_result.json"
    )
    assert result_json.exists(), f"Result JSON not found: {result_json}"

    with open(result_json) as f:
        data = json.load(f)

    assert data["onnx_path"], "onnx_path should be in result"
    assert data["error"] is None, f"Training error: {data['error']}"
    assert data["metrics"] is not None, "metrics should not be None in cluster"


@pytest.mark.slow
def test_xgboost_training_without_val(job_client):
    """不传验证集时训练也能正常完成。"""
    test_name = "test_no_val"
    job_id = submit_training_job(
        entrypoint="python tests/training/jobs/xgboost_train_job.py",
        env_vars={
            "TEST_NAME": test_name,
            "NUM_ROUNDS": "5",
            "MAX_DEPTH": "3",
            "USE_VAL": "false",
        },
    )

    result = wait_for_job(job_client, job_id)
    assert result["status"] == JobStatus.SUCCEEDED, f"Job failed:\n{result['logs']}"

    result_json = (
        Path.home() / "Docker/ray-cluster/workspace/onnx" / f"{test_name}_result.json"
    )
    with open(result_json) as f:
        data = json.load(f)

    assert data["onnx_path"]
    assert data["error"] is None


@pytest.mark.slow
def test_onnx_model_inference(job_client):
    """导出的 ONNX 模型可以用 onnxruntime 做推理，输出 shape 正确。"""
    import onnxruntime as ort

    test_name = "test_inference"
    job_id = submit_training_job(
        entrypoint="python tests/training/jobs/xgboost_train_job.py",
        env_vars={
            "TEST_NAME": test_name,
            "NUM_ROUNDS": "5",
            "MAX_DEPTH": "3",
            "USE_VAL": "true",
        },
    )

    result = wait_for_job(job_client, job_id)
    assert result["status"] == JobStatus.SUCCEEDED, f"Job failed:\n{result['logs']}"

    # 读取 ONNX 模型（通过挂载路径）
    onnx_path = Path.home() / "Docker/ray-cluster/workspace/onnx" / f"{test_name}.onnx"
    assert onnx_path.exists(), f"ONNX model not found: {onnx_path}"

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.zeros((3, 10), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy})

    # binary:logistic 输出 label (shape [3,]) 和 probabilities (shape [3, 2])
    assert len(outputs) >= 1
    assert outputs[0].shape[0] == 3


@pytest.mark.slow
def test_max_rows_per_worker_limits_data(job_client):
    """max_rows_per_worker 生效时训练不报错，且实际使用行数受限。"""
    test_name = "test_max_rows"
    job_id = submit_training_job(
        entrypoint="python tests/training/jobs/xgboost_train_job.py",
        env_vars={
            "TEST_NAME": test_name,
            "NUM_ROUNDS": "3",
            "MAX_ROWS_PER_WORKER": "50",
            "USE_VAL": "false",
        },
    )

    result = wait_for_job(job_client, job_id)
    assert result["status"] == JobStatus.SUCCEEDED, f"Job failed:\n{result['logs']}"

    result_json = (
        Path.home() / "Docker/ray-cluster/workspace/onnx" / f"{test_name}_result.json"
    )
    with open(result_json) as f:
        data = json.load(f)

    assert data["onnx_path"]
    assert data["error"] is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", "-m", "slow", __file__]))
