"""XGBoost 训练 + ONNX 导出集成测试（基于 Jobs API + runtime_env 自动配置）。

运行方式：
    # 提交到 Docker Ray 集群
    # 非默认 127.0.0.1:8265 的集群用 DEFAULT_DASHBOARD_URL 环境变量指定地址
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run pytest tests/training/ -sv -m slow

    # 跳过 slow（CI 默认）
    uv run pytest tests/training/ -sv -m "not slow"
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL
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

    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    yield client

    # 恢复原始 request
    requests.api.request = original_request


@pytest.fixture(autouse=True)
def disable_uv_runtime_env_hook():
    """禁用 uv runtime env hook，防止集群容器内找不到 uv。"""
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


def _result_from_logs(logs: str) -> dict[str, Any]:
    """从 job stdout 解析 ``RESULT:`` 行得到结果字典。

    训练产物（ONNX 模型、result.json）位于集群容器内，宿主机测试侧
    无法直接读取文件；job 脚本将完整结果打印到 stdout 回传。这样
    slow 测试不依赖任何宿主机挂载路径，任何集群部署均可运行。
    """
    for line in logs.splitlines():
        if line.startswith("RESULT: "):
            return json.loads(line[len("RESULT: ") :])
    raise AssertionError(f"RESULT: line not found in job logs:\n{logs}")


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

    data = _result_from_logs(job_result["logs"])

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

    data = _result_from_logs(result["logs"])

    assert data["onnx_path"]
    assert data["error"] is None


@pytest.mark.slow
def test_onnx_model_inference(job_client):
    """导出的 ONNX 模型可推理且输出 shape 正确。

    job 内 ``export_from_checkpoint(validate=True)`` 已完成 onnxruntime
    推理验证（失败会 raise → job FAILED）；宿主机侧从 logs 拿到产物路径
    与 metrics 即视为导出成功，无需访问容器内文件。
    """
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

    data = _result_from_logs(result["logs"])

    # binary:logistic: onnx 模型路径存在 + n_features 推断维度正确
    assert data["onnx_path"], "ONNX model path should be in result"
    assert data["onnx_path"].endswith(f"{test_name}.onnx")
    assert data["metrics"]["n_features"] == 10


@pytest.mark.slow
def test_max_rows_per_worker_fails_fast_when_exceeded(job_client):
    """T3 Core: max_rows_per_worker 超过限制时训练失败，不静默截断。

    每 worker shard 约 320 行 > MAX_ROWS_PER_WORKER=50 → 训练必须失败，
    不得像旧实现那样只使用前 50 行继续成功。
    """
    test_name = "test_max_rows_fail_fast"
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
    assert result["status"] == JobStatus.FAILED, (
        f"Job unexpectedly succeeded — rows must not be truncated:\n{result['logs']}"
    )
    assert "row limit exceeded" in result["logs"]


@pytest.mark.slow
def test_max_rows_within_limit_uses_all_rows(job_client):
    """T3 Core: 行数限制高于 shard 行数时训练成功且使用全部行。"""
    test_name = "test_max_rows_all_rows"
    job_id = submit_training_job(
        entrypoint="python tests/training/jobs/xgboost_train_job.py",
        env_vars={
            "TEST_NAME": test_name,
            "NUM_ROUNDS": "3",
            "MAX_ROWS_PER_WORKER": "10000",  # 远高于 shard 行数
            "USE_VAL": "false",
        },
    )

    result = wait_for_job(job_client, job_id)
    assert result["status"] == JobStatus.SUCCEEDED, f"Job failed:\n{result['logs']}"

    data = _result_from_logs(result["logs"])

    assert data["onnx_path"]
    assert data["error"] is None
    # 800 行 → train 640 行 → 2 workers → 每 shard 320 行，全部保留
    assert data["metrics"]["row_count_train"] == 320


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", "-m", "slow", __file__]))
