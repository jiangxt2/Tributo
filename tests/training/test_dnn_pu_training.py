"""DNN and PU training integration tests executed through Ray Jobs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any, cast

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL
from tributo.training import submit_training_job, wait_for_job

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def job_client() -> Iterator[JobSubmissionClient]:
    """Connect to the Docker Ray cluster through its Jobs API."""
    import requests

    original_request = requests.api.request

    def patched_request(method: str, url: str, **kwargs: Any) -> Any:
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("User-Agent", "curl/7.64.1")
        kwargs["headers"] = headers
        return original_request(method, url, **kwargs)

    patcher = pytest.MonkeyPatch()
    patcher.setattr(requests.api, "request", patched_request)
    try:
        client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
        yield client
    finally:
        patcher.undo()


@pytest.fixture(autouse=True)
def disable_uv_runtime_env_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cluster image provides dependencies without a nested uv runtime."""
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _result_from_logs(logs: str) -> dict[str, Any]:
    for line in logs.splitlines():
        if line.startswith("RESULT: "):
            return cast(dict[str, Any], json.loads(line.removeprefix("RESULT: ")))
    raise AssertionError(f"RESULT line not found in job logs:\n{logs}")


@pytest.mark.parametrize("algorithm", ["dnn", "dnn_nnpu", "pu"])
def test_training_completes_on_ray_cluster(
    job_client: JobSubmissionClient, algorithm: str
) -> None:
    """The public training API produces finite metrics and a real ONNX model."""
    job_id = submit_training_job(
        entrypoint="python tests/training/jobs/dnn_pu_train_job.py",
        env_vars={"ALGORITHM": algorithm},
        run_id=f"correctness-{algorithm}-{uuid.uuid4().hex}",
    )

    job_result = wait_for_job(job_client, job_id, timeout=300)
    assert job_result["status"] == JobStatus.SUCCEEDED, (
        f"{algorithm} job failed:\n{job_result['logs']}"
    )

    result = _result_from_logs(job_result["logs"])
    assert result["algorithm"] == algorithm
    assert result["status"] == "succeeded"
    assert result["epoch"] == 1
    assert result["onnx_exists"] is True
    assert result["onnx_size"] > 0
    if algorithm == "pu":
        assert result["class_prior"] == pytest.approx(0.35)
    else:
        assert result["class_prior"] is None
    if algorithm in {"dnn_nnpu", "pu"}:
        assert "train_optimization_objective" in result
        assert 0.0 <= result["train_observed_label_accuracy"] <= 1.0
        assert result["train_acc"] == result["train_observed_label_accuracy"]
        assert result["val_acc"] == result["val_observed_label_accuracy"]
