"""Real Ray Jobs tests for sklearn and user-function portable execution."""

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
    """Connect to the existing Docker Ray cluster through the Jobs API."""
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
        yield JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    finally:
        patcher.undo()


@pytest.fixture(autouse=True)
def disable_uv_runtime_env_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use dependencies already installed in the controlled cluster image."""
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _result_from_logs(logs: str) -> dict[str, Any]:
    for line in logs.splitlines():
        marker = "RESULT: "
        marker_index = line.find(marker)
        if marker_index >= 0:
            payload = line[marker_index + len(marker) :]
            result, _end = json.JSONDecoder().raw_decode(payload)
            return cast(dict[str, Any], result)
    raise AssertionError(f"RESULT line not found in Ray Job logs:\n{logs}")


@pytest.mark.parametrize("channel", ["sklearn", "function"])
def test_portable_channel_executes_in_real_ray_job(
    job_client: JobSubmissionClient,
    channel: str,
) -> None:
    """Each channel loads implementation code in a real Ray Worker task."""
    job_id = submit_training_job(
        entrypoint=(
            "python tests/integration/algorithm_jobs/portable_execution_job.py"
        ),
        env_vars={"PORTABLE_CHANNEL": channel},
        run_id=f"portable-{channel}-{uuid.uuid4().hex}",
    )
    job_result = wait_for_job(job_client, job_id, timeout=240)
    assert job_result["status"] == JobStatus.SUCCEEDED, job_result["logs"]
    result = _result_from_logs(job_result["logs"])

    assert result["channel"] == channel
    assert result["actual_ray"] == "2.55.1"
    assert result["driver_imported_user_module"] is False
    assert result["driver_imported_sklearn"] is False
    if channel == "sklearn":
        assert result["fit_accuracy"] == 1.0
        assert result["evaluate_accuracy"] == 1.0
        assert result["prediction_count"] == 8
        assert len(result["artifact_sha256"]) == 64
        assert result["actual_sklearn"]
        assert result["close_calls"] == ["closed", "closed", "closed"]
    else:
        assert result["positive_rate"] == 0.5
        assert result["threshold"] == 0.8
        assert result["worker_id"]
        assert result["artifact_kinds"] == ["report", "checkpoint"]
        assert result["close_calls"] == ["closed"]


def test_user_failure_is_normalized_in_real_ray_job(
    job_client: JobSubmissionClient,
) -> None:
    """User exceptions become stable Worker failures without failing the Job."""
    job_id = submit_training_job(
        entrypoint=(
            "python tests/integration/algorithm_jobs/portable_execution_job.py"
        ),
        env_vars={"PORTABLE_CHANNEL": "function_failure"},
        run_id=f"portable-function-failure-{uuid.uuid4().hex}",
    )
    job_result = wait_for_job(job_client, job_id, timeout=240)
    assert job_result["status"] == JobStatus.SUCCEEDED, job_result["logs"]
    result = _result_from_logs(job_result["logs"])

    assert result["channel"] == "function_failure"
    assert result["status"] == "failed"
    assert result["failure_category"] == "execution"
    assert result["error_type"] == "AlgorithmExecutionError"
    assert "hunter2" not in result["error_message"]
    assert "abc123" not in result["error_message"]
    assert "alice:private" not in result["error_message"]
    assert result["error_message"].count("<redacted>") == 3
    assert result["driver_imported_user_module"] is False
    assert result["driver_imported_sklearn"] is False
    assert result["close_calls"] == ["cancelled"]


def test_sklearn_executes_with_production_ingestion_bridge(
    job_client: JobSubmissionClient,
) -> None:
    """Managed sklearn consumes a real RayDataHandle opened by the Gateway."""
    job_id = submit_training_job(
        entrypoint=(
            "python tests/integration/algorithm_jobs/portable_execution_job.py"
        ),
        env_vars={"PORTABLE_CHANNEL": "ingestion_sklearn"},
        run_id=f"portable-ingestion-sklearn-{uuid.uuid4().hex}",
    )
    job_result = wait_for_job(job_client, job_id, timeout=240)
    assert job_result["status"] == JobStatus.SUCCEEDED, job_result["logs"]
    result = _result_from_logs(job_result["logs"])

    assert result["channel"] == "ingestion_sklearn"
    assert result["fit_accuracy"] == 1.0
    assert result["evaluate_accuracy"] == 1.0
    assert result["prediction_count"] == 8
    assert result["input_resolver"] == "tributo.ingestion"
    assert result["input_engine"] == "tributo.ray_data"
    assert result["ingestion_lifecycle"] == ["closed", "closed", "closed"]
    assert result["request_body_in_provenance"] is False
    assert result["driver_imported_user_module"] is False
    assert result["driver_imported_sklearn"] is False


def test_user_function_owns_ingestion_lifecycle_in_real_ray_job(
    job_client: JobSubmissionClient,
) -> None:
    """User success, failure, and cancellation release one Gateway owner."""
    job_id = submit_training_job(
        entrypoint=(
            "python tests/integration/algorithm_jobs/portable_execution_job.py"
        ),
        env_vars={"PORTABLE_CHANNEL": "ingestion_function"},
        run_id=f"portable-ingestion-function-{uuid.uuid4().hex}",
    )
    job_result = wait_for_job(job_client, job_id, timeout=240)
    assert job_result["status"] == JobStatus.SUCCEEDED, job_result["logs"]
    result = _result_from_logs(job_result["logs"])
    success = result["success"]
    failure = result["failure"]
    cancellation = result["cancellation"]

    assert result["channel"] == "ingestion_function"
    assert success["positive_rate"] == 0.5
    assert success["input_resolver"] == "tributo.ingestion"
    assert success["input_engine"] == "tributo.ray_data"
    assert success["ingestion_lifecycle"] == ["closed"]
    assert success["request_body_in_provenance"] is False
    assert failure["status"] == "failed"
    assert failure["failure_category"] == "execution"
    assert failure["ingestion_lifecycle"] == ["cancelled"]
    assert cancellation["cancelled"] is True
    assert cancellation["ingestion_lifecycle"] == ["cancelled"]
    assert result["driver_imported_user_module"] is False
    assert result["driver_imported_sklearn"] is False


@pytest.mark.distributed
def test_sklearn_uses_ray_joblib_workers_in_real_ray_job(
    job_client: JobSubmissionClient,
) -> None:
    """Explicit framework-managed sklearn uses multiple real Ray Workers."""
    job_id = submit_training_job(
        entrypoint=(
            "python tests/integration/algorithm_jobs/portable_execution_job.py"
        ),
        env_vars={"PORTABLE_CHANNEL": "distributed_sklearn"},
        run_id=f"portable-distributed-sklearn-{uuid.uuid4().hex}",
    )
    job_result = wait_for_job(job_client, job_id, timeout=300)
    assert job_result["status"] == JobStatus.SUCCEEDED, job_result["logs"]
    result = _result_from_logs(job_result["logs"])

    assert result["channel"] == "distributed_sklearn"
    assert result["task_count"] == 6
    assert len(result["worker_ids"]) >= 2
    assert result["framework_parallelism"] == 2
    assert result["actual_ray"] == "2.55.1"
    assert result["actual_sklearn"]
    assert result["driver_imported_user_module"] is False
    assert result["driver_imported_sklearn"] is False


@pytest.mark.distributed
def test_user_function_shards_and_reduces_in_real_ray_job(
    job_client: JobSubmissionClient,
) -> None:
    """Custom user code covers production Ray input once across two ranks."""
    job_id = submit_training_job(
        entrypoint=(
            "python tests/integration/algorithm_jobs/portable_execution_job.py"
        ),
        env_vars={"PORTABLE_CHANNEL": "distributed_function"},
        run_id=f"portable-distributed-function-{uuid.uuid4().hex}",
    )
    job_result = wait_for_job(job_client, job_id, timeout=300)
    assert job_result["status"] == JobStatus.SUCCEEDED, job_result["logs"]
    result = _result_from_logs(job_result["logs"])

    expected_values = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
    assert result["channel"] == "distributed_function"
    assert result["row_count"] == 8
    assert result["positive_rate"] == 0.5
    assert result["ranks"] == [0, 1]
    assert len(set(result["worker_ids"])) == 2
    assert sorted(result["shard_values"]) == expected_values
    assert len(result["shard_values"]) == len(set(result["shard_values"]))
    assert [worker["world_rank"] for worker in result["runtime_workers"]] == [0, 1]
    assert all(worker["world_size"] == 2 for worker in result["runtime_workers"])
    assert result["reducer_worker"]["worker_id"]
    assert result["artifact_kinds"] == [
        "report",
        "checkpoint",
        "report",
        "checkpoint",
    ]
    assert result["input_resolver"] == "tributo.ingestion"
    assert result["input_engine"] == "tributo.ray_data"
    assert result["ingestion_lifecycle"] == ["closed"]
    assert result["driver_imported_user_module"] is False
    assert result["driver_imported_sklearn"] is False
