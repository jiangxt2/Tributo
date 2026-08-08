"""Docker Ray cluster gate for the model-export architecture."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL
from tributo.training import submit_training_job, wait_for_job

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _result_from_logs(logs: str) -> dict[str, Any]:
    for line in logs.splitlines():
        if line.startswith("RESULT: "):
            result: dict[str, Any] = json.loads(line.removeprefix("RESULT: "))
            return result
    raise AssertionError(f"RESULT line not found in Ray Job logs:\n{logs}")


def test_xgboost_bundle_runtime_through_ray_jobs_api() -> None:
    """Run Linux XGBoost→ONNX+UBJ+JSON→S3 Bundle→Runtime in Docker Ray."""
    project_root = Path(__file__).parents[2]
    dashboard_url = os.environ.get("TRIBUTO_RAY_DASHBOARD_URL", DEFAULT_DASHBOARD_URL)
    run_id = f"model-export-architecture-{uuid.uuid4().hex}"
    job_id = submit_training_job(
        "python tests/integration/jobs/model_export_architecture_job.py",
        dashboard_url=dashboard_url,
        project_root=project_root,
        run_id=run_id,
        env_vars={
            "S3_ENDPOINT": os.environ.get(
                "TRIBUTO_RAY_MINIO_ENDPOINT", "http://minio:9000"
            ),
            "AWS_ACCESS_KEY_ID": os.environ.get(
                "TRIBUTO_RAY_MINIO_ACCESS_KEY", "minioadmin"
            ),
            "AWS_SECRET_ACCESS_KEY": os.environ.get(
                "TRIBUTO_RAY_MINIO_SECRET_KEY", "minioadmin123"
            ),
            "AWS_DEFAULT_REGION": "us-east-1",
            "TRIBUTO_STORAGE_PROFILE_TEST": json.dumps({"path_style": True}),
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        },
    )
    result = wait_for_job(
        JobSubmissionClient(dashboard_url),
        job_id,
        timeout=300,
        poll_interval=2,
    )
    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    payload = _result_from_logs(result["logs"])
    assert payload["status"] == "succeeded"
    assert payload["artifact_kinds"] == ["model"]
    assert payload["formats"] == {
        "json-model": "xgboost-json",
        "native-model": "ubj",
        "onnx-model": "onnx",
    }
    assert len(payload["manifest_sha256"]) == 64
    assert len(payload["event_id"]) == 64
