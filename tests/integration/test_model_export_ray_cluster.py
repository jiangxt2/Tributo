"""Docker Ray cluster gate for the model-export architecture."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tests.support.it_versions import load_it_component_versions
from tributo._common import DEFAULT_DASHBOARD_URL
from tributo.training import submit_training_job, wait_for_job

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _result_from_logs(logs: str) -> dict[str, Any]:
    for line in logs.splitlines():
        if line.startswith("RESULT: "):
            result: dict[str, Any] = json.loads(line.removeprefix("RESULT: "))
            return result
    raise AssertionError(f"RESULT line not found in Ray Job logs:\n{logs}")


def _run_model_export_job() -> dict[str, Any]:
    """Submit the golden path through Ray Jobs and return its result payload."""
    if os.environ.get("TRIBUTO_DOCKER_MODEL_EXPORT_IT") != "1":
        pytest.fail("model-export IT must run inside its isolated Docker Ray project")
    project_root = Path(__file__).parents[2]
    dashboard_url = os.environ.get("TRIBUTO_RAY_DASHBOARD_URL", DEFAULT_DASHBOARD_URL)
    run_id = f"model-export-architecture-{uuid.uuid4().hex}"
    job_id = submit_training_job(
        "/opt/tributo/.venv/bin/python "
        "tests/integration/jobs/model_export_architecture_job.py",
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
            "MLFLOW_TRACKING_URI": os.environ.get(
                "MLFLOW_TRACKING_URI", "http://mlflow:5000"
            ),
            "PYTHONPATH": "/opt/tributo/.venv/lib/python3.12/site-packages",
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            **load_it_component_versions(),
        },
    )
    result = wait_for_job(
        JobSubmissionClient(dashboard_url),
        job_id,
        timeout=600,
        poll_interval=2,
    )
    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    payload = _result_from_logs(result["logs"])
    return payload


def test_xgboost_bundle_runtime_through_ray_jobs_api() -> None:
    """Run the Docker Ray golden path exclusively through the Ray Jobs API."""
    payload = _run_model_export_job()

    assert payload["status"] == "succeeded"
    assert payload["artifact_kinds"] == ["model"]
    assert payload["formats"] == {
        "native": "ubj",
        "onnx-model": "onnx",
    }
    assert len(payload["manifest_sha256"]) == 64
    assert payload["batch_rows"] == 2
    assert payload["http_rows"] == 2
    assert payload["mlflow_runs"] == 1
    assert payload["model_versions_created"] == 0
    assert payload["python_version"] == load_it_component_versions()["PYTHON_VERSION"]
    assert payload["versions"] == {
        package: load_it_component_versions()[key]
        for package, key in {
            "boto3": "BOTO3_VERSION",
            "botocore": "BOTOCORE_VERSION",
            "ray": "RAY_VERSION",
            "mlflow": "MLFLOW_VERSION",
            "xgboost": "XGBOOST_VERSION",
            "onnx": "ONNX_VERSION",
            "onnxruntime": "ONNXRUNTIME_VERSION",
            "onnxmltools": "ONNXMLTOOLS_VERSION",
            "torch": "TORCH_VERSION",
            "transformers": "TRANSFORMERS_VERSION",
            "pyarrow": "PYARROW_VERSION",
            "pandas": "PANDAS_VERSION",
        }.items()
    }
    assert payload["alive_nodes"] >= 2
