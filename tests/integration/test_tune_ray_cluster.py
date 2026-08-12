"""Docker Ray cluster Gate for Tune trial correctness."""

from __future__ import annotations

import json
import math
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


def _run_tune_job() -> dict[str, Any]:
    """Submit the Tune Gate through Ray Jobs and return its result payload."""
    if os.environ.get("TRIBUTO_DOCKER_TUNE_IT") != "1":
        pytest.fail("Tune IT must run inside its isolated Docker Ray project")
    project_root = Path(__file__).parents[2]
    dashboard_url = os.environ.get("TRIBUTO_RAY_DASHBOARD_URL", DEFAULT_DASHBOARD_URL)
    run_id = f"tune-trial-correctness-{uuid.uuid4().hex}"
    job_id = submit_training_job(
        "/opt/tributo/.venv/bin/python "
        "tests/integration/jobs/tune_trial_correctness_job.py",
        dashboard_url=dashboard_url,
        project_root=project_root,
        run_id=run_id,
        env_vars={
            "PYTHONPATH": "/opt/tributo/.venv/lib/python3.12/site-packages",
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            # TensorBoard logging is outside this Gate and imports optional Torch.
            "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS": "1",
            **load_it_component_versions(),
        },
    )
    result = wait_for_job(
        JobSubmissionClient(dashboard_url),
        job_id,
        timeout=900,
        poll_interval=2,
    )
    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    return _result_from_logs(result["logs"])


def test_tune_trials_are_fit_only_and_isolated_through_ray_jobs_api() -> None:
    payload = _run_tune_job()
    print(f"TUNE_GATE_RESULT: {json.dumps(payload, sort_keys=True)}")

    assert payload["status"] == "succeeded"
    assert payload["num_trials"] == 2
    assert payload["num_errors"] == 0
    assert len(payload["target_values"]) == 2
    assert all(math.isfinite(value) for value in payload["target_values"])
    assert set(payload["best_params"]) == {"model.max_depth"}
    assert len(set(payload["result_paths"])) == 2
    assert len(set(payload["checkpoint_paths"])) == 2
    assert len(set(payload["inner_storage_roots"])) == 2
    assert all(path.startswith("trials/") for path in payload["inner_storage_roots"])
    assert payload["bundle_manifests"] == 0
    assert payload["summary_exists"] is True
    assert payload["alive_nodes"] >= 4
    versions = load_it_component_versions()
    assert payload["python_version"] == versions["PYTHON_VERSION"]
    assert payload["versions"] == {
        "ray": versions["RAY_VERSION"],
        "xgboost": versions["XGBOOST_VERSION"],
    }
