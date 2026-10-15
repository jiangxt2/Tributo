"""Ray Jobs integration contract for Framework Native input coverage."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL
from tributo.training import wait_for_job

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def job_client() -> Iterator[JobSubmissionClient]:
    """Connect to the owned Docker Ray cluster through its Jobs API."""
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


def _object_from_logs(logs: str, marker: str) -> dict[str, Any]:
    marker_offset = logs.find(marker)
    if marker_offset < 0:
        raise AssertionError(f"{marker.strip()} line not found in job logs")
    value, _ = json.JSONDecoder().raw_decode(
        logs[marker_offset + len(marker) :].lstrip()
    )
    return cast(dict[str, Any], value)


def test_framework_native_roles_are_counted_on_ray_cluster(
    job_client: JobSubmissionClient,
) -> None:
    if os.environ.get("TRIBUTO_DOCKER_FRAMEWORK_NATIVE_COVERAGE") != "1":
        pytest.fail(
            "Framework Native coverage Gate must run in its owned Docker Ray cluster"
        )
    core_wheel_value = os.environ.get("TRIBUTO_CORE_WHEEL")
    if not core_wheel_value:
        pytest.fail("Framework Native coverage Gate requires the Core wheel")
    core_wheel = Path(core_wheel_value)
    if not core_wheel.is_file() or core_wheel.suffix != ".whl":
        pytest.fail(
            f"Framework Native coverage Core wheel is unavailable: {core_wheel}"
        )

    run_id = f"framework-native-coverage-{uuid.uuid4().hex}"
    runtime_env = {
        "working_dir": "/workspace/tributo-src",
        "py_modules": [str(core_wheel)],
        "env_vars": {
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "TRIBUTO_RUN_ID": run_id,
        },
    }
    job_id = job_client.submit_job(
        entrypoint="python tests/training/jobs/framework_native_input_coverage_job.py",
        runtime_env=runtime_env,
        submission_id=run_id,
    )
    result = wait_for_job(job_client, job_id, timeout=1800)
    message = job_client.get_job_info(job_id).message or ""
    logs = str(result["logs"])
    assert result["status"] == JobStatus.SUCCEEDED, f"{message}\n{logs}"

    evidence = _object_from_logs(logs, "FRAMEWORK_NATIVE_COVERAGE_RESULT: ")
    assert evidence["cluster_nodes"] >= 2
    assert evidence["canonical_rows"] == {"train": 5, "validation": 5}
    workers = evidence["workers"]
    assert len(workers) == 2
    assert sorted(worker["input_rows"]["train"] for worker in workers) == [2, 3]
    assert [worker["input_rows"]["validation"] for worker in workers] == [5, 5]
    training_identities = [
        identity for worker in workers for identity in worker["train_row_ids"]
    ]
    assert sorted(training_identities) == list(range(5))
    assert len(set(training_identities)) == 5
    assert all(
        sorted(worker["validation_row_ids"]) == list(range(5)) for worker in workers
    )
