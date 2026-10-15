"""Ray Jobs contract for Core-owned partitioned graph input."""

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


def test_core_graph_partition_contract_runs_on_two_workers(
    job_client: JobSubmissionClient,
) -> None:
    if os.environ.get("TRIBUTO_DOCKER_GRAPH_CORE_GATE") != "1":
        pytest.fail("Core graph Gate must run in its owned Docker Ray cluster")
    core_wheel_value = os.environ.get("TRIBUTO_CORE_WHEEL")
    if not core_wheel_value:
        pytest.fail("graph Core Gate requires the fixed Tributo Core Wheel")
    core_wheel = Path(core_wheel_value)
    if not core_wheel.is_file() or core_wheel.suffix != ".whl":
        pytest.fail(f"graph Core Gate Wheel is unavailable: {core_wheel}")

    run_id = f"graph-core-{uuid.uuid4().hex}"
    gate_root = Path(f"/workspace/tributo-work/{run_id}")
    runtime_env = {
        "working_dir": "/workspace/tributo-src",
        "py_modules": [str(core_wheel)],
        "env_vars": {
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "TRIBUTO_GRAPH_CORE_GATE_ROOT": str(gate_root),
            "TRIBUTO_RUN_ID": run_id,
            "TRIBUTO_ATTEMPT_ID": "attempt-1",
        },
    }
    job_id = job_client.submit_job(
        entrypoint="python tests/training/jobs/graph_core_gate_job.py",
        runtime_env=runtime_env,
        submission_id=run_id,
    )
    result = wait_for_job(job_client, job_id, timeout=1800)
    message = job_client.get_job_info(job_id).message or ""
    logs = str(result["logs"])
    assert result["status"] == JobStatus.SUCCEEDED, f"{message}\n{logs}"

    evidence = _object_from_logs(logs, "GRAPH_CORE_RESULT: ")
    graph = evidence["torch_evidence"]["graph_partition"]
    assert evidence["distributed"] is True
    assert evidence["cross_node"] is True
    assert evidence["driver_materialized_training_rows"] == 0
    assert graph["partition_count"] == 2
    assert graph["total_nodes"] == 8
    assert graph["total_edges"] == 8

    workers = evidence["torch_evidence"]["workers"]
    assert len(workers) == 2
    assert len({worker["node_id"] for worker in workers}) == 2
    assert all(set(worker["input_rows"]) == {"train"} for worker in workers)
    seed_role = next(
        role for role in evidence["torch_evidence"]["roles"] if role["role"] == "train"
    )
    assert seed_role["observed_rows"] == 8
    assert seed_role["rows_per_rank"] == [4, 4]
    assert sum(worker["graph"]["seed_rows"] for worker in workers) == 18
    assert all(worker["rows_processed"] == 8 for worker in workers)
    assert all(worker["batch_count"] == 4 for worker in workers)
    assert all(worker["graph"]["batch_count"] == 5 for worker in workers)
    assert all(worker["graph"]["touched_partitions"] == [0, 1] for worker in workers)
