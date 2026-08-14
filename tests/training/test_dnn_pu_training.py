"""Formal distributed algorithm integration tests executed through Ray Jobs."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo.training import wait_for_job

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


def _result_from_logs(logs: str) -> list[dict[str, Any]]:
    for line in logs.splitlines():
        if line.startswith("RESULT: "):
            return cast(
                list[dict[str, Any]],
                json.loads(line.removeprefix("RESULT: ")),
            )
    raise AssertionError(f"RESULT line not found in job logs:\n{logs}")


def _plugin_wheel() -> Path:
    configured = os.environ.get("TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL")
    if not configured:
        pytest.fail("distributed algorithm IT requires a third-party plugin wheel")
    wheel = Path(configured)
    if not wheel.is_file() or wheel.suffix != ".whl":
        pytest.fail(f"third-party plugin wheel is unavailable: {wheel}")
    return wheel


def _submit_gate_job(
    job_client: JobSubmissionClient,
    *,
    mode: str,
    root: Path,
    plugin_wheel: Path,
) -> dict[str, Any]:
    submission_id = f"distributed-algorithm-{mode}-{uuid.uuid4().hex}"
    runtime_env = build_runtime_env(
        env_vars={
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "TRIBUTO_DISTRIBUTED_GATE_PROFILE": mode,
            "TRIBUTO_DISTRIBUTED_GATE_ROOT": str(root),
            "TRIBUTO_RUN_ID": submission_id,
            "TRIBUTO_ATTEMPT_ID": "attempt-1",
        }
    )
    py_modules = runtime_env.get("py_modules")
    if not isinstance(py_modules, list):
        raise AssertionError("Ray runtime_env did not expose py_modules")
    runtime_env["py_modules"] = [*py_modules, str(plugin_wheel)]
    job_id = job_client.submit_job(
        entrypoint="python tests/training/jobs/distributed_algorithm_gate_job.py",
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    result = wait_for_job(job_client, job_id, timeout=900)
    job_info = job_client.get_job_info(job_id)
    result["message"] = job_info.message or ""
    if log_path := os.environ.get("TRIBUTO_DISTRIBUTED_GATE_LOG"):
        with Path(log_path).open("a", encoding="utf-8") as stream:
            stream.write(
                f"===== {mode} status={result['status']} job_id={job_id} =====\n"
            )
            if result["message"]:
                stream.write(f"Ray Job message: {result['message']}\n")
            stream.write(str(result["logs"]))
            stream.write("\n")
    return result


def test_formal_distributed_algorithms_complete_on_ray_cluster(
    job_client: JobSubmissionClient,
) -> None:
    """Prove built-ins, a wheel plugin, and required-artifact atomicity."""
    if os.environ.get("TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT") != "1":
        pytest.fail("distributed algorithm IT must run in its owned Docker cluster")
    plugin_wheel = _plugin_wheel()
    gate_root = Path(
        f"/workspace/tributo-work/tributo-distributed-gate-{uuid.uuid4().hex}"
    )
    success_root = gate_root / "success"
    failure_root = gate_root / "required-artifact-failure"
    try:
        job_result = _submit_gate_job(
            job_client,
            mode="docker-distributed",
            root=success_root,
            plugin_wheel=plugin_wheel,
        )
        assert job_result["status"] == JobStatus.SUCCEEDED, (
            "distributed algorithm Gate failed:\n"
            f"{job_result['message']}\n{job_result['logs']}"
        )

        results = _result_from_logs(str(job_result["logs"]))
        assert {result["algorithm"] for result in results} == {
            "dnn",
            "pu",
            "xgboost",
            "multinomial_nb",
            "third_party_multinomial_nb",
        }
        for result in results:
            receipt = result["receipt"]
            assert result["status"] == "succeeded"
            assert receipt["execution_profile"] == "local"
            assert receipt["requested_worker_count"] == 2
            assert receipt["distributed"] is True
            assert receipt["cross_node"] is True
            assert receipt["kubernetes_distributed_supported"] is False
            assert receipt["driver_materialized_training_rows"] == 0
            assert len(receipt["workers"]) == 2
            assert len({worker["node_id"] for worker in receipt["workers"]}) == 2
            assert len({worker["shard_id"] for worker in receipt["workers"]}) == 2
            assert receipt["artifact_ids"]

        failure_result = _submit_gate_job(
            job_client,
            mode="docker-required-artifact-failure",
            root=failure_root,
            plugin_wheel=plugin_wheel,
        )
        assert failure_result["status"] == JobStatus.FAILED
        assert not tuple(failure_root.rglob("manifest.json"))
    finally:
        shutil.rmtree(gate_root, ignore_errors=True)
