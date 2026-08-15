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
from tributo.algorithms import AlgorithmArtifact, ImageProfile
from tributo.algorithms.api import EnvironmentSpec
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
    marker = "RESULT: "
    decoder = json.JSONDecoder()
    offset = 0
    while (marker_offset := logs.find(marker, offset)) >= 0:
        payload = logs[marker_offset + len(marker) :].lstrip()
        try:
            result, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            offset = marker_offset + len(marker)
            continue
        return cast(list[dict[str, Any]], result)
    raise AssertionError(f"RESULT line not found in job logs:\n{logs}")


def _plugin_wheel() -> Path:
    configured = os.environ.get("TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL")
    if not configured:
        pytest.fail("distributed algorithm IT requires a third-party plugin wheel")
    wheel = Path(configured)
    if not wheel.is_file() or wheel.suffix != ".whl":
        pytest.fail(f"third-party plugin wheel is unavailable: {wheel}")
    return wheel


def _algorithm_image_profile() -> ImageProfile:
    digest = os.environ.get("TRIBUTO_ALGORITHM_IMAGE_DIGEST", "a" * 64)
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return ImageProfile(
        profile_id="data-ingestion.cpu.v1",
        image_uri=os.environ.get("TRIBUTO_IT_RUNTIME_IMAGE", "ray:2.55.1"),
        image_digest=digest,
        wheel_tags=("py3-none-any",),
        installed_distributions={"pip": "24.3.1"},
        algorithm_ids=("offline.algorithm",),
        pip_check_baseline=(
            "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform",
        ),
    )


def _submit_gate_job(
    job_client: JobSubmissionClient,
    *,
    mode: str,
    root: Path,
    plugin_wheel: Path,
) -> dict[str, Any]:
    submission_id = f"distributed-algorithm-{mode}-{uuid.uuid4().hex}"
    artifact = AlgorithmArtifact(
        source=str(plugin_wheel),
        package_name="tributo-test-distributed-algorithm",
        package_version="0.1.0",
        plugin_names=("third_party_mean_regressor",),
        wheel_tags=("py3-none-any",),
    )
    environment = EnvironmentSpec(
        environment_id="example.mean_regression.v1",
        dependencies=("tributo-test-distributed-algorithm==0.1.0",),
    )
    runtime_env = build_runtime_env(
        env_vars={
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "TRIBUTO_DISTRIBUTED_GATE_PROFILE": mode,
            "TRIBUTO_DISTRIBUTED_GATE_ROOT": str(root),
            "TRIBUTO_RUN_ID": submission_id,
            "TRIBUTO_ATTEMPT_ID": "attempt-1",
        },
        algorithm_artifact=artifact,
        image_profile=_algorithm_image_profile(),
        declared_dependencies=environment.dependencies,
    )
    if str(plugin_wheel) not in runtime_env.get("py_modules", []):
        raise AssertionError("Ray runtime_env did not include the algorithm Wheel")
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
            "third_party_mean_regressor",
        }
        assert {
            (result["algorithm"], result["worker_count"])
            for result in results
            if result["algorithm"] == "third_party_mean_regressor"
        } == {
            ("third_party_mean_regressor", 1),
            ("third_party_mean_regressor", 2),
        }
        for result in results:
            receipt = result["receipt"]
            assert result["status"] == "succeeded"
            assert receipt["execution_profile"] == "local"
            assert receipt["requested_worker_count"] == result["worker_count"]
            assert receipt["distributed"] is (result["worker_count"] >= 2)
            assert receipt["cross_node"] is (result["worker_count"] >= 2)
            assert receipt["kubernetes_distributed_supported"] is False
            assert receipt["driver_materialized_training_rows"] == 0
            assert len(receipt["workers"]) == result["worker_count"]
            assert (
                len({worker["node_id"] for worker in receipt["workers"]})
                == (result["worker_count"])
            )
            assert (
                len({worker["shard_id"] for worker in receipt["workers"]})
                == (result["worker_count"])
            )
            if result["algorithm"] == "third_party_mean_regressor":
                assert receipt["result_policy"] == "fit_only"
                assert receipt["artifact_ids"] == []
            else:
                assert receipt["result_policy"] == "bundle_required"
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


def _offline_bundle() -> Path:
    configured = os.environ.get("TRIBUTO_OFFLINE_ALGORITHM_BUNDLE")
    if not configured:
        pytest.fail("algorithm distribution IT requires an offline Bundle")
    bundle = Path(configured)
    if not (bundle / "manifest.json").is_file():
        pytest.fail(f"offline algorithm Bundle is unavailable: {bundle}")
    return bundle


def _offline_job_entrypoint() -> str:
    code = (
        "import importlib.metadata as metadata, json, ray; "
        "from tributo_test_offline_algorithm import verify_runtime; "
        "ray.init(); "
        "driver = verify_runtime(); "
        "workers = ray.get([ray.remote(verify_runtime).remote() for _ in range(2)]); "
        "assert metadata.version('tributo-test-offline-dependency') == '1.0.0'; "
        "assert all(item['dependency_marker'] == 'offline-dependency-installed' for item in workers); "
        "print('ALGORITHM_DISTRIBUTION_RESULT: ' + json.dumps({'driver': driver, 'workers': workers}, sort_keys=True))"
    )
    return f"python -c {json.dumps(code)}"


def test_offline_wheelhouse_installs_unique_dependency_on_driver_and_workers(
    job_client: JobSubmissionClient,
) -> None:
    """Prove repeated offline Bundle submissions are reproducible on all nodes."""
    if os.environ.get("TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT") != "1":
        pytest.fail("algorithm distribution IT must run in its owned Docker cluster")
    bundle = _offline_bundle()
    profile = _algorithm_image_profile()
    artifact = AlgorithmArtifact(
        source=str(bundle),
        mode="offline_wheelhouse",
    )
    results: list[dict[str, Any]] = []
    for attempt in (1, 2):
        submission_id = f"offline-algorithm-{attempt}-{uuid.uuid4().hex}"
        runtime_env = build_runtime_env(
            env_vars={
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                "TRIBUTO_RUN_ID": submission_id,
                "TRIBUTO_ATTEMPT_ID": f"attempt-{attempt}",
            },
            algorithm_artifact=artifact,
            image_profile=profile,
        )
        assert runtime_env["pip"]["pip_check"] is False
        job_id = job_client.submit_job(
            entrypoint=_offline_job_entrypoint(),
            runtime_env=runtime_env,
            submission_id=submission_id,
        )
        result = wait_for_job(job_client, job_id, timeout=900)
        result["message"] = job_client.get_job_info(job_id).message or ""
        results.append(result)
        assert result["status"] == JobStatus.SUCCEEDED, (
            f"offline algorithm Job failed on attempt {attempt}: "
            f"{result['message']}\n{result['logs']}"
        )
        logs = str(result["logs"])
        assert "ALGORITHM_DISTRIBUTION_RESULT:" in logs
        assert '"distribution_mode": "offline_wheelhouse"' in logs
        assert '"dependency_version": "1.0.0"' in logs
        assert "offline-dependency-installed" in logs

    assert len(results) == 2


def test_remote_offline_wheelhouse_archive_installs_on_driver_and_workers(
    job_client: JobSubmissionClient,
) -> None:
    """Prove Ray fetches an attested ZIP Bundle from the internal S3 endpoint."""
    if os.environ.get("TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT") != "1":
        pytest.fail("algorithm distribution IT must run in its owned Docker cluster")
    bundle = _offline_bundle()
    remote_uri = os.environ.get("TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_URI")
    archive_sha256 = os.environ.get("TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_SHA256")
    if not remote_uri or not archive_sha256:
        pytest.fail("remote offline Bundle IT requires an S3 URI and archive digest")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    artifact = AlgorithmArtifact(
        source=remote_uri,
        mode="offline_wheelhouse",
        sha256=archive_sha256,
        package_name="tributo-test-offline-algorithm",
        package_version="1.0.0",
        wheel_tags=("py3-none-any",),
        manifest=manifest,
    )
    submission_id = f"offline-algorithm-remote-{uuid.uuid4().hex}"
    runtime_env = build_runtime_env(
        env_vars={
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "TRIBUTO_RUN_ID": submission_id,
            "TRIBUTO_ATTEMPT_ID": "attempt-remote",
        },
        algorithm_artifact=artifact,
        image_profile=_algorithm_image_profile(),
    )
    assert runtime_env["working_dir"] == remote_uri
    assert runtime_env["pip"]["packages"] == [
        "-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/requirements.lock"
    ]
    job_id = job_client.submit_job(
        entrypoint=_offline_job_entrypoint(),
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    result = wait_for_job(job_client, job_id, timeout=900)
    message = job_client.get_job_info(job_id).message or ""
    assert result["status"] == JobStatus.SUCCEEDED, (
        f"remote offline algorithm Job failed: {message}\n{result['logs']}"
    )
    logs = str(result["logs"])
    assert "ALGORITHM_DISTRIBUTION_RESULT:" in logs
    assert '"distribution_mode": "offline_wheelhouse"' in logs
    assert '"dependency_version": "1.0.0"' in logs
    assert "offline-dependency-installed" in logs
