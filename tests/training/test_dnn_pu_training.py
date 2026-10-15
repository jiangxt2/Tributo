"""Core-owned fixture plugin integration tests executed through Ray Jobs."""

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

from tests.support.runtime_versions import RAY_VERSION
from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo.algorithms import AlgorithmArtifact, ImageProfile
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
    for line in logs.splitlines():
        if not line.startswith(marker):
            continue
        try:
            result, _ = decoder.raw_decode(line[len(marker) :].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(result, list):
            return cast(list[dict[str, Any]], result)
    raise AssertionError(f"RESULT line not found in job logs:\n{logs}")


def _object_from_logs(logs: str, marker: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    marker_offset = logs.find(marker)
    if marker_offset < 0:
        raise AssertionError(f"{marker.strip()} line not found in job logs")
    value, _ = decoder.raw_decode(logs[marker_offset + len(marker) :].lstrip())
    return cast(dict[str, Any], value)


def _torch_recipe_plugin_bundle() -> Path:
    configured = os.environ.get("TRIBUTO_TORCH_RECIPE_PLUGIN_BUNDLE")
    if not configured:
        pytest.fail("fixture plugin IT requires a Torch recipe Bundle")
    bundle = Path(configured)
    if not (bundle / "manifest.json").is_file():
        pytest.fail(f"Torch recipe plugin Bundle is unavailable: {bundle}")
    return bundle


def _algorithm_image_profile() -> ImageProfile:
    digest = os.environ.get("TRIBUTO_ALGORITHM_IMAGE_DIGEST", "a" * 64)
    if digest.startswith("sha256:"):
        digest = digest.removeprefix("sha256:")
    return ImageProfile(
        profile_id="data-ingestion.cpu.v1",
        image_uri=os.environ.get("TRIBUTO_IT_RUNTIME_IMAGE", f"ray:{RAY_VERSION}"),
        image_digest=digest,
        wheel_tags=("py3-none-any",),
        installed_distributions={"pip": "24.3.1"},
        algorithm_ids=("offline.algorithm", "torch_recipe.fixture"),
        pip_check_baseline=(
            "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform",
        ),
    )


def _submit_torch_recipe_gate_job(
    job_client: JobSubmissionClient,
    *,
    root: Path,
    plugin_bundle: Path,
) -> dict[str, Any]:
    submission_id = f"torch-recipe-{uuid.uuid4().hex}"
    artifact = AlgorithmArtifact(
        source=str(plugin_bundle),
        mode="offline_wheelhouse",
        package_name="tributo-test-torch-recipe-algorithm",
        package_version="0.1.0",
        plugin_names=("third_party_binary_linear",),
        wheel_tags=("py3-none-any",),
    )
    runtime_env = build_runtime_env(
        env_vars={
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "TRIBUTO_DISTRIBUTED_GATE_PROFILE": "torch-recipe",
            "TRIBUTO_DISTRIBUTED_GATE_ROOT": str(root),
            "TRIBUTO_RUN_ID": submission_id,
            "TRIBUTO_ATTEMPT_ID": "attempt-1",
        },
        algorithm_artifact=artifact,
        image_profile=_algorithm_image_profile(),
        declared_dependencies=("tributo-test-torch-recipe-algorithm==0.1.0",),
    )
    if runtime_env.get("working_dir") != str(plugin_bundle):
        raise AssertionError("Ray runtime_env did not include the Torch recipe Bundle")
    if "pip" not in runtime_env:
        raise AssertionError("Ray runtime_env did not install the Torch recipe Bundle")
    entrypoint_code = (
        "from tests.training.jobs.torch_recipe_gate_job import main; "
        "raise SystemExit(main())"
    )
    job_id = job_client.submit_job(
        entrypoint=f"python -c {json.dumps(entrypoint_code)}",
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    result = wait_for_job(job_client, job_id, timeout=900)
    result["message"] = job_client.get_job_info(job_id).message or ""
    if log_path := os.environ.get("TRIBUTO_DISTRIBUTED_GATE_LOG"):
        with Path(log_path).open("a", encoding="utf-8") as stream:
            stream.write(
                f"===== torch-recipe status={result['status']} job_id={job_id} =====\n"
            )
            if result["message"]:
                stream.write(f"Ray Job message: {result['message']}\n")
            stream.write(str(result["logs"]))
            stream.write("\n")
    return result


def test_out_of_tree_torch_recipe_completes_on_ray_cluster(
    job_client: JobSubmissionClient,
) -> None:
    """Prove an out-of-tree recipe Bundle, uneven shards, and model export."""
    if os.environ.get("TRIBUTO_DOCKER_ALGORITHM_CONTRACT_IT") != "1":
        pytest.fail("fixture plugin IT must run in its owned Docker Ray Jobs cluster")
    gate_root = Path(f"/workspace/tributo-work/tributo-recipe-{uuid.uuid4().hex}")
    try:
        job_result = _submit_torch_recipe_gate_job(
            job_client,
            root=gate_root,
            plugin_bundle=_torch_recipe_plugin_bundle(),
        )
        assert job_result["status"] == JobStatus.SUCCEEDED, (
            f"Torch recipe Gate failed:\n{job_result['message']}\n{job_result['logs']}"
        )
        results = _result_from_logs(str(job_result["logs"]))
        assert len(results) == 1
        result = results[0]
        receipt = result["receipt"]
        assert result["algorithm"] == "third_party_binary_linear"
        assert result["worker_count"] == 2
        assert receipt["execution_profile"] == "cluster"
        assert receipt["runtime_owned"] is False
        assert receipt["distributed"] is True
        assert receipt["cross_node"] is True
        assert receipt["cluster_distributed"] is True
        assert receipt["result_policy"] == "bundle_required"
        assert receipt["artifact_ids"]
        assert sum(worker["input_rows"]["train"] for worker in receipt["workers"]) == 65
        assert all(
            worker["batch_count"] <= worker["collective_steps"]
            for worker in receipt["workers"]
        )
    finally:
        shutil.rmtree(gate_root, ignore_errors=True)


def _offline_bundle() -> Path:
    configured = os.environ.get("TRIBUTO_OFFLINE_ALGORITHM_BUNDLE")
    if not configured:
        pytest.fail("fixture plugin IT requires an offline Bundle")
    bundle = Path(configured)
    if not (bundle / "manifest.json").is_file():
        pytest.fail(f"offline algorithm Bundle is unavailable: {bundle}")
    return bundle


def _offline_job_entrypoint() -> str:
    code = (
        "import importlib.metadata as metadata, importlib.util, json, os, ray, sys; "
        "module_spec = importlib.util.find_spec('tributo_test_offline_algorithm'); "
        "distributions = sorted(distribution.metadata.get('Name', '') "
        "for distribution in metadata.distributions() "
        "if distribution.metadata.get('Name', '').lower().startswith("
        "'tributo-test-offline-')); "
        "print('ALGORITHM_RUNTIME_ENV_PROBE: ' + json.dumps({"
        "'python': sys.executable, 'cwd': os.getcwd(), 'sys_path': sys.path, "
        "'module_origin': module_spec.origin if module_spec else None, "
        "'distributions': distributions, "
        "'mode': os.environ.get('TRIBUTO_ALGORITHM_DISTRIBUTION_MODE')"
        "}, sort_keys=True)); "
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
    if os.environ.get("TRIBUTO_DOCKER_ALGORITHM_CONTRACT_IT") != "1":
        pytest.fail("fixture plugin IT must run in its owned Docker Ray Jobs cluster")
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
    if os.environ.get("TRIBUTO_DOCKER_ALGORITHM_CONTRACT_IT") != "1":
        pytest.fail("fixture plugin IT must run in its owned Docker Ray Jobs cluster")
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
    assert runtime_env["pip"]["packages"][0] == (
        "-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/requirements.lock"
    )
    assert runtime_env["pip"]["packages"][1].startswith(
        "# tributo-runtime-env-cache-key="
    )
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
