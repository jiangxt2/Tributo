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

from tests.training.jobs.official_algorithm_matrix import (
    ALL_ENTRY_POINTS,
    CATEGORY_ENTRY_POINTS,
    parse_entry_point_selection,
)
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


def _plugin_wheel() -> Path:
    configured = os.environ.get("TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL")
    if not configured:
        pytest.fail("distributed algorithm IT requires a third-party plugin wheel")
    wheel = Path(configured)
    if not wheel.is_file() or wheel.suffix != ".whl":
        pytest.fail(f"third-party plugin wheel is unavailable: {wheel}")
    return wheel


def _torch_recipe_plugin_wheel() -> Path:
    configured = os.environ.get("TRIBUTO_TORCH_RECIPE_PLUGIN_WHEEL")
    if not configured:
        pytest.fail("distributed algorithm IT requires a Torch recipe plugin wheel")
    wheel = Path(configured)
    if not wheel.is_file() or wheel.suffix != ".whl":
        pytest.fail(f"Torch recipe plugin wheel is unavailable: {wheel}")
    return wheel


def _official_algorithm_wheels() -> tuple[Path, ...]:
    configured = (
        os.environ.get("TRIBUTO_CORE_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CLASSICAL_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_TIMESERIES_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_REPRESENTATION_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_GRAPH_PYG_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_TABULAR_TORCH_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_RECSYS_TORCH_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_TRANSFORMERS_NLP_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CAUSAL_CORE_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CAUSAL_DISCOVERY_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_MULTISTAGE_TORCH_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_BOOSTING_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CAUSAL_XLEARNER_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CAUSAL_DR_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CAUSAL_DOWHY_WHEEL"),
        os.environ.get("TRIBUTO_OFFICIAL_CATBOOST_WHEEL"),
    )
    if any(not value for value in configured):
        pytest.fail(
            "official algorithm IT requires classical, timeseries, representation, "
            "graph-pyg, tabular-torch, recsys-torch, transformers-nlp, causal-core, causal-discovery, multistage-torch, boosting, causal-xlearner, causal-dr, causal-dowhy, and catboost Wheels"
        )
    wheels = tuple(Path(cast(str, value)) for value in configured)
    if any(not wheel.is_file() or wheel.suffix != ".whl" for wheel in wheels):
        pytest.fail(f"official algorithm Wheel is unavailable: {wheels}")
    return wheels


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


def _submit_torch_recipe_gate_job(
    job_client: JobSubmissionClient,
    *,
    root: Path,
    plugin_wheel: Path,
) -> dict[str, Any]:
    submission_id = f"torch-recipe-{uuid.uuid4().hex}"
    artifact = AlgorithmArtifact(
        source=str(plugin_wheel),
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
    if str(plugin_wheel) not in runtime_env.get("py_modules", []):
        raise AssertionError("Ray runtime_env did not include the Torch recipe Wheel")
    job_id = job_client.submit_job(
        entrypoint="python tests/training/jobs/torch_recipe_gate_job.py",
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


def _submit_official_algorithm_gate_job(
    job_client: JobSubmissionClient,
    *,
    root: Path,
    wheels: tuple[Path, ...],
    entrypoint: str = "python tests/training/jobs/official_algorithm_gate_job.py",
    category: str = "all",
    entry_points: frozenset[str] | None = None,
) -> dict[str, Any]:
    submission_id = f"official-algorithms-{category}-{uuid.uuid4().hex}"
    runtime_env = {
        "working_dir": "/workspace/tributo-src",
        "pip": {
            "packages": [str(wheel) for wheel in wheels],
            "pip_check": False,
            "pip_install_options": [
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
            ],
        },
        "env_vars": {
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "TRIBUTO_OFFICIAL_ALGORITHM_GATE_ROOT": str(root),
            "TRIBUTO_OFFICIAL_ALGORITHM_CATEGORY": category,
            "TRIBUTO_OFFICIAL_ALGORITHM_ENTRY_POINTS": ",".join(
                sorted(entry_points or ())
            ),
            "TRIBUTO_OFFICIAL_DISTRIBUTED_INFERENCE": "1",
            "TRIBUTO_RUN_ID": submission_id,
            "TRIBUTO_ATTEMPT_ID": "attempt-1",
        },
    }
    job_id = job_client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        submission_id=submission_id,
    )
    result = wait_for_job(job_client, job_id, timeout=1800)
    result["message"] = job_client.get_job_info(job_id).message or ""
    if log_path := os.environ.get("TRIBUTO_DISTRIBUTED_GATE_LOG"):
        with Path(log_path).open("a", encoding="utf-8") as stream:
            stream.write(
                f"===== official-algorithms category={category} "
                f"status={result['status']} "
                f"job_id={job_id} =====\n"
            )
            if result["message"]:
                stream.write(f"Ray Job message: {result['message']}\n")
            stream.write(str(result["logs"]))
            stream.write("\n")
    return result


def test_official_algorithm_wheels_complete_on_ray_cluster(
    job_client: JobSubmissionClient,
) -> None:
    """Prove official decomposition and RecipeV2 Wheels train across nodes."""
    if os.environ.get("TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT") != "1":
        pytest.fail("official algorithm IT must run in its owned Docker cluster")
    wheels = _official_algorithm_wheels()
    try:
        selected_entry_points = parse_entry_point_selection(
            os.environ.get("TRIBUTO_OFFICIAL_GATE_ENTRY_POINTS", "")
        )
    except ValueError as exc:
        pytest.fail(str(exc))
    requested_categories = os.environ.get("TRIBUTO_OFFICIAL_GATE_CATEGORIES", "")
    if requested_categories:
        categories = tuple(
            value.strip() for value in requested_categories.split(",") if value.strip()
        )
        if not categories or len(set(categories)) != len(categories):
            pytest.fail("official algorithm Gate categories must be unique names")
        unknown = sorted(set(categories) - set(CATEGORY_ENTRY_POINTS))
        if unknown:
            pytest.fail(f"unknown official algorithm Gate categories: {unknown}")
    else:
        categories = tuple(CATEGORY_ENTRY_POINTS)
    if selected_entry_points is not None:
        covered_entry_points = frozenset(
            entry_point
            for category in categories
            for entry_point in CATEGORY_ENTRY_POINTS[category]
        )
        uncovered_entry_points = sorted(selected_entry_points - covered_entry_points)
        if uncovered_entry_points:
            pytest.fail(
                "selected official algorithm entry points are outside the selected "
                f"categories: {uncovered_entry_points}"
            )
        categories = tuple(
            category
            for category in categories
            if selected_entry_points.intersection(CATEGORY_ENTRY_POINTS[category])
        )
    records: list[dict[str, Any]] = []
    category_logs: dict[str, str] = {}
    for category in categories:
        expected_entry_points = tuple(
            entry_point
            for entry_point in CATEGORY_ENTRY_POINTS[category]
            if selected_entry_points is None or entry_point in selected_entry_points
        )
        gate_root = Path(
            "/workspace/tributo-work/"
            f"tributo-official-algorithm-gate-{category}-{uuid.uuid4().hex}"
        )
        result = _submit_official_algorithm_gate_job(
            job_client,
            root=gate_root,
            wheels=wheels,
            category=category,
            entry_points=selected_entry_points,
        )
        assert result["status"] == JobStatus.SUCCEEDED, (
            f"official {category} Gate failed:\n{result['message']}\n{result['logs']}"
        )
        logs = str(result["logs"])
        category_logs[category] = logs
        category_records = _result_from_logs(logs)
        assert {record["entry_point"] for record in category_records} == set(
            expected_entry_points
        )
        inference_result = _object_from_logs(logs, "INFERENCE_RESULT: ")
        assert inference_result == {
            "all_distributed": True,
            "record_count": len(expected_entry_points),
        }
        records.extend(category_records)
    expected_selected = selected_entry_points or frozenset(
        entry_point
        for category in categories
        for entry_point in CATEGORY_ENTRY_POINTS[category]
    )
    assert len(records) == len(expected_selected)
    assert {record["entry_point"] for record in records} == expected_selected
    if expected_selected == ALL_ENTRY_POINTS:
        assert expected_selected == ALL_ENTRY_POINTS
    if "classical" in category_logs:
        classical_logs = category_logs["classical"]
        for marker in (
            "TUNE_RESULT: ",
            "BASELINE_RESULT: ",
            "RECOVERY_RESULT: ",
            "FAILURE_RESULT: ",
        ):
            assert _object_from_logs(classical_logs, marker) == {"skipped": True}
    expected_strategies = {
        "ray_joblib_estimator",
        "ray_parallel_ensemble",
        "ray_iterative_optimization",
        "ray_map_reduce",
        "ray_train_recipe_v2",
        "framework_native",
    }
    actual_strategies = {record["receipt"]["strategy"] for record in records}
    assert actual_strategies
    assert actual_strategies <= expected_strategies
    if expected_selected == ALL_ENTRY_POINTS:
        assert actual_strategies == expected_strategies
    for record in records:
        receipt = record["receipt"]
        assert record["status"] == "succeeded"
        assert receipt["distributed"] is True
        assert receipt["driver_materialized_training_rows"] == 0
        assert len(receipt["workers"]) == 2
        assert receipt["cluster_distributed"] is True
        assert len({worker["node_id"] for worker in receipt["workers"]}) == 2
        distributed_inference = record["distributed_inference"]
        assert distributed_inference["status"] == "succeeded"
        assert distributed_inference["row_count"] == 16
        assert distributed_inference["node_count"] == 2
        assert len(distributed_inference["manifest_sha256"]) == 64
        assert len(distributed_inference["result_id"]) == 64


def test_priority_algorithm_wheels_complete_on_ray_cluster(
    job_client: JobSubmissionClient,
) -> None:
    """Prove the selected XGBoost, RF, LR, and causal paths cross nodes."""
    if os.environ.get("TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT") != "1":
        pytest.fail("priority algorithm IT must run in its owned Docker cluster")
    wheels = _official_algorithm_wheels()
    gate_root = Path(
        f"/workspace/tributo-work/tributo-priority-algorithm-gate-{uuid.uuid4().hex}"
    )
    result = _submit_official_algorithm_gate_job(
        job_client,
        root=gate_root,
        wheels=wheels,
        entrypoint="python tests/training/jobs/priority_algorithm_gate_job.py",
    )
    assert result["status"] == JobStatus.SUCCEEDED, (
        f"priority algorithm Gate failed:\n{result['message']}\n{result['logs']}"
    )
    records = _result_from_logs(str(result["logs"]))
    assert {record["algorithm"] for record in records} == {
        "random_forest",
        "logistic_regression",
        "xgboost",
        "linear_dml_ate",
        "doubly_robust_ate",
        "x_learner",
    }
    assert len(records) == 6
    assert all(record["onnx_exported"] is True for record in records)
    assert all(
        record["inference_roundtrip"] is True
        for record in records
        if record["algorithm"]
        in {"random_forest", "logistic_regression", "xgboost", "x_learner"}
    )
    for record in records:
        receipt = record["receipt"]
        assert record["status"] == "succeeded"
        assert receipt["distributed"] is True
        assert receipt["cluster_distributed"] is True
        assert len({worker["node_id"] for worker in receipt["workers"]}) == 2
    baseline = _object_from_logs(str(result["logs"]), "BASELINE_RESULT: ")
    assert baseline["random_forest_exact"] is True
    assert baseline["logistic_prediction_equivalent"] is True
    recovery = _object_from_logs(str(result["logs"]), "RECOVERY_RESULT: ")
    assert recovery["ensemble_resumed"] is True
    assert recovery["iterative_resumed"] is True
    assert recovery["ensemble_corruption_rejected"] is True
    assert recovery["iterative_corruption_rejected"] is True
    inference = _object_from_logs(str(result["logs"]), "INFERENCE_RESULT: ")
    assert inference["node_count"] == 2


def test_out_of_tree_torch_recipe_completes_on_ray_cluster(
    job_client: JobSubmissionClient,
) -> None:
    """Prove the low-code recipe Wheel, uneven shards, checkpoint, and Bundle."""
    if os.environ.get("TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT") != "1":
        pytest.fail("distributed algorithm IT must run in its owned Docker cluster")
    gate_root = Path(f"/workspace/tributo-work/tributo-recipe-{uuid.uuid4().hex}")
    try:
        job_result = _submit_torch_recipe_gate_job(
            job_client,
            root=gate_root,
            plugin_wheel=_torch_recipe_plugin_wheel(),
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
