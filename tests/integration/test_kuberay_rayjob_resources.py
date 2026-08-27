"""Heavyweight manual KubeRay IT for resource-profile compilation and execution.

Run only through ``scripts/run_kuberay_rayjob_it.sh`` after an explicit request.
This module is not a default development or pre-check test.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import pytest

from tributo.kuberay_submission import (
    KubeRayDeploymentConfig,
    KubeRayJobRequest,
    KubeRayJobSubmission,
    KubeRayJobSubmitter,
    KubeRayResourceProfile,
    KubeRayWorkerResources,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.manual_it]

_BYTES_PER_GIB = 1024**3
_RESULT_MARKER = "RESULT: "
_RESULT_DECODER = json.JSONDecoder()


def _require_it_environment() -> tuple[str, str]:
    if os.environ.get("TRIBUTO_KUBERAY_IT") != "1":
        pytest.fail("KubeRay IT must run through scripts/run_kuberay_rayjob_it.sh")
    namespace = os.environ.get("TRIBUTO_KUBERAY_NAMESPACE")
    image = os.environ.get("TRIBUTO_KUBERAY_RUNTIME_IMAGE")
    if not namespace or not image:
        pytest.fail(
            "TRIBUTO_KUBERAY_NAMESPACE and TRIBUTO_KUBERAY_RUNTIME_IMAGE are required"
        )
    return namespace, image


def _submitter() -> KubeRayJobSubmitter:
    from kubernetes import client, config

    config.load_kube_config()
    return KubeRayJobSubmitter(
        custom_objects_api=client.CustomObjectsApi(),
        core_api=client.CoreV1Api(),
    )


def _request(
    *,
    namespace: str,
    image: str,
    worker_count: int,
    num_cpus: float,
    memory_bytes: int,
) -> KubeRayJobRequest:
    profile = KubeRayResourceProfile(
        worker_count=worker_count,
        resources_per_worker=KubeRayWorkerResources(
            num_cpus=num_cpus,
            memory_bytes=memory_bytes,
        ),
    )
    run_id = f"kuberay-resource-{uuid.uuid4().hex}"
    entrypoint = (
        "python -m tributo.kuberay_xgboost_gate "
        f"--worker-count {worker_count} "
        f"--num-cpus {num_cpus:g} "
        f"--memory-bytes {memory_bytes} "
        f"--storage-key {run_id}"
    )
    return KubeRayJobRequest(
        entrypoint=entrypoint,
        run_id=run_id,
        resource_profile=profile,
        deployment=KubeRayDeploymentConfig(
            namespace=namespace,
            image=image,
            ray_version="2.55.1",
            head_num_cpus=1,
            head_memory_bytes=4 * _BYTES_PER_GIB,
            image_pull_policy="Never",
            pod_env={
                "OMP_NUM_THREADS": "1",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "RAY_memory_usage_threshold": "0.99",
            },
            volumes=(
                {
                    "name": "ray-storage",
                    "hostPath": {
                        "path": "/var/local/tributo-kuberay-shared",
                        "type": "Directory",
                    },
                },
            ),
            volume_mounts=(
                {
                    "name": "ray-storage",
                    "mountPath": "/tmp/tributo-kuberay-shared",
                },
            ),
        ),
    )


def _wait_for_cluster_name(
    submitter: KubeRayJobSubmitter,
    submission: KubeRayJobSubmission,
    *,
    timeout: float = 300,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = submitter.get_status(submission)
        if status.ray_cluster_name:
            return status.ray_cluster_name
        if status.job_deployment_status == "Failed":
            raise AssertionError(status.message or "KubeRay deployment failed")
        time.sleep(2)
    raise AssertionError("KubeRay did not publish a RayCluster name in time")


def _wait_for_result(
    submitter: KubeRayJobSubmitter,
    submission: KubeRayJobSubmission,
    *,
    timeout: float = 60,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        logs = submitter.get_logs(submission)
        for line in reversed(logs.splitlines()):
            marker_index = line.find(_RESULT_MARKER)
            if marker_index >= 0:
                payload = line[marker_index + len(_RESULT_MARKER) :].lstrip()
                try:
                    result, _ = _RESULT_DECODER.raw_decode(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(result, dict):
                    return result
        time.sleep(1)
    raise AssertionError("KubeRay submitter logs contain no XGBoost result")


def _wait_for_deployment_complete(
    submitter: KubeRayJobSubmitter,
    submission: KubeRayJobSubmission,
    *,
    timeout: float = 300,
) -> None:
    """Wait for KubeRay's post-success RayCluster shutdown to finish."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = submitter.get_status(submission)
        if status.job_deployment_status == "Complete":
            return
        if status.job_deployment_status == "Failed":
            raise AssertionError(status.message or "KubeRay deployment failed")
        time.sleep(2)
    raise AssertionError("KubeRay deployment did not complete its shutdown")


def _assert_worker_pods(
    submitter: KubeRayJobSubmitter,
    cluster_name: str,
    *,
    expected_count: int,
    expected_cpu: float,
    expected_memory: str,
) -> None:
    deadline = time.monotonic() + 600
    worker_pods: list[Any] = []
    while time.monotonic() < deadline:
        pods = submitter.list_pods(
            namespace=os.environ["TRIBUTO_KUBERAY_NAMESPACE"],
            label_selector=f"ray.io/cluster={cluster_name}",
        )
        worker_pods = [
            pod
            for pod in pods
            if (pod.metadata.labels or {}).get("ray.io/node-type") == "worker"
        ]
        if len(worker_pods) == expected_count and all(
            pod.status.phase == "Running" for pod in worker_pods
        ):
            break
        time.sleep(2)
    assert len(worker_pods) == expected_count
    for pod in worker_pods:
        container = next(
            item for item in pod.spec.containers if item.name == "ray-worker"
        )
        assert str(container.resources.requests["cpu"]) == str(int(expected_cpu))
        assert str(container.resources.requests["memory"]) == expected_memory
        assert str(container.resources.limits["cpu"]) == str(int(expected_cpu))
        assert str(container.resources.limits["memory"]) == expected_memory


@pytest.mark.parametrize(
    ("worker_count", "num_cpus", "memory_bytes", "memory_quantity"),
    [
        (2, 1, _BYTES_PER_GIB, "1Gi"),
        (3, 2, 2 * _BYTES_PER_GIB, "2Gi"),
    ],
)
def test_kuberay_resource_profiles_submit_xgboost_and_execute(
    worker_count: int,
    num_cpus: float,
    memory_bytes: int,
    memory_quantity: str,
) -> None:
    """Each profile changes KubeRay resources and preserves workload correctness."""
    namespace, image = _require_it_environment()
    submitter = _submitter()
    request = _request(
        namespace=namespace,
        image=image,
        worker_count=worker_count,
        num_cpus=num_cpus,
        memory_bytes=memory_bytes,
    )
    submission = submitter.submit(request)
    try:
        cluster_name = _wait_for_cluster_name(submitter, submission)
        submitter.validate_submission_resources(request, submission)
        _assert_worker_pods(
            submitter,
            cluster_name,
            expected_count=worker_count,
            expected_cpu=num_cpus,
            expected_memory=memory_quantity,
        )

        status = submitter.wait(submission, timeout=900, poll_interval=2)
        assert status.job_status == "SUCCEEDED", status.message
        _wait_for_deployment_complete(submitter, submission)
        result = _wait_for_result(submitter, submission)
        assert result["status"] == "succeeded"
        assert result["algorithm"] == "xgboost"
        assert result["ray_version"] == "2.55.1"
        assert result["node_count"] >= 2
        assert result["requested_worker_count"] == worker_count
        assert result["requested_resources_per_worker"] == {
            "num_cpus": num_cpus,
            "num_gpus": 0.0,
            "memory_bytes": memory_bytes,
            "custom": {},
        }
        ray_resources = result["ray_resources"]
        assert ray_resources["required_resources_per_worker"] == {
            "CPU": num_cpus,
            "GPU": 0.0,
            "memory": float(memory_bytes),
        }
        assert ray_resources["required_total_resources"] == {
            "CPU": num_cpus * worker_count,
            "GPU": 0.0,
            "memory": float(memory_bytes * worker_count),
        }
        assert ray_resources["eligible_node_count"] >= worker_count
        assert ray_resources["cluster_resources"]["CPU"] >= num_cpus * worker_count
        assert (
            ray_resources["cluster_resources"]["memory"] >= memory_bytes * worker_count
        )
        assert len(ray_resources["alive_node_resources"]) >= worker_count
        receipt = result["execution_receipt"]
        assert receipt["requested_worker_count"] == worker_count
        assert receipt["distributed"] is True
        assert receipt["cluster_distributed"] is True
        assert receipt["execution_capability"] == "single_model_distributed"
        assert isinstance(result["outputs"].get("bundle_uri"), str)
        assert len(receipt["workers"]) == worker_count
        for worker in receipt["workers"]:
            resources = worker["resources"]
            assert resources["num_cpus"] >= num_cpus
    finally:
        submitter.cleanup(submission)
        submitter.wait_deleted(submission, timeout=300, poll_interval=2)
