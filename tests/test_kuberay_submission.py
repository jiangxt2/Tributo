"""Unit tests for KubeRay resource compilation and submission identity."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tributo.algorithms.api import WorkerResources
from tributo.exceptions import (
    JobConfigurationError,
    JobExecutionError,
    JobSubmissionError,
    JobTimeoutError,
)
from tributo.kuberay_submission import (
    KubeRayDeploymentConfig,
    KubeRayJobRequest,
    KubeRayJobStatus,
    KubeRayJobSubmitter,
    KubeRayResourceProfile,
    KubeRayWorkerResources,
    build_kuberay_rayjob_manifest,
)


def _request(
    *,
    worker_count: int = 2,
    num_cpus: float = 2,
    memory_bytes: int = 2 * 1024**3,
    num_gpus: float = 0,
) -> KubeRayJobRequest:
    return KubeRayJobRequest(
        entrypoint="python -m tributo.cli algo run --config execution.json",
        run_id="run-1",
        resource_profile=KubeRayResourceProfile(
            worker_count=worker_count,
            resources_per_worker=KubeRayWorkerResources(
                num_cpus=num_cpus,
                memory_bytes=memory_bytes,
                num_gpus=num_gpus,
                custom={"accelerator": 1},
            ),
            entrypoint_num_cpus=1,
            entrypoint_custom_resources={"submitter": 1},
        ),
        deployment=KubeRayDeploymentConfig(
            namespace="tributo-it",
            image="registry.example/tributo-ray:it-1",
            service_account_name="tributo-runner",
        ),
        metadata={"tributo.test": "resource-profile"},
    )


def test_manifest_separates_worker_and_entrypoint_resources() -> None:
    manifest = build_kuberay_rayjob_manifest(_request())

    assert manifest["apiVersion"] == "ray.io/v1"
    assert manifest["kind"] == "RayJob"
    assert manifest["metadata"]["namespace"] == "tributo-it"
    assert manifest["metadata"]["labels"]["app.kubernetes.io/managed-by"] == ("tributo")

    spec = manifest["spec"]
    assert spec["submissionMode"] == "K8sJobMode"
    assert spec["jobId"] == manifest["metadata"]["name"]
    assert spec["entrypointNumCpus"] == 1.0
    assert spec["entrypointResources"] == '{"submitter":1.0}'
    assert "entrypointMemory" not in spec

    cluster = spec["rayClusterSpec"]
    worker_group = cluster["workerGroupSpecs"][0]
    assert worker_group["replicas"] == 2
    assert worker_group["minReplicas"] == 2
    assert worker_group["maxReplicas"] == 2
    assert worker_group["rayStartParams"] == {
        "num-cpus": "2",
        "resources": '{"accelerator":1.0}',
    }
    worker_container = worker_group["template"]["spec"]["containers"][0]
    assert worker_container["resources"] == {
        "requests": {"cpu": "2", "memory": "2Gi"},
        "limits": {"cpu": "2", "memory": "2Gi"},
    }
    assert worker_group["template"]["spec"]["serviceAccountName"] == ("tributo-runner")
    assert (
        spec["submitterPodTemplate"]["spec"]["serviceAccountName"] == "tributo-runner"
    )
    assert cluster["headGroupSpec"]["rayStartParams"] == {"num-cpus": "1"}


def test_profile_digest_changes_when_worker_resources_change() -> None:
    first = _request(worker_count=1).resource_profile.digest()
    second = _request(worker_count=2).resource_profile.digest()
    third = _request(memory_bytes=4 * 1024**3).resource_profile.digest()

    assert len(first) == 64
    assert first != second
    assert first != third


def test_submission_identity_changes_when_executable_request_changes() -> None:
    first = build_kuberay_rayjob_manifest(_request())
    second = build_kuberay_rayjob_manifest(
        _request().model_copy(update={"entrypoint": "python another_job.py"})
    )

    assert first["metadata"]["name"] != second["metadata"]["name"]
    assert (
        first["spec"]["metadata"]["tributo.request_digest"]
        != second["spec"]["metadata"]["tributo.request_digest"]
    )


def test_manifest_carries_shared_pod_storage_and_environment() -> None:
    request = _request()
    deployment = request.deployment.model_copy(
        update={
            "pod_env": {"OMP_NUM_THREADS": "1"},
            "volumes": (
                {
                    "name": "shared",
                    "emptyDir": {},
                },
            ),
            "volume_mounts": (
                {
                    "name": "shared",
                    "mountPath": "/tmp/shared",
                },
            ),
        }
    )
    manifest = build_kuberay_rayjob_manifest(
        request.model_copy(update={"deployment": deployment})
    )

    head_container = manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"][
        "spec"
    ]["containers"][0]
    assert head_container["env"] == [{"name": "OMP_NUM_THREADS", "value": "1"}]
    assert head_container["volumeMounts"] == [
        {"name": "shared", "mountPath": "/tmp/shared"}
    ]
    assert manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "volumes"
    ] == [{"name": "shared", "emptyDir": {}}]


def test_profile_rejects_reserved_custom_resource_names() -> None:
    with pytest.raises(ValueError, match="reserved Ray resource"):
        KubeRayWorkerResources(
            num_cpus=1,
            memory_bytes=1024,
            custom={"CPU": 1},
        )


def test_manifest_renders_whole_gpu_resources() -> None:
    manifest = build_kuberay_rayjob_manifest(_request(num_gpus=1))
    worker_group = manifest["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]

    assert worker_group["rayStartParams"]["num-gpus"] == "1"
    assert worker_group["template"]["spec"]["containers"][0]["resources"] == {
        "requests": {"cpu": "2", "memory": "2Gi", "nvidia.com/gpu": "1"},
        "limits": {"cpu": "2", "memory": "2Gi", "nvidia.com/gpu": "1"},
    }


@pytest.mark.parametrize(
    "metadata", [{"tributo.request_digest": "secret"}, {"api_key": "secret"}]
)
def test_request_rejects_reserved_or_sensitive_metadata(
    metadata: dict[str, str],
) -> None:
    payload = _request().model_dump(mode="python")
    payload["metadata"] = metadata

    with pytest.raises(ValueError, match="metadata"):
        KubeRayJobRequest.model_validate(payload)


def test_existing_worker_resource_contract_preserves_optional_memory() -> None:
    resources = WorkerResources(num_cpus=2, memory_bytes=1024)

    assert resources.to_dict() == {
        "num_cpus": 2.0,
        "num_gpus": 0.0,
        "custom": {},
        "memory_bytes": 1024,
    }
    assert resources.scaled(2).memory_bytes == 2048
    assert WorkerResources().to_dict() == {
        "num_cpus": 1.0,
        "num_gpus": 0.0,
        "custom": {},
    }


def test_manifest_rejects_fractional_kubernetes_gpu() -> None:
    with pytest.raises(JobConfigurationError, match="whole numbers"):
        build_kuberay_rayjob_manifest(_request(num_gpus=0.5))


def test_submit_create_and_status_are_associated_with_ray_identities() -> None:
    api = MagicMock()
    api.create_namespaced_custom_object.return_value = {
        "metadata": {"name": "created-name", "uid": "uid-1"},
        "status": {
            "jobId": "ray-job-1",
            "rayClusterName": "ray-cluster-1",
        },
    }
    api.get_namespaced_custom_object.return_value = {
        "status": {
            "jobStatus": "SUCCEEDED",
            "jobDeploymentStatus": "Complete",
            "jobId": "ray-job-1",
            "rayClusterName": "ray-cluster-1",
        }
    }

    request = _request()
    submitter = KubeRayJobSubmitter(custom_objects_api=api)
    submission = submitter.submit(request)
    status = submitter.wait(submission, timeout=1, poll_interval=0.01)

    assert submission.name == "created-name"
    assert submission.uid == "uid-1"
    assert submission.ray_job_id == "ray-job-1"
    assert submission.ray_cluster_name == "ray-cluster-1"
    assert isinstance(status, KubeRayJobStatus)
    assert status.terminal
    assert status.ray_job_id == "ray-job-1"
    api.create_namespaced_custom_object.assert_called_once()


def test_wait_raises_typed_timeout_for_non_terminal_status() -> None:
    api = MagicMock()
    api.create_namespaced_custom_object.return_value = {"metadata": {}}
    api.get_namespaced_custom_object.return_value = {"status": {}}
    submission = KubeRayJobSubmitter(custom_objects_api=api).submit(_request())

    with pytest.raises(JobTimeoutError, match="did not complete"):
        KubeRayJobSubmitter(custom_objects_api=api).wait(
            submission,
            timeout=0.01,
            poll_interval=0.01,
        )


def test_submit_reconciles_same_profile_after_conflict() -> None:
    class ConflictError(Exception):
        status = 409

    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = ConflictError()
    request = _request()
    manifest = build_kuberay_rayjob_manifest(request)
    api.get_namespaced_custom_object.return_value = {
        "metadata": {
            "name": manifest["metadata"]["name"],
            "labels": manifest["metadata"]["labels"],
        }
    }

    submission = KubeRayJobSubmitter(custom_objects_api=api).submit(request)

    assert submission.submission_id == manifest["spec"]["jobId"]
    api.get_namespaced_custom_object.assert_called_once()


def test_submit_rejects_conflict_with_different_resource_profile() -> None:
    class ConflictError(Exception):
        status = 409

    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = ConflictError()
    api.get_namespaced_custom_object.return_value = {
        "metadata": {"labels": {"tributo.io/resource-profile": "different"}}
    }

    with pytest.raises(JobSubmissionError, match="different resource profile"):
        KubeRayJobSubmitter(custom_objects_api=api).submit(_request())


def test_validate_submission_resources_checks_observed_raycluster() -> None:
    api = MagicMock()
    request = _request()
    manifest = build_kuberay_rayjob_manifest(request)
    api.create_namespaced_custom_object.return_value = {
        "metadata": {"name": manifest["metadata"]["name"]},
        "status": {"rayClusterName": "ray-cluster-1"},
    }
    api.get_namespaced_custom_object.return_value = {
        "spec": manifest["spec"]["rayClusterSpec"]
    }
    submitter = KubeRayJobSubmitter(custom_objects_api=api)
    submission = submitter.submit(request)

    submitter.validate_submission_resources(request, submission)
    assert api.get_namespaced_custom_object.call_args.args[3] == "rayclusters"


def test_validate_submission_resources_rejects_observed_worker_drift() -> None:
    api = MagicMock()
    request = _request()
    manifest = build_kuberay_rayjob_manifest(request)
    observed_spec = json.loads(json.dumps(manifest["spec"]["rayClusterSpec"]))
    observed_spec["workerGroupSpecs"][0]["replicas"] = 1
    api.create_namespaced_custom_object.return_value = {
        "metadata": {"name": manifest["metadata"]["name"]},
        "status": {"rayClusterName": "ray-cluster-1"},
    }
    api.get_namespaced_custom_object.return_value = {"spec": observed_spec}
    submitter = KubeRayJobSubmitter(custom_objects_api=api)
    submission = submitter.submit(request)

    with pytest.raises(JobExecutionError, match="do not match"):
        submitter.validate_submission_resources(request, submission)


def test_cleanup_is_scoped_to_the_submitted_rayjob() -> None:
    api = MagicMock()
    api.create_namespaced_custom_object.return_value = {"metadata": {}}
    request = _request()
    submitter = KubeRayJobSubmitter(custom_objects_api=api)
    submission = submitter.submit(request)

    assert submitter.cleanup(submission) is True
    delete_args = api.delete_namespaced_custom_object.call_args.args
    delete_kwargs = api.delete_namespaced_custom_object.call_args.kwargs
    assert delete_kwargs["body"] == {"propagationPolicy": "Foreground"}
    assert delete_args[2] == "tributo-it"
    assert delete_args[4] == submission.name


def test_wait_deleted_confirms_rayjob_disappearance() -> None:
    class NotFoundError(Exception):
        status = 404

    api = MagicMock()
    api.create_namespaced_custom_object.return_value = {"metadata": {}}
    api.get_namespaced_custom_object.side_effect = NotFoundError()
    request = _request()
    submitter = KubeRayJobSubmitter(custom_objects_api=api)
    submission = submitter.submit(request)

    submitter.wait_deleted(submission, timeout=1, poll_interval=0.01)


def test_get_logs_reads_only_submitter_pods_for_the_rayjob() -> None:
    custom_api = MagicMock()
    custom_api.create_namespaced_custom_object.return_value = {"metadata": {}}
    core_api = MagicMock()
    core_api.list_namespaced_pod.return_value.items = [
        type(
            "Pod",
            (),
            {"metadata": type("Metadata", (), {"name": "submitter-pod"})()},
        )()
    ]
    core_api.read_namespaced_pod_log.return_value = 'RESULT: {"status": "succeeded"}'
    request = _request()
    submitter = KubeRayJobSubmitter(
        custom_objects_api=custom_api,
        core_api=core_api,
    )
    submission = submitter.submit(request)

    assert "RESULT:" in submitter.get_logs(submission)
    core_api.list_namespaced_pod.assert_called_once_with(
        "tributo-it",
        label_selector=f"job-name={submission.name}",
    )
