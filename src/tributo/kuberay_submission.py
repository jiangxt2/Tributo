"""KubeRay ``RayJob`` submission contracts and adapter.

This module owns one workload submission boundary.  It does not provision a
Kubernetes cluster, install KubeRay, or implement an operator.  The adapter
creates and observes a namespaced KubeRay ``RayJob`` whose embedded
``RayClusterSpec`` is derived from an explicit resource profile.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tributo._common.config import StrictConfigModel
from tributo._common.submission_id import generate_submission_id
from tributo.exceptions import (
    JobConfigurationError,
    JobExecutionError,
    JobSubmissionError,
    JobTimeoutError,
)
from tributo.util.annotations import PublicAPI

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_RESOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$")
_RESERVED_RESOURCES = frozenset({"CPU", "GPU", "memory", "object_store_memory"})
_TERMINAL_JOB_STATUSES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})
_TERMINAL_DEPLOYMENT_STATUSES = frozenset({"Complete", "Failed"})
_RESERVED_METADATA_KEYS = frozenset(
    {"tributo.request_digest", "tributo.resource_profile_digest"}
)
_SENSITIVE_METADATA_PARTS = frozenset(
    {
        "access_key",
        "access_key_id",
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "secret_access_key",
        "token",
    }
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BYTES_PER_KIB = 1024
_BYTES_PER_MIB = 1024**2
_BYTES_PER_GIB = 1024**3


def _finite_non_negative(value: float, field_name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return float(value)


def _format_cpu_quantity(value: float) -> str:
    """Format a Ray CPU quantity as a Kubernetes-compatible quantity."""
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _format_memory_quantity(value: int) -> str:
    """Format bytes using binary Kubernetes memory units when exact."""
    if value % _BYTES_PER_GIB == 0:
        return f"{value // _BYTES_PER_GIB}Gi"
    if value % _BYTES_PER_MIB == 0:
        return f"{value // _BYTES_PER_MIB}Mi"
    if value % _BYTES_PER_KIB == 0:
        return f"{value // _BYTES_PER_KIB}Ki"
    return str(value)


def _resource_requirements(
    *,
    num_cpus: float,
    memory_bytes: int,
    num_gpus: float,
) -> dict[str, dict[str, str]]:
    """Build equal Kubernetes requests and limits for one Ray container."""
    resources = {
        "cpu": _format_cpu_quantity(num_cpus),
        "memory": _format_memory_quantity(memory_bytes),
    }
    if num_gpus:
        if not num_gpus.is_integer():
            raise JobConfigurationError(
                "Kubernetes GPU resources must be whole numbers; fractional "
                "GPU allocation is not supported by this KubeRay profile"
            )
        resources["nvidia.com/gpu"] = str(int(num_gpus))
    return {"requests": dict(resources), "limits": dict(resources)}


def _validate_resource_mapping(
    value: Mapping[str, float], field_name: str
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, amount in value.items():
        if not isinstance(name, str) or not _RESOURCE_NAME_RE.fullmatch(name):
            raise ValueError(f"{field_name} contains an invalid resource name")
        if name in _RESERVED_RESOURCES:
            raise ValueError(
                f"{field_name} must not redefine the reserved Ray resource {name!r}"
            )
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise ValueError(f"{field_name} amounts must be finite numbers")
        normalized[name] = _finite_non_negative(float(amount), field_name)
    return dict(sorted(normalized.items()))


@PublicAPI(stability="alpha")
class KubeRayWorkerResources(StrictConfigModel):
    """Logical and Kubernetes resources reserved for one Ray worker."""

    num_cpus: float = Field(gt=0)
    memory_bytes: int = Field(gt=0)
    num_gpus: float = Field(default=0, ge=0)
    custom: dict[str, float] = Field(default_factory=dict)

    @field_validator("num_cpus", "num_gpus")
    @classmethod
    def validate_numeric_resource(cls, value: float, info: Any) -> float:
        return _finite_non_negative(float(value), info.field_name)

    @field_validator("memory_bytes")
    @classmethod
    def validate_memory(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("memory_bytes must be a positive integer")
        return value

    @field_validator("custom")
    @classmethod
    def validate_custom(cls, value: Mapping[str, float]) -> dict[str, float]:
        return _validate_resource_mapping(value, "custom")


@PublicAPI(stability="alpha")
class KubeRayResourceProfile(StrictConfigModel):
    """Per-job resource request independent of platform deployment settings."""

    worker_count: int = Field(ge=1)
    resources_per_worker: KubeRayWorkerResources
    entrypoint_num_cpus: float | None = Field(default=None, ge=0)
    entrypoint_num_gpus: float | None = Field(default=None, ge=0)
    entrypoint_custom_resources: dict[str, float] = Field(default_factory=dict)

    @field_validator("entrypoint_num_cpus", "entrypoint_num_gpus")
    @classmethod
    def validate_entrypoint_resource(
        cls, value: float | None, info: Any
    ) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(float(value), info.field_name)

    @field_validator("entrypoint_custom_resources")
    @classmethod
    def validate_entrypoint_custom_resources(
        cls, value: Mapping[str, float]
    ) -> dict[str, float]:
        return _validate_resource_mapping(value, "entrypoint_custom_resources")

    def digest(self) -> str:
        """Return a stable digest of only the resource profile."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@PublicAPI(stability="alpha")
class KubeRayDeploymentConfig(StrictConfigModel):
    """Platform-owned settings used to render one KubeRay RayJob."""

    namespace: str = "default"
    image: str = Field(min_length=1)
    worker_image: str | None = None
    ray_version: str = Field(default="2.55.1", min_length=1)
    head_num_cpus: float = Field(default=1, gt=0)
    head_memory_bytes: int = Field(default=_BYTES_PER_GIB, gt=0)
    head_num_gpus: float = Field(default=0, ge=0)
    service_account_name: str | None = None
    submitter_image: str | None = None
    submitter_service_account_name: str | None = None
    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"
    image_pull_secrets: tuple[str, ...] = ()
    submitter_pod_template: dict[str, Any] | None = None
    pod_env: dict[str, str] = Field(default_factory=dict)
    volumes: tuple[dict[str, Any], ...] = ()
    volume_mounts: tuple[dict[str, Any], ...] = ()
    runtime_env_yaml: str | None = None
    worker_group_name: str = "tributo-workers"
    shutdown_after_job_finishes: bool = True
    ttl_seconds_after_finished: int | None = Field(default=None, ge=0)

    @field_validator("namespace", "worker_group_name")
    @classmethod
    def validate_dns_label(cls, value: str, info: Any) -> str:
        if not _DNS_LABEL_RE.fullmatch(value) or len(value) > 63:
            raise ValueError(f"{info.field_name} must be a Kubernetes DNS label")
        return value

    @field_validator("service_account_name", "submitter_service_account_name")
    @classmethod
    def validate_service_account(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) > 63 or _DNS_LABEL_RE.fullmatch(value) is None
        ):
            raise ValueError("service_account_name must be a Kubernetes DNS label")
        return value

    @field_validator("image_pull_secrets")
    @classmethod
    def validate_image_pull_secrets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        secrets = tuple(value)
        if any(not _DNS_LABEL_RE.fullmatch(name) or len(name) > 63 for name in secrets):
            raise ValueError("image_pull_secrets must contain Kubernetes DNS labels")
        if len(set(secrets)) != len(secrets):
            raise ValueError("image_pull_secrets must not contain duplicates")
        return secrets

    @field_validator("pod_env")
    @classmethod
    def validate_pod_env(cls, value: Mapping[str, str]) -> dict[str, str]:
        if any(not _ENV_NAME_RE.fullmatch(name) for name in value):
            raise ValueError("pod_env contains an invalid environment variable name")
        return dict(value)

    @field_validator("volumes", "volume_mounts")
    @classmethod
    def validate_pod_fragments(
        cls, value: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        fragments = tuple(dict(item) for item in value)
        names = [item.get("name") for item in fragments]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("volumes and volume_mounts require named fragments")
        if len(set(names)) != len(names):
            raise ValueError("volumes and volume_mounts must not contain duplicates")
        return fragments

    @field_validator("head_num_cpus", "head_num_gpus")
    @classmethod
    def validate_head_resource(cls, value: float, info: Any) -> float:
        return _finite_non_negative(float(value), info.field_name)

    @field_validator("head_memory_bytes")
    @classmethod
    def validate_head_memory(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("head_memory_bytes must be a positive integer")
        return value

    @field_validator("image", "worker_image", "submitter_image", "ray_version")
    @classmethod
    def validate_non_empty_string(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_ttl(self) -> KubeRayDeploymentConfig:
        if self.ttl_seconds_after_finished is not None and not (
            self.shutdown_after_job_finishes
        ):
            raise ValueError(
                "ttl_seconds_after_finished requires shutdown_after_job_finishes"
            )
        return self

    @model_validator(mode="after")
    def validate_submitter_template(self) -> KubeRayDeploymentConfig:
        if self.submitter_pod_template is not None:
            spec = self.submitter_pod_template.get("spec")
            if not isinstance(spec, Mapping) or not isinstance(
                spec.get("containers"), list
            ):
                raise ValueError("submitter_pod_template must contain spec.containers")
        volume_names = {item["name"] for item in self.volumes}
        mount_names = {item["name"] for item in self.volume_mounts}
        if not mount_names.issubset(volume_names):
            raise ValueError("volume_mounts must reference declared volumes")
        return self

    @property
    def resolved_worker_image(self) -> str:
        """Return the worker image, falling back to the common image."""
        return self.worker_image or self.image


@PublicAPI(stability="alpha")
class KubeRayJobRequest(StrictConfigModel):
    """One immutable request accepted by :class:`KubeRayJobSubmitter`."""

    entrypoint: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(default="attempt-1", min_length=1)
    resource_profile: KubeRayResourceProfile
    deployment: KubeRayDeploymentConfig
    metadata: dict[str, str] = Field(default_factory=dict)
    request_digest: str | None = Field(default=None, pattern=_DIGEST_RE.pattern)

    @field_validator("entrypoint", "run_id", "attempt_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("string fields must not be empty")
        return value.strip()

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Mapping[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for name, item in value.items():
            if name in _RESERVED_METADATA_KEYS:
                raise ValueError(f"metadata must not define reserved key {name!r}")
            normalized_name = name.lower().replace("-", "_").replace(".", "_")
            if normalized_name in _SENSITIVE_METADATA_PARTS or any(
                normalized_name.startswith(f"{part}_")
                or normalized_name.endswith(f"_{part}")
                or f"_{part}_" in normalized_name
                for part in _SENSITIVE_METADATA_PARTS
            ):
                raise ValueError("metadata must not contain credential fields")
            normalized[name] = item
        return normalized


@PublicAPI(stability="alpha")
class KubeRayJobSubmission(StrictConfigModel):
    """Identity returned after a KubeRay RayJob CR is accepted."""

    namespace: str
    name: str
    uid: str | None = None
    run_id: str
    attempt_id: str
    submission_id: str
    resource_profile_digest: str = Field(pattern=_DIGEST_RE.pattern)
    ray_job_id: str | None = None
    ray_cluster_name: str | None = None


@PublicAPI(stability="alpha")
class KubeRayJobStatus(StrictConfigModel):
    """Observed KubeRay RayJob state and its linked Ray identities."""

    namespace: str
    name: str
    job_status: str = ""
    job_deployment_status: str = ""
    ray_job_id: str | None = None
    ray_cluster_name: str | None = None
    reason: str | None = None
    message: str | None = None

    @property
    def terminal(self) -> bool:
        """Return whether either KubeRay lifecycle reaches a terminal state."""
        return self.job_status in _TERMINAL_JOB_STATUSES or (
            self.job_deployment_status in _TERMINAL_DEPLOYMENT_STATUSES
        )


def _submission_id(request: KubeRayJobRequest) -> str:
    request_digest = _request_digest(request)
    return generate_submission_id(
        "kuberay",
        request.run_id,
        request.attempt_id,
        request_digest,
    )


def _request_digest(request: KubeRayJobRequest) -> str:
    """Return a stable digest for the executable RayJob request."""
    if request.request_digest is not None:
        return request.request_digest
    payload = {
        "attempt_id": request.attempt_id,
        "deployment": request.deployment.model_dump(mode="json"),
        "entrypoint": request.entrypoint,
        "metadata": dict(sorted(request.metadata.items())),
        "resource_profile": request.resource_profile.model_dump(mode="json"),
        "run_id": request.run_id,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ray_custom_resources(resources: Mapping[str, float]) -> str | None:
    if not resources:
        return None
    return json.dumps(dict(sorted(resources.items())), separators=(",", ":"))


def _pod_template(
    *,
    name: str,
    image: str,
    image_pull_policy: str,
    resources: dict[str, dict[str, str]],
    service_account_name: str | None,
    image_pull_secrets: tuple[str, ...],
    pod_env: Mapping[str, str],
    volumes: tuple[dict[str, Any], ...],
    volume_mounts: tuple[dict[str, Any], ...],
    ports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pod_spec: dict[str, Any] = {
        "containers": [
            {
                "name": name,
                "image": image,
                "imagePullPolicy": image_pull_policy,
                "resources": resources,
            }
        ]
    }
    if pod_env:
        pod_spec["containers"][0]["env"] = [
            {"name": name, "value": value} for name, value in sorted(pod_env.items())
        ]
    if volumes:
        pod_spec["volumes"] = [dict(item) for item in volumes]
    if volume_mounts:
        pod_spec["containers"][0]["volumeMounts"] = [
            dict(item) for item in volume_mounts
        ]
    if service_account_name is not None:
        pod_spec["serviceAccountName"] = service_account_name
    if image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": name} for name in image_pull_secrets]
    if ports:
        pod_spec["containers"][0]["ports"] = ports
    return {"spec": pod_spec}


@PublicAPI(stability="alpha")
def build_kuberay_rayjob_manifest(request: KubeRayJobRequest) -> dict[str, Any]:
    """Compile a validated Tributo request into a KubeRay ``RayJob`` object."""
    submission_id = _submission_id(request)
    profile = request.resource_profile
    deployment = request.deployment
    worker = profile.resources_per_worker
    profile_digest = profile.digest()
    request_digest = _request_digest(request)
    labels = {
        "app.kubernetes.io/managed-by": "tributo",
        "tributo.io/submission-id": submission_id,
        "tributo.io/resource-profile": profile_digest[:16],
        "tributo.io/request-digest": request_digest[:16],
    }
    job_metadata = dict(request.metadata)
    job_metadata["tributo.resource_profile_digest"] = profile_digest
    job_metadata["tributo.request_digest"] = request_digest

    worker_ray_start_params = {
        "num-cpus": _format_cpu_quantity(worker.num_cpus),
    }
    if worker.num_gpus:
        worker_ray_start_params["num-gpus"] = _format_cpu_quantity(worker.num_gpus)
    custom_resources = _ray_custom_resources(worker.custom)
    if custom_resources is not None:
        worker_ray_start_params["resources"] = custom_resources

    head_resources = _resource_requirements(
        num_cpus=deployment.head_num_cpus,
        memory_bytes=deployment.head_memory_bytes,
        num_gpus=deployment.head_num_gpus,
    )
    worker_resources = _resource_requirements(
        num_cpus=worker.num_cpus,
        memory_bytes=worker.memory_bytes,
        num_gpus=worker.num_gpus,
    )

    ray_cluster_spec: dict[str, Any] = {
        "rayVersion": deployment.ray_version,
        "headGroupSpec": {
            "rayStartParams": {
                "num-cpus": _format_cpu_quantity(deployment.head_num_cpus)
            },
            "template": _pod_template(
                name="ray-head",
                image=deployment.image,
                image_pull_policy=deployment.image_pull_policy,
                resources=head_resources,
                service_account_name=deployment.service_account_name,
                image_pull_secrets=deployment.image_pull_secrets,
                pod_env=deployment.pod_env,
                volumes=deployment.volumes,
                volume_mounts=deployment.volume_mounts,
                ports=[
                    {"name": "gcs-server", "containerPort": 6379},
                    {"name": "dashboard", "containerPort": 8265},
                    {"name": "client", "containerPort": 10001},
                ],
            ),
        },
        "workerGroupSpecs": [
            {
                "groupName": deployment.worker_group_name,
                "replicas": profile.worker_count,
                "minReplicas": profile.worker_count,
                "maxReplicas": profile.worker_count,
                "rayStartParams": worker_ray_start_params,
                "template": _pod_template(
                    name="ray-worker",
                    image=deployment.resolved_worker_image,
                    image_pull_policy=deployment.image_pull_policy,
                    resources=worker_resources,
                    service_account_name=deployment.service_account_name,
                    image_pull_secrets=deployment.image_pull_secrets,
                    pod_env=deployment.pod_env,
                    volumes=deployment.volumes,
                    volume_mounts=deployment.volume_mounts,
                ),
            }
        ],
    }

    spec: dict[str, Any] = {
        "submissionMode": "K8sJobMode",
        "entrypoint": request.entrypoint,
        "jobId": submission_id,
        "metadata": job_metadata,
        "rayClusterSpec": ray_cluster_spec,
        "shutdownAfterJobFinishes": deployment.shutdown_after_job_finishes,
    }
    if deployment.submitter_pod_template is not None:
        spec["submitterPodTemplate"] = deployment.submitter_pod_template
    else:
        submitter_template = _pod_template(
            name="rayjob-submitter",
            image=deployment.submitter_image or deployment.image,
            image_pull_policy=deployment.image_pull_policy,
            resources={"requests": {}, "limits": {}},
            service_account_name=(
                deployment.submitter_service_account_name
                or deployment.service_account_name
            ),
            image_pull_secrets=deployment.image_pull_secrets,
            pod_env=deployment.pod_env,
            volumes=deployment.volumes,
            volume_mounts=deployment.volume_mounts,
        )
        submitter_template["spec"]["restartPolicy"] = "Never"
        submitter_template["spec"]["containers"][0].pop("resources", None)
        spec["submitterPodTemplate"] = submitter_template
    if deployment.runtime_env_yaml is not None:
        spec["runtimeEnvYAML"] = deployment.runtime_env_yaml
    if deployment.ttl_seconds_after_finished is not None:
        spec["ttlSecondsAfterFinished"] = deployment.ttl_seconds_after_finished
    if profile.entrypoint_num_cpus is not None:
        spec["entrypointNumCpus"] = profile.entrypoint_num_cpus
    if profile.entrypoint_num_gpus is not None:
        spec["entrypointNumGpus"] = profile.entrypoint_num_gpus
    entrypoint_resources = _ray_custom_resources(profile.entrypoint_custom_resources)
    if entrypoint_resources is not None:
        spec["entrypointResources"] = entrypoint_resources

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": submission_id,
            "namespace": deployment.namespace,
            "labels": labels,
        },
        "spec": spec,
    }


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status", None)
    return value if isinstance(value, int) else None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    raise JobExecutionError("Kubernetes API returned an unsupported object")


@PublicAPI(stability="alpha")
class KubeRayJobSubmitter:
    """Submit and observe one KubeRay RayJob without owning the cluster."""

    group = "ray.io"
    version = "v1"
    plural = "rayjobs"

    def __init__(
        self,
        *,
        custom_objects_api: Any | None = None,
        core_api: Any | None = None,
        context: str | None = None,
    ):
        if custom_objects_api is None:
            loaded_custom_objects_api, loaded_core_api = self._load_kubernetes_apis(
                context
            )
            custom_objects_api = loaded_custom_objects_api
            if core_api is None:
                core_api = loaded_core_api
        self._custom_objects_api = custom_objects_api
        self._core_api = core_api

    @staticmethod
    def _load_kubernetes_apis(context: str | None) -> tuple[Any, Any]:
        try:
            from kubernetes import client, config
        except ImportError as exc:
            raise JobConfigurationError(
                "KubeRay submission requires the optional 'kuberay' extra; "
                "install tributo[kuberay]"
            ) from exc
        try:
            config.load_kube_config(context=context)
        except Exception as kubeconfig_error:
            try:
                config.load_incluster_config()
            except Exception as incluster_error:
                raise JobConfigurationError(
                    "unable to load Kubernetes kubeconfig or in-cluster config"
                ) from incluster_error
            del kubeconfig_error
        return client.CustomObjectsApi(), client.CoreV1Api()

    def _get(self, namespace: str, name: str) -> dict[str, Any]:
        return self._get_custom_object(namespace, self.plural, name)

    def _get_custom_object(
        self, namespace: str, plural: str, name: str
    ) -> dict[str, Any]:
        try:
            return _as_dict(
                self._custom_objects_api.get_namespaced_custom_object(
                    self.group,
                    self.version,
                    namespace,
                    plural,
                    name,
                )
            )
        except Exception as exc:
            raise JobExecutionError(
                f"failed to read KubeRay object {plural}/{name!r}"
            ) from exc

    @staticmethod
    def _submission_from_object(
        request: KubeRayJobRequest, obj: Mapping[str, Any]
    ) -> KubeRayJobSubmission:
        metadata = obj.get("metadata") or {}
        status = obj.get("status") or {}
        return KubeRayJobSubmission(
            namespace=request.deployment.namespace,
            name=str(metadata.get("name") or _submission_id(request)),
            uid=(str(metadata["uid"]) if metadata.get("uid") is not None else None),
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            submission_id=_submission_id(request),
            resource_profile_digest=request.resource_profile.digest(),
            ray_job_id=(str(status["jobId"]) if status.get("jobId") else None),
            ray_cluster_name=(
                str(status["rayClusterName"]) if status.get("rayClusterName") else None
            ),
        )

    def submit(self, request: KubeRayJobRequest) -> KubeRayJobSubmission:
        """Create or reconcile one namespaced KubeRay RayJob."""
        manifest = build_kuberay_rayjob_manifest(request)
        namespace = request.deployment.namespace
        name = manifest["metadata"]["name"]
        try:
            obj = self._custom_objects_api.create_namespaced_custom_object(
                self.group,
                self.version,
                namespace,
                self.plural,
                manifest,
            )
        except Exception as exc:
            if _status_code(exc) != 409:
                raise JobSubmissionError(
                    f"failed to submit KubeRay RayJob {name!r}"
                ) from exc
            existing = self._get(namespace, name)
            labels = (existing.get("metadata") or {}).get("labels") or {}
            expected_digest = request.resource_profile.digest()[:16]
            if labels.get("tributo.io/resource-profile") != expected_digest:
                raise JobSubmissionError(
                    f"KubeRay RayJob {name!r} exists with a different resource profile"
                ) from exc
            obj = existing
        return self._submission_from_object(request, _as_dict(obj))

    def get_status(self, submission: KubeRayJobSubmission) -> KubeRayJobStatus:
        """Read one KubeRay RayJob status."""
        obj = self._get(submission.namespace, submission.name)
        status = obj.get("status") or {}
        return KubeRayJobStatus(
            namespace=submission.namespace,
            name=submission.name,
            job_status=str(status.get("jobStatus") or ""),
            job_deployment_status=str(status.get("jobDeploymentStatus") or ""),
            ray_job_id=(str(status["jobId"]) if status.get("jobId") else None),
            ray_cluster_name=(
                str(status["rayClusterName"]) if status.get("rayClusterName") else None
            ),
            reason=(str(status["reason"]) if status.get("reason") else None),
            message=(str(status["message"]) if status.get("message") else None),
        )

    def get_ray_cluster(self, submission: KubeRayJobSubmission) -> dict[str, Any]:
        """Return the RayCluster linked to one submitted RayJob."""
        cluster_name = (
            submission.ray_cluster_name or self.get_status(submission).ray_cluster_name
        )
        if not cluster_name:
            raise JobExecutionError(
                f"KubeRay RayJob {submission.name!r} has no RayCluster identity"
            )
        return self._get_custom_object(
            submission.namespace,
            "rayclusters",
            cluster_name,
        )

    def list_pods(
        self,
        *,
        namespace: str,
        label_selector: str = "",
    ) -> list[Any]:
        """List Pods through the injected Kubernetes Core API."""
        if self._core_api is None:
            raise JobConfigurationError(
                "Kubernetes Pod listing requires the Kubernetes CoreV1 API"
            )
        try:
            pods = self._core_api.list_namespaced_pod(
                namespace,
                label_selector=label_selector,
            )
        except Exception as exc:
            raise JobExecutionError(
                f"failed to list Kubernetes Pods in namespace {namespace!r}"
            ) from exc
        items = getattr(pods, "items", None)
        if items is None and isinstance(pods, dict):
            items = pods.get("items")
        return list(items or ())

    def validate_submission_resources(
        self,
        request: KubeRayJobRequest,
        submission: KubeRayJobSubmission,
    ) -> None:
        """Verify the observed RayCluster still matches the submitted profile."""
        try:
            expected = build_kuberay_rayjob_manifest(request)["spec"]["rayClusterSpec"]
            observed = self.get_ray_cluster(submission).get("spec") or {}
            expected_groups = expected["workerGroupSpecs"]
            observed_groups = observed.get("workerGroupSpecs") or []
            group_name = request.deployment.worker_group_name
            matching_groups = [
                group
                for group in observed_groups
                if group.get("groupName") == group_name
            ]
            if len(matching_groups) != 1:
                raise JobExecutionError(
                    f"KubeRay RayCluster has no unique worker group {group_name!r}"
                )
            expected_group = expected_groups[0]
            observed_group = matching_groups[0]
            expected_container = expected_group["template"]["spec"]["containers"][0]
            observed_container = observed_group["template"]["spec"]["containers"][0]
        except JobExecutionError:
            raise
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise JobExecutionError(
                "KubeRay RayCluster resource specification is malformed"
            ) from exc
        if any(
            observed_group.get(field) != expected_group.get(field)
            for field in ("replicas", "minReplicas", "maxReplicas", "rayStartParams")
        ) or observed_container.get("resources") != expected_container.get("resources"):
            raise JobExecutionError(
                "observed KubeRay worker resources do not match the submitted profile"
            )

    def get_logs(self, submission: KubeRayJobSubmission) -> str:
        """Read logs from the KubeRay submitter Pod for one RayJob."""
        if self._core_api is None:
            raise JobConfigurationError(
                "KubeRay log retrieval requires the Kubernetes CoreV1 API"
            )
        pod_items = self.list_pods(
            namespace=submission.namespace,
            label_selector=f"job-name={submission.name}",
        )
        if not pod_items:
            return ""
        logs: list[str] = []
        for pod in pod_items:
            metadata = getattr(pod, "metadata", None)
            pod_name = getattr(metadata, "name", None)
            if isinstance(pod, dict):
                pod_name = (pod.get("metadata") or {}).get("name")
            if not isinstance(pod_name, str) or not pod_name:
                continue
            try:
                logs.append(
                    str(
                        self._core_api.read_namespaced_pod_log(
                            pod_name,
                            submission.namespace,
                        )
                    )
                )
            except Exception as exc:
                raise JobExecutionError(
                    f"failed to read KubeRay submitter logs for {pod_name!r}"
                ) from exc
        return "\n".join(logs)

    def wait(
        self,
        submission: KubeRayJobSubmission,
        *,
        timeout: float = 1800,
        poll_interval: float = 2,
    ) -> KubeRayJobStatus:
        """Wait for a KubeRay RayJob or raise a typed timeout."""
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_status(submission)
            if status.terminal:
                return status
            time.sleep(poll_interval)
        raise JobTimeoutError(
            f"KubeRay RayJob {submission.name!r} did not complete within {timeout}s"
        )

    def cleanup(self, submission: KubeRayJobSubmission) -> bool:
        """Delete only the RayJob owned by this submission."""
        try:
            self._custom_objects_api.delete_namespaced_custom_object(
                self.group,
                self.version,
                submission.namespace,
                self.plural,
                submission.name,
                body={"propagationPolicy": "Foreground"},
            )
        except Exception as exc:
            if _status_code(exc) == 404:
                return False
            raise JobExecutionError(
                f"failed to clean up KubeRay RayJob {submission.name!r}"
            ) from exc
        return True

    def wait_deleted(
        self,
        submission: KubeRayJobSubmission,
        *,
        timeout: float = 300,
        poll_interval: float = 2,
    ) -> None:
        """Wait until the RayJob and owned RayCluster deletion is observable."""
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._custom_objects_api.get_namespaced_custom_object(
                    self.group,
                    self.version,
                    submission.namespace,
                    self.plural,
                    submission.name,
                )
            except Exception as exc:
                if _status_code(exc) == 404:
                    return
                raise JobExecutionError(
                    f"failed to observe deletion of KubeRay RayJob {submission.name!r}"
                ) from exc
            time.sleep(poll_interval)
        raise JobTimeoutError(
            f"KubeRay RayJob {submission.name!r} was not deleted within {timeout}s"
        )


__all__ = [
    "KubeRayDeploymentConfig",
    "KubeRayJobRequest",
    "KubeRayJobStatus",
    "KubeRayJobSubmission",
    "KubeRayJobSubmitter",
    "KubeRayResourceProfile",
    "KubeRayWorkerResources",
    "build_kuberay_rayjob_manifest",
]
