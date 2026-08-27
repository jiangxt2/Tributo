"""Owned local Ray lifecycle and deployment-neutral cluster attachment."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from tributo._common.immutable import FrozenDict
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    ExecutionProfile,
    WorkerResources,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class LocalRuntimeOptions:
    """Optional Ray resource-registration overrides for ``local[*]``."""

    num_cpus: int | None = None
    num_gpus: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("num_cpus", "num_gpus"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise AlgorithmConfigurationError(
                    f"local runtime {field_name} override must be a non-negative "
                    "integer"
                )


@PublicAPI(stability="alpha")
class RayRuntimeSession:
    """One reference-counted lease on a Ray driver connection."""

    def __init__(
        self,
        manager: RayRuntimeManager,
        profile: ExecutionProfile,
        *,
        owned: bool,
        cluster_resources: Mapping[str, float],
        resource_preflight: str,
    ) -> None:
        self._manager = manager
        self.profile = profile
        self.owned = owned
        self.cluster_resources = FrozenDict(cluster_resources)
        self.resource_preflight = resource_preflight
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this lease has been released."""
        return self._closed

    @property
    def runtime_owned(self) -> bool:
        """Return whether Tributo created the underlying local Ray runtime."""
        return self.owned and self.profile is ExecutionProfile.LOCAL

    def close(self) -> None:
        """Release this lease; only the manager-owned final lease shuts Ray down."""
        if self._closed:
            return
        self._closed = True
        self._manager._release()

    def __enter__(self) -> RayRuntimeSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


@DeveloperAPI
class RayRuntimeManager:
    """Create ``local[*]`` or attach to Ray without owning a remote cluster."""

    def __init__(
        self,
        ray_module: Any | None = None,
        *,
        allow_external_cluster_connection: bool = True,
        allow_external_kubernetes_connection: bool | None = None,
        default_local_options: LocalRuntimeOptions | None = None,
    ) -> None:
        self._ray_module = ray_module
        if allow_external_kubernetes_connection is not None:
            warnings.warn(
                "allow_external_kubernetes_connection is deprecated; use "
                "allow_external_cluster_connection",
                DeprecationWarning,
                stacklevel=2,
            )
            if allow_external_cluster_connection is not True:
                raise AlgorithmConfigurationError(
                    "pass only one external cluster connection option"
                )
            allow_external_cluster_connection = allow_external_kubernetes_connection
        if not isinstance(allow_external_cluster_connection, bool):
            raise AlgorithmConfigurationError(
                "allow_external_cluster_connection must be a boolean"
            )
        self._allow_external_cluster_connection = allow_external_cluster_connection
        if default_local_options is not None and not isinstance(
            default_local_options, LocalRuntimeOptions
        ):
            raise AlgorithmConfigurationError(
                "default_local_options must be LocalRuntimeOptions"
            )
        self._default_local_options = default_local_options
        self._lock = threading.RLock()
        self._active_profile: ExecutionProfile | None = None
        self._manager_owned = False
        self._leases = 0

    def open(
        self,
        profile: ExecutionProfile,
        *,
        local_options: LocalRuntimeOptions | None = None,
        resources_per_worker: WorkerResources | None = None,
        worker_count: int = 1,
    ) -> RayRuntimeSession:
        """Open and preflight one driver connection before creating workers."""
        try:
            resolved_profile = ExecutionProfile(profile)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "execution profile must be 'local' or 'cluster'"
            ) from exc
        if (
            not isinstance(worker_count, int)
            or isinstance(worker_count, bool)
            or worker_count < 1
        ):
            raise AlgorithmConfigurationError("worker_count must be a positive integer")
        if resolved_profile is ExecutionProfile.CLUSTER and local_options:
            raise AlgorithmConfigurationError(
                "local runtime overrides are invalid for cluster execution"
            )
        if resolved_profile is ExecutionProfile.LOCAL and local_options is None:
            local_options = self._default_local_options
        ray = self._ray()
        with self._lock:
            started_here = False
            if ray.is_initialized():
                self._validate_existing_connection(resolved_profile)
            else:
                kwargs: dict[str, Any] = {
                    "address": (
                        "local"
                        if resolved_profile is ExecutionProfile.LOCAL
                        else "auto"
                    )
                }
                if resolved_profile is ExecutionProfile.LOCAL and local_options:
                    if local_options.num_cpus is not None:
                        kwargs["num_cpus"] = local_options.num_cpus
                    if local_options.num_gpus is not None:
                        kwargs["num_gpus"] = local_options.num_gpus
                try:
                    ray.init(**kwargs)
                except Exception as exc:
                    raise AlgorithmConfigurationError(
                        f"Ray {resolved_profile.value} initialization failed: "
                        f"{type(exc).__name__}"
                    ) from exc
                started_here = True
                self._active_profile = resolved_profile
                self._manager_owned = True
            try:
                cluster_resources = {
                    str(name): float(value)
                    for name, value in ray.cluster_resources().items()
                }
                resource_preflight = (
                    "validated"
                    if resolved_profile is ExecutionProfile.LOCAL
                    else "deferred_to_ray"
                )
                if (
                    resources_per_worker is not None
                    and resolved_profile is ExecutionProfile.LOCAL
                ):
                    self.validate_resources(
                        resources_per_worker,
                        worker_count,
                        cluster_resources=cluster_resources,
                        nodes=ray.nodes(),
                    )
            except BaseException:
                if started_here:
                    ray.shutdown()
                    self._active_profile = None
                    self._manager_owned = False
                raise
            self._leases += 1
            return RayRuntimeSession(
                self,
                resolved_profile,
                owned=self._manager_owned,
                cluster_resources=cluster_resources,
                resource_preflight=resource_preflight,
            )

    def _ray(self) -> Any:
        if self._ray_module is None:
            try:
                import ray
            except ImportError as exc:
                raise AlgorithmConfigurationError(
                    "Ray is required for algorithm execution"
                ) from exc
            self._ray_module = ray
        return self._ray_module

    def _validate_existing_connection(self, profile: ExecutionProfile) -> None:
        if self._active_profile is not None:
            if self._active_profile is not profile:
                raise AlgorithmConfigurationError(
                    "existing Ray connection profile conflicts with the request"
                )
            return
        if profile is ExecutionProfile.LOCAL:
            raise AlgorithmConfigurationError(
                "an externally initialized Ray connection cannot be proven to be "
                "Tributo-owned local[*]; refusing implicit reuse"
            )
        if not self._allow_external_cluster_connection:
            raise AlgorithmConfigurationError(
                "reuse of an externally initialized Ray cluster is disabled"
            )
        self._active_profile = ExecutionProfile.CLUSTER
        self._manager_owned = False

    def _release(self) -> None:
        with self._lock:
            if self._leases < 1:
                return
            self._leases -= 1
            if self._leases == 0:
                if self._manager_owned:
                    self._ray().shutdown()
                self._active_profile = None
                self._manager_owned = False

    @staticmethod
    def validate_resources(
        resources_per_worker: WorkerResources,
        worker_count: int,
        *,
        cluster_resources: Mapping[str, float],
        nodes: list[Mapping[str, Any]],
    ) -> None:
        """Validate total and per-node feasibility without CPU/GPU fallback."""
        total = resources_per_worker.scaled(worker_count)
        required = {
            "CPU": total.num_cpus,
            "GPU": total.num_gpus,
            **dict(total.custom),
        }
        if total.memory_bytes is not None:
            required["memory"] = total.memory_bytes
        shortages = {
            name: (amount, float(cluster_resources.get(name, 0.0)))
            for name, amount in required.items()
            if amount > float(cluster_resources.get(name, 0.0))
        }
        if shortages:
            details = ", ".join(
                f"{name} required={required_amount:g} available={available:g}"
                for name, (required_amount, available) in sorted(shortages.items())
            )
            raise AlgorithmConfigurationError(
                f"insufficient Ray cluster resources: {details}"
            )
        per_worker = {
            "CPU": resources_per_worker.num_cpus,
            "GPU": resources_per_worker.num_gpus,
            **dict(resources_per_worker.custom),
        }
        if resources_per_worker.memory_bytes is not None:
            per_worker["memory"] = resources_per_worker.memory_bytes
        alive_resources = [
            node.get("Resources", {}) for node in nodes if node.get("Alive", False)
        ]
        if not any(
            all(
                float(resources.get(name, 0.0)) >= amount
                for name, amount in per_worker.items()
            )
            for resources in alive_resources
        ):
            raise AlgorithmConfigurationError(
                "no alive Ray node can satisfy resources_per_worker"
            )


__all__ = ["LocalRuntimeOptions", "RayRuntimeManager", "RayRuntimeSession"]
