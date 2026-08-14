"""Unit tests for Ray runtime ownership and resource preflight."""

from __future__ import annotations

from typing import Any

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    ExecutionProfile,
    WorkerResources,
)
from tributo.algorithms.core import LocalRuntimeOptions, RayRuntimeManager


class FakeRay:
    def __init__(self, *, initialized: bool = False) -> None:
        self.initialized = initialized
        self.init_calls: list[dict[str, Any]] = []
        self.shutdown_calls = 0
        self.resources = {"CPU": 8.0, "GPU": 0.0}
        self.node_resources = [{"Alive": True, "Resources": {"CPU": 8.0, "GPU": 0.0}}]

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, **kwargs: Any) -> None:
        self.init_calls.append(kwargs)
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.initialized = False

    def cluster_resources(self) -> dict[str, float]:
        return dict(self.resources)

    def nodes(self) -> list[dict[str, Any]]:
        return list(self.node_resources)


def test_local_runtime_uses_ray_detection_by_default_and_shuts_owned_connection() -> (
    None
):
    ray = FakeRay()
    manager = RayRuntimeManager(ray)

    with manager.open(ExecutionProfile.LOCAL) as session:
        assert session.owned is True
        assert session.runtime_owned is True
        assert ray.init_calls == [{"address": "local"}]
        assert ray.shutdown_calls == 0

    assert ray.shutdown_calls == 1


def test_local_runtime_passes_only_explicit_resource_overrides() -> None:
    ray = FakeRay()
    manager = RayRuntimeManager(ray)

    manager.open(
        ExecutionProfile.LOCAL,
        local_options=LocalRuntimeOptions(num_cpus=4, num_gpus=0),
    ).close()

    assert ray.init_calls == [{"address": "local", "num_cpus": 4.0, "num_gpus": 0.0}]


def test_local_runtime_rejects_fractional_raylet_resource_overrides() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="integer"):
        LocalRuntimeOptions(num_cpus=1.5)


def test_local_runtime_uses_constructor_resource_overrides() -> None:
    ray = FakeRay()
    manager = RayRuntimeManager(
        ray,
        default_local_options=LocalRuntimeOptions(num_cpus=3, num_gpus=0),
    )

    manager.open(ExecutionProfile.LOCAL).close()

    assert ray.init_calls == [{"address": "local", "num_cpus": 3.0, "num_gpus": 0.0}]


def test_nested_owned_leases_shutdown_only_after_the_last_release() -> None:
    ray = FakeRay()
    manager = RayRuntimeManager(ray)

    first = manager.open(ExecutionProfile.LOCAL)
    second = manager.open(ExecutionProfile.LOCAL)
    first.close()
    assert ray.shutdown_calls == 0
    second.close()
    assert ray.shutdown_calls == 1


def test_external_connection_is_never_implicitly_reused_as_local() -> None:
    ray = FakeRay(initialized=True)
    manager = RayRuntimeManager(ray)

    with pytest.raises(AlgorithmConfigurationError, match="externally initialized"):
        manager.open(ExecutionProfile.LOCAL)

    assert ray.shutdown_calls == 0


def test_explicit_external_kubernetes_connection_is_borrowed_not_shutdown() -> None:
    ray = FakeRay(initialized=True)
    manager = RayRuntimeManager(
        ray,
        allow_external_kubernetes_connection=True,
    )

    session = manager.open(ExecutionProfile.KUBERNETES)
    assert session.owned is False
    assert session.runtime_owned is False
    session.close()

    assert ray.shutdown_calls == 0


def test_kubernetes_connection_requires_verified_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    ray = FakeRay()
    manager = RayRuntimeManager(ray)

    with pytest.raises(AlgorithmConfigurationError, match="Kubernetes process"):
        manager.open(ExecutionProfile.KUBERNETES)

    assert ray.init_calls == []


def test_resource_failure_closes_only_the_connection_started_by_manager() -> None:
    ray = FakeRay()
    manager = RayRuntimeManager(ray)

    with pytest.raises(AlgorithmConfigurationError, match="insufficient.*GPU"):
        manager.open(
            ExecutionProfile.LOCAL,
            resources_per_worker=WorkerResources(num_cpus=1, num_gpus=1),
            worker_count=2,
        )

    assert ray.shutdown_calls == 1


def test_per_worker_placement_is_validated_separately_from_cluster_total() -> None:
    ray = FakeRay()
    ray.resources = {"CPU": 8.0, "GPU": 2.0}
    ray.node_resources = [
        {"Alive": True, "Resources": {"CPU": 4.0, "GPU": 0.5}},
        {"Alive": True, "Resources": {"CPU": 4.0, "GPU": 1.5}},
    ]
    manager = RayRuntimeManager(ray)

    with pytest.raises(AlgorithmConfigurationError, match="no alive Ray node"):
        manager.open(
            ExecutionProfile.LOCAL,
            resources_per_worker=WorkerResources(num_cpus=1, num_gpus=2),
            worker_count=1,
        )

    assert ray.shutdown_calls == 1


def test_custom_resources_are_checked_globally_and_per_worker() -> None:
    ray = FakeRay()
    ray.resources = {"CPU": 8.0, "GPU": 0.0, "accelerator_type_a": 1.0}
    ray.node_resources = [
        {
            "Alive": True,
            "Resources": {
                "CPU": 8.0,
                "GPU": 0.0,
                "accelerator_type_a": 0.25,
            },
        }
    ]
    manager = RayRuntimeManager(ray)

    with pytest.raises(AlgorithmConfigurationError, match="no alive Ray node"):
        manager.open(
            ExecutionProfile.LOCAL,
            resources_per_worker=WorkerResources(
                num_cpus=1,
                custom={"accelerator_type_a": 0.5},
            ),
            worker_count=2,
        )

    assert ray.shutdown_calls == 1
