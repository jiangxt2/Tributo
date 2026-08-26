"""Planner tests for invocation-scoped distributed runtime resolution."""

from __future__ import annotations

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmOperation,
    ExecutionProfile,
    ExecutionRequest,
    RuntimeTopology,
    WorkerResources,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.input import FakeInputResolver

from .conftest import (
    map_reduce_registration,
    request_for,
    sklearn_registration,
)


def _planner(registration) -> AlgorithmPlanner:
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    return AlgorithmPlanner(registry, {resolver.resolver_id: resolver})


def _request(worker_count: int = 2) -> ExecutionRequest:
    return ExecutionRequest(
        algorithm_request=request_for(
            "external_map_reduce",
            AlgorithmOperation.FIT,
            config={"alpha": 1.0},
        ),
        profile=ExecutionProfile.LOCAL,
        worker_count=worker_count,
    )


def test_planner_resolves_runtime_from_spec_and_execution_request() -> None:
    plan = _planner(map_reduce_registration()).plan(
        _request(),
        available_resources={"CPU": 2.0},
    )

    assert plan.runtime.topology is RuntimeTopology.RAY_MAP_REDUCE
    assert plan.runtime.worker_count == 2
    assert plan.runtime.execution_profile is ExecutionProfile.LOCAL
    assert plan.runtime.distribution_digest == plan.distribution_spec.digest
    assert plan.to_dict()["distribution_spec"]["strategy"] == "ray_map_reduce"


def test_formal_registration_requires_explicit_execution_request() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="ExecutionRequest"):
        _planner(map_reduce_registration()).plan(
            request_for("external_map_reduce", AlgorithmOperation.FIT)
        )


def test_legacy_joblib_registration_cannot_claim_formal_distribution() -> None:
    request = ExecutionRequest(
        algorithm_request=request_for("external_sklearn", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )

    with pytest.raises(AlgorithmConfigurationError, match="legacy compatibility"):
        _planner(sklearn_registration(framework_managed=True)).plan(request)


def test_planner_fails_before_input_open_when_cluster_resources_are_insufficient() -> (
    None
):
    with pytest.raises(AlgorithmConfigurationError, match="insufficient.*CPU"):
        _planner(map_reduce_registration()).plan(
            _request(worker_count=4),
            available_resources={"CPU": 3.0},
        )


def test_planner_rejects_untested_gpu_override_without_cpu_fallback() -> None:
    request = ExecutionRequest(
        algorithm_request=_request().algorithm_request,
        profile=ExecutionProfile.LOCAL,
        worker_count=2,
        resources_per_worker=WorkerResources(num_cpus=1, num_gpus=1),
    )

    with pytest.raises(AlgorithmConfigurationError, match="GPU.*exactly match"):
        _planner(map_reduce_registration()).plan(request)


def test_planner_carries_requested_custom_resources_into_runtime_binding() -> None:
    request = ExecutionRequest(
        algorithm_request=_request().algorithm_request,
        profile=ExecutionProfile.LOCAL,
        worker_count=2,
        resources_per_worker=WorkerResources(
            num_cpus=1,
            num_gpus=0,
            custom={"accelerator_type_a": 0.25},
        ),
    )

    plan = _planner(map_reduce_registration()).plan(
        request,
        available_resources={"CPU": 2.0, "accelerator_type_a": 0.5},
    )

    assert plan.runtime.custom_resources == {"accelerator_type_a": 0.25}
    assert plan.to_dict()["runtime"]["custom_resources"] == {"accelerator_type_a": 0.25}
