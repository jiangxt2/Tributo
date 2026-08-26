"""Tests for fit-only Tune execution of portable algorithm Wheels."""

from __future__ import annotations

from dataclasses import replace

from tests.algorithms.conftest import map_reduce_registration, request_for
from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRequest,
    ExecutionProfile,
    ExecutionRequest,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.input import FakeInputResolver
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.training.algorithm_spec import Capability
from tributo.training.portable_tune import (
    PortableTuneRunner,
    _fit_only_plan,
    _trial_request,
)
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec


def _resolved_plan():
    registration = map_reduce_registration()
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})
    return planner.plan(
        ExecutionRequest(
            algorithm_request=request_for(
                "external_map_reduce", AlgorithmOperation.FIT
            ),
            profile=ExecutionProfile.CLUSTER,
            worker_count=2,
        )
    )


def test_fit_only_plan_rebinds_runtime_digest_and_disables_output_contract() -> None:
    plan = _fit_only_plan(_resolved_plan())
    assert plan.distribution_spec is not None
    assert plan.distribution_spec.result_policy.value == "fit_only"
    assert plan.runtime.distribution_digest == plan.distribution_spec.digest
    assert plan.contract_bindings is None
    plan.validate_integrity()


def test_trial_request_applies_only_sampled_config_and_isolates_checkpoint() -> None:
    base = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="external",
            operation=AlgorithmOperation.FIT,
            input_binding=request_for("external", AlgorithmOperation.FIT).input_binding,
            algorithm_config={
                "learning_rate": 0.1,
                "runtime": {"checkpoint_dir": "/old"},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
        resume_from="/resume",
    )
    trial = _trial_request(
        base,
        {"learning_rate": 0.2},
        checkpoint_dir="/trial/checkpoint",
    )
    assert trial.resume_from is None
    assert trial.algorithm_request.algorithm_config["learning_rate"] == 0.2
    assert trial.algorithm_request.algorithm_config["runtime"] == {
        "checkpoint_dir": "/trial/checkpoint"
    }


def test_portable_tune_requires_tunable_descriptor_and_cluster_profile() -> None:
    registration = map_reduce_registration()
    descriptor = registration.spec
    from tributo.algorithms.api import DistributedAlgorithmDescriptor

    distributed = DistributedAlgorithmDescriptor(
        package_name="tests-portable-tune",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        registration=replace(
            registration,
            spec=replace(
                descriptor,
                capabilities=(*descriptor.capabilities, Capability.TUNABLE),
            ),
        ),
    )
    request = ExecutionRequest(
        algorithm_request=request_for("external_map_reduce", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    runner = PortableTuneRunner(
        distributed,
        request,
        TuneSearchConfig(metric="loss", num_samples=1),
        SearchSpaceSpec(
            parameters=(
                SearchParamSpec(
                    path="learning_rate",
                    kind="uniform",
                    lower=0.01,
                    upper=0.1,
                ),
            )
        ),
        InputExecutionContext({}),
        InputResolutionContext({}),
    )
    assert runner is not None
