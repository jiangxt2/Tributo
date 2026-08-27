"""Contract tests for formal distributed-training declarations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    CollectivePolicy,
    DistributionSpec,
    DistributionStrategy,
    ExecutionProfile,
    FrameworkNativePolicy,
    InputDistribution,
    MapReducePolicy,
    MetricReduction,
    StateCoordination,
    StateField,
    WorkerRange,
    WorkerResources,
)


def _collective_spec() -> DistributionSpec:
    return DistributionSpec(
        strategy=DistributionStrategy.RAY_TRAIN_COLLECTIVE,
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=2),
        input_distribution=InputDistribution.SHARDED,
        state_coordination=StateCoordination.ALL_REDUCE,
        policy=CollectivePolicy(
            backend="gloo",
            metric_reducers={
                "loss": MetricReduction.SUM_COUNT,
                "accuracy": MetricReduction.SUM_COUNT,
            },
        ),
    )


def _map_reduce_spec() -> DistributionSpec:
    return DistributionSpec(
        strategy=DistributionStrategy.RAY_MAP_REDUCE,
        supported_worker_range=WorkerRange(1, 32),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(),
        input_distribution=InputDistribution.SHARDED,
        state_coordination=StateCoordination.ASSOCIATIVE_REDUCE,
        policy=MapReducePolicy(
            state_schema=(
                StateField("class_count", "float64", (None,)),
                StateField("feature_count", "float64", (None, None)),
            ),
            max_partial_state_bytes=8 * 1024 * 1024,
            reducer_ref="tests.support.decomposition_algorithms:merge_states",
            finalizer_ref="tests.support.decomposition_algorithms:finalize_model",
        ),
    )


@pytest.mark.parametrize("spec", [_collective_spec(), _map_reduce_spec()])
def test_distribution_spec_json_round_trip_is_deterministic(
    spec: DistributionSpec,
) -> None:
    restored = DistributionSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.digest == spec.digest
    assert restored.to_dict() == spec.to_dict()


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("policy", "same_world_size_resume"), "false"),
        (("policy", "checkpoint_owner_rank"), "0"),
        (("resources_per_worker", "num_cpus"), "1"),
    ],
)
def test_distribution_spec_round_trip_rejects_type_coercion(
    path: tuple[str, str], invalid: object
) -> None:
    payload = _collective_spec().to_dict()
    nested = payload[path[0]]
    assert isinstance(nested, dict)
    nested[path[1]] = invalid

    with pytest.raises(AlgorithmConfigurationError):
        DistributionSpec.from_dict(payload)


def test_distribution_spec_rejects_unknown_api_version() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="api_version"):
        replace(_collective_spec(), api_version=2)


def test_distribution_spec_rejects_strategy_policy_mismatch() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="CollectivePolicy"):
        replace(_collective_spec(), policy=_map_reduce_spec().policy)


def test_framework_native_requires_framework_owned_data_and_evidence() -> None:
    policy = FrameworkNativePolicy(
        framework="external-framework",
        evidence_collector_ref="tests.example:collect_evidence",
    )
    with pytest.raises(AlgorithmConfigurationError, match="framework_owned"):
        DistributionSpec(
            strategy=DistributionStrategy.FRAMEWORK_NATIVE,
            supported_worker_range=WorkerRange(1, 16),
            supported_execution_profiles=(ExecutionProfile.CLUSTER,),
            resources_per_worker=WorkerResources(),
            input_distribution=InputDistribution.SHARDED,
            state_coordination=StateCoordination.FRAMEWORK_NATIVE,
            policy=policy,
        )


def test_framework_native_component_stages_are_canonical_and_round_trip() -> None:
    policy = FrameworkNativePolicy(
        framework="external-multistage",
        evidence_collector_ref="tests.example:collect",
        component_stages=("mu0", "mu1", "tau0", "tau1", "propensity"),
    )
    spec = DistributionSpec(
        strategy=DistributionStrategy.FRAMEWORK_NATIVE,
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(ExecutionProfile.LOCAL,),
        resources_per_worker=WorkerResources(),
        input_distribution=InputDistribution.FRAMEWORK_OWNED,
        state_coordination=StateCoordination.FRAMEWORK_NATIVE,
        policy=policy,
    )

    restored = DistributionSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.policy.component_stages == policy.component_stages


def test_empty_framework_component_stages_preserve_legacy_canonical_payload() -> None:
    spec = DistributionSpec(
        strategy=DistributionStrategy.FRAMEWORK_NATIVE,
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(ExecutionProfile.LOCAL,),
        resources_per_worker=WorkerResources(),
        input_distribution=InputDistribution.FRAMEWORK_OWNED,
        state_coordination=StateCoordination.FRAMEWORK_NATIVE,
        policy=FrameworkNativePolicy(
            framework="external-framework",
            evidence_collector_ref="tests.example:collect",
        ),
    )

    payload = spec.to_dict()
    assert "component_stages" not in payload["policy"]
    restored = DistributionSpec.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.digest == spec.digest


def test_worker_resources_are_immutable_and_scale_per_worker() -> None:
    resources = WorkerResources(
        num_cpus=1.5,
        num_gpus=0,
        custom={"accelerator_type_cpu": 0.25},
    )

    assert resources.scaled(4).to_dict() == {
        "num_cpus": 6.0,
        "num_gpus": 0.0,
        "custom": {"accelerator_type_cpu": 1.0},
    }
    with pytest.raises(TypeError):
        resources.custom["other"] = 1.0


def test_standalone_is_not_an_execution_profile() -> None:
    with pytest.raises(ValueError):
        ExecutionProfile("standalone")


def test_legacy_kubernetes_profile_deserializes_to_cluster() -> None:
    payload = _collective_spec().to_dict()
    payload["supported_execution_profiles"] = ["local", "kubernetes"]

    with pytest.deprecated_call(match="use 'cluster'"):
        restored = DistributionSpec.from_dict(payload)

    assert restored.supported_execution_profiles == (
        ExecutionProfile.CLUSTER,
        ExecutionProfile.LOCAL,
    )
    assert restored.to_dict()["supported_execution_profiles"] == [
        "cluster",
        "local",
    ]
