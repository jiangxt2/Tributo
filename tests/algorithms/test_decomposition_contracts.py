"""Contract tests for Core-owned algorithm decomposition strategies."""

from __future__ import annotations

import pytest

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    InputDistribution,
    IterativeOptimizationPolicy,
    JoblibEstimatorPolicy,
    ParallelEnsemblePolicy,
    ResultPolicy,
    StateCoordination,
    WorkerRange,
    WorkerResources,
)

from .conftest import make_spec


@pytest.mark.parametrize(
    ("strategy", "input_distribution", "coordination", "policy"),
    [
        (
            DistributionStrategy.RAY_JOBLIB_ESTIMATOR,
            InputDistribution.FULL_DATASET,
            StateCoordination.ESTIMATOR_INTERNAL,
            JoblibEstimatorPolicy(max_materialized_rows=1024),
        ),
        (
            DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
            InputDistribution.FULL_DATASET,
            StateCoordination.ORDERED_ENSEMBLE,
            ParallelEnsemblePolicy(max_units=16, max_retries=1),
        ),
        (
            DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
            InputDistribution.SHARDED,
            StateCoordination.ITERATIVE_GLOBAL,
            IterativeOptimizationPolicy(max_rounds=20, max_retries=1),
        ),
    ],
)
def test_decomposition_distribution_spec_round_trip(
    strategy: DistributionStrategy,
    input_distribution: InputDistribution,
    coordination: StateCoordination,
    policy: object,
) -> None:
    spec = DistributionSpec(
        strategy=strategy,
        supported_worker_range=WorkerRange(1, 8),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=1),
        input_distribution=input_distribution,
        state_coordination=coordination,
        policy=policy,
        result_policy=ResultPolicy.FIT_ONLY,
    )

    restored = DistributionSpec.from_dict(spec.to_dict())

    assert restored == spec
    assert restored.digest == spec.digest


def _environment() -> EnvironmentSpec:
    return EnvironmentSpec(
        environment_id="example.decomposition.v1",
        dependencies=("example-algorithms==1.0.0",),
    )


def test_joblib_builder_selects_full_dataset_runtime() -> None:
    descriptor = AlgorithmBuilder.from_joblib_estimator_recipe(
        spec=make_spec(
            "example_joblib",
            operations=("fit",),
            mode=ExecutionMode.JOBLIB_ESTIMATOR,
        ),
        implementation_id="example.joblib.random_forest",
        implementation_version="1.0.0",
        recipe="example_algorithms:RandomForestRecipe",
        environment=_environment(),
        allowed_config_keys=("n_estimators",),
        supported_worker_range=WorkerRange(2, 8),
        supported_execution_profiles=(ExecutionProfile.CLUSTER,),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name="example-algorithms",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        result_policy=ResultPolicy.FIT_ONLY,
        descriptor_api_version=1,
    )

    registration = descriptor.registration
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy
        is DistributionStrategy.RAY_JOBLIB_ESTIMATOR
    )
    assert registration.implementation.runtime_id == "tributo.ray_joblib_estimator"
    assert (
        registration.distribution_spec.input_distribution
        is InputDistribution.FULL_DATASET
    )


def test_ensemble_and_iterative_builders_select_distinct_runtimes() -> None:
    ensemble = AlgorithmBuilder.from_parallel_ensemble(
        spec=make_spec(
            "example_ensemble",
            operations=("fit",),
            mode=ExecutionMode.PARALLEL_ENSEMBLE,
        ),
        implementation_id="example.ensemble.random_forest",
        implementation_version="1.0.0",
        algorithm="example_algorithms:RandomForestEnsemble",
        environment=_environment(),
        allowed_config_keys=("n_estimators",),
        supported_worker_range=WorkerRange(2, 8),
        supported_execution_profiles=(ExecutionProfile.CLUSTER,),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name="example-algorithms",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        policy=ParallelEnsemblePolicy(max_units=1024),
        result_policy=ResultPolicy.FIT_ONLY,
        descriptor_api_version=1,
    )
    iterative = AlgorithmBuilder.from_iterative_optimization(
        spec=make_spec(
            "example_iterative",
            operations=("fit",),
            mode=ExecutionMode.ITERATIVE_OPTIMIZATION,
        ),
        implementation_id="example.iterative.logistic",
        implementation_version="1.0.0",
        algorithm="example_algorithms:LogisticOptimization",
        environment=_environment(),
        allowed_config_keys=("max_rounds",),
        supported_worker_range=WorkerRange(2, 8),
        supported_execution_profiles=(ExecutionProfile.CLUSTER,),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name="example-algorithms",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        policy=IterativeOptimizationPolicy(max_rounds=100),
        result_policy=ResultPolicy.FIT_ONLY,
        descriptor_api_version=1,
    )

    assert ensemble.registration.implementation.runtime_id == (
        "tributo.ray_parallel_ensemble"
    )
    assert iterative.registration.implementation.runtime_id == (
        "tributo.ray_iterative_optimization"
    )
