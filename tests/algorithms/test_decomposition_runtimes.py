"""Execution tests for Core-owned single-model decomposition runtimes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import ray

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ExecutionRequest,
    ImplementationDescriptor,
    InputDistribution,
    IterativeOptimizationPolicy,
    JoblibEstimatorPolicy,
    ParallelEnsemblePolicy,
    QualifiedReference,
    ResultPolicy,
    RuntimeTopology,
    StateCoordination,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.core import (
    AlgorithmPlanner,
    AlgorithmRegistrationRegistry,
    AlgorithmRunCoordinator,
)
from tributo.algorithms.input import (
    FakeInputInvocation,
    FakeInputResolver,
    FakeInputRuntimeAdapter,
    FakeTabularPayload,
)
from tributo.algorithms.spi import InputExecutionContext
from tributo.integrations.algorithm_runtimes.iterative_optimization import (
    RayIterativeOptimizationRuntime,
)
from tributo.integrations.algorithm_runtimes.joblib_estimator import (
    RayJoblibEstimatorRuntime,
)
from tributo.integrations.algorithm_runtimes.parallel_ensemble import (
    RayParallelUnitRuntime,
)

from .conftest import make_spec, request_for

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


@pytest.fixture(scope="module", autouse=True)
def local_ray() -> Iterator[None]:
    """Run the decomposition DAGs on real local Ray processes."""
    ray.init(num_cpus=4, include_dashboard=True, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def _compatibility(topology: RuntimeTopology) -> BackendInputCompatibility:
    return BackendInputCompatibility(
        accepted_input_views=("materialized_tabular",),
        accepted_ingestion_engines=("tributo.fake_tabular",),
        required_input_capabilities=("materializable", "shardable"),
        supported_explicit_adapters=(
            QualifiedReference.parse("tributo.algorithms.input.fake:prepare_input"),
        ),
        distribution_policy=(topology,),
    )


def _registration(
    strategy: DistributionStrategy,
    *,
    max_rounds: int = 4,
) -> AlgorithmRegistration:
    if strategy is DistributionStrategy.RAY_JOBLIB_ESTIMATOR:
        mode = ExecutionMode.JOBLIB_ESTIMATOR
        topology = RuntimeTopology.RAY_JOBLIB_ESTIMATOR
        implementation = "tests.support.decomposition_algorithms:JoblibProbeRecipe"
        runtime_id = "tributo.ray_joblib_estimator"
        input_distribution = InputDistribution.FULL_DATASET
        coordination = StateCoordination.ESTIMATOR_INTERNAL
        policy = JoblibEstimatorPolicy(max_materialized_rows=64)
        allowed_keys = ("task_count",)
    elif strategy is DistributionStrategy.RAY_PARALLEL_ENSEMBLE:
        mode = ExecutionMode.PARALLEL_ENSEMBLE
        topology = RuntimeTopology.RAY_PARALLEL_ENSEMBLE
        implementation = (
            "tests.support.decomposition_algorithms:ParallelThresholdEnsemble"
        )
        runtime_id = "tributo.ray_parallel_ensemble"
        input_distribution = InputDistribution.FULL_DATASET
        coordination = StateCoordination.ORDERED_ENSEMBLE
        policy = ParallelEnsemblePolicy(max_units=8)
        allowed_keys = ("runtime", "seed", "unit_count")
    else:
        mode = ExecutionMode.ITERATIVE_OPTIMIZATION
        topology = RuntimeTopology.RAY_ITERATIVE_OPTIMIZATION
        implementation = (
            "tests.support.decomposition_algorithms:BinaryLogisticOptimization"
        )
        runtime_id = "tributo.ray_iterative_optimization"
        input_distribution = InputDistribution.SHARDED
        coordination = StateCoordination.ITERATIVE_GLOBAL
        policy = IterativeOptimizationPolicy(max_rounds=max_rounds)
        allowed_keys = ("feature_count", "runtime")
    name = strategy.value
    return AlgorithmRegistration(
        spec=make_spec(name, operations=("fit",), mode=mode),
        implementation=ImplementationDescriptor(
            implementation_id=f"tests.{name}",
            version="1.0.0",
            execution_mode=mode,
            implementation_ref=QualifiedReference.parse(implementation),
            executable_factory_ref=QualifiedReference.parse(
                "tributo.integrations.algorithm_runtimes.decomposition:create_algorithm"
            ),
            operations=(AlgorithmOperation.FIT,),
            input_compatibility=_compatibility(topology),
            allowed_config_keys=allowed_keys,
            runtime_id=runtime_id,
            worker_input_adapter_ref=QualifiedReference.parse(
                "tributo.algorithms.input.fake:prepare_input"
            ),
        ),
        environment=EnvironmentSpec(environment_id=f"tests.{name}"),
        distribution_spec=DistributionSpec(
            strategy=strategy,
            supported_worker_range=WorkerRange(1, 4),
            supported_execution_profiles=(ExecutionProfile.LOCAL,),
            resources_per_worker=WorkerResources(num_cpus=0),
            input_distribution=input_distribution,
            state_coordination=coordination,
            policy=policy,
            result_policy=ResultPolicy.FIT_ONLY,
        ),
        is_default=True,
    )


def _execute(
    strategy: DistributionStrategy,
    runtime: object,
    *,
    config: dict[str, object],
    max_rounds: int = 4,
    resume_from: str | None = None,
):
    registration = _registration(strategy, max_rounds=max_rounds)
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})
    plan = planner.plan(
        ExecutionRequest(
            algorithm_request=request_for(
                strategy.value,
                AlgorithmOperation.FIT,
                config=config,
            ),
            profile=ExecutionProfile.LOCAL,
            worker_count=2,
            resume_from=resume_from,
        )
    )
    coordinator = AlgorithmRunCoordinator(
        resolvers={resolver.resolver_id: resolver},
        input_adapters={resolver.resolver_id: FakeInputRuntimeAdapter()},
        runtimes={runtime.runtime_id: runtime},
    )
    return coordinator.execute(
        plan,
        InputExecutionContext(
            {
                "binary-fixture": FakeInputInvocation(
                    FakeTabularPayload(
                        {
                            "x0": (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0),
                            "x1": (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0),
                            "label": (0, 0, 0, 0, 1, 1, 1, 1),
                        }
                    )
                )
            }
        ),
    )


@pytest.mark.slow
def test_joblib_runtime_observes_estimator_internal_workers() -> None:
    result = _execute(
        DistributionStrategy.RAY_JOBLIB_ESTIMATOR,
        RayJoblibEstimatorRuntime(),
        config={"task_count": 4},
    )

    receipt = result.execution_receipt
    assert receipt is not None
    assert receipt.distributed is True
    assert receipt.execution_capability == "estimator_internal_parallel"
    assert len(receipt.workers) == 2
    assert receipt.state.details["joblib_task_count"] >= 2
    assert receipt.driver_materialized_training_rows == 0


@pytest.mark.slow
def test_parallel_ensemble_runtime_proves_unit_coverage() -> None:
    result = _execute(
        DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
        RayParallelUnitRuntime(),
        config={"seed": 7, "unit_count": 4},
    )

    receipt = result.execution_receipt
    assert receipt is not None
    assert receipt.distributed is True
    assert receipt.execution_capability == "single_model_distributed"
    assert receipt.state.details["unit_count"] == 4
    assert len({worker.worker_id for worker in receipt.workers}) == 2


@pytest.mark.slow
def test_parallel_ensemble_checkpoint_resumes_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ensemble-checkpoint"
    config = {
        "runtime": {"checkpoint_dir": str(checkpoint)},
        "seed": 7,
        "unit_count": 4,
    }
    first = _execute(
        DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
        RayParallelUnitRuntime(),
        config=config,
    )
    resumed = _execute(
        DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
        RayParallelUnitRuntime(),
        config=config,
        resume_from=str(checkpoint),
    )

    assert first.execution_receipt is not None
    assert resumed.execution_receipt is not None
    assert resumed.execution_receipt.state.details["resumed"] is True
    assert resumed.execution_receipt.state.details["restored_unit_count"] == 4
    assert (checkpoint / "manifest.json").is_file()

    (checkpoint / "workers" / "rank-0.bin").write_bytes(b"corrupted")
    with pytest.raises(AlgorithmExecutionError, match="digest mismatch"):
        _execute(
            DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
            RayParallelUnitRuntime(),
            config=config,
            resume_from=str(checkpoint),
        )


@pytest.mark.slow
def test_iterative_runtime_proves_round_coverage_and_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "logistic-checkpoint"
    result = _execute(
        DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
        RayIterativeOptimizationRuntime(),
        config={
            "feature_count": 2,
            "runtime": {"checkpoint_dir": str(checkpoint)},
        },
    )

    receipt = result.execution_receipt
    assert receipt is not None
    assert receipt.distributed is True
    assert receipt.state.details["rounds_completed"] == 4
    assert receipt.state.details["expected_input_rows"] == 8
    assert receipt.state.details["observed_input_rows"] == 8
    assert (checkpoint / "manifest.json").is_file()
    assert (checkpoint / "state.bin").is_file()


@pytest.mark.slow
def test_iterative_checkpoint_resumes_and_rejects_corruption(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iterative-resume"
    config = {
        "feature_count": 2,
        "runtime": {"checkpoint_dir": str(checkpoint)},
    }
    _execute(
        DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
        RayIterativeOptimizationRuntime(),
        config=config,
        max_rounds=2,
    )
    resumed = _execute(
        DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
        RayIterativeOptimizationRuntime(),
        config=config,
        max_rounds=4,
        resume_from=str(checkpoint),
    )

    assert resumed.execution_receipt is not None
    assert resumed.execution_receipt.state.details["resumed"] is True
    assert resumed.execution_receipt.state.details["rounds_completed"] == 4

    (checkpoint / "state.bin").write_bytes(b"corrupted")
    with pytest.raises(AlgorithmExecutionError, match="Ray iterative optimization"):
        _execute(
            DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
            RayIterativeOptimizationRuntime(),
            config=config,
            max_rounds=6,
            resume_from=str(checkpoint),
        )
