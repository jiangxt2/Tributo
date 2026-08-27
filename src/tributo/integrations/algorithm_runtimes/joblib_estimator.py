"""Ray Joblib Runtime for estimators with internal independent fit tasks."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    DistributionStrategy,
    JoblibEstimatorPolicy,
    ResolvedAlgorithmPlan,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.spi import (
    JoblibEstimatorRecipe,
    MaterializedTabularInputView,
    RuntimeExecutionEnvelope,
    WorkerInputPayload,
    WorkerInputPayloadSet,
)
from tributo.integrations.algorithm_runtimes.decomposition import (
    actual_versions,
    codec_payload,
    execution_result,
    load_algorithm,
    prepare_input,
    runtime_identity,
)
from tributo.util.annotations import DeveloperAPI

RAY_JOBLIB_ESTIMATOR_RUNTIME_ID = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_JOBLIB_ESTIMATOR
].runtime_id


@dataclass(frozen=True)
class _JoblibFitResult:
    execution: AlgorithmExecutionResult
    actual_versions: Mapping[str, str]
    model_digest: str
    row_count: int
    task_id: str
    job_id: str
    fit_operations: tuple[str, ...]


def _policy(plan: ResolvedAlgorithmPlan) -> JoblibEstimatorPolicy:
    plan.validate_integrity()
    spec = plan.distribution_spec
    if (
        spec is None
        or spec.strategy is not DistributionStrategy.RAY_JOBLIB_ESTIMATOR
        or not isinstance(spec.policy, JoblibEstimatorPolicy)
    ):
        raise AlgorithmConfigurationError(
            "RayJoblibEstimatorRuntime requires a ray_joblib_estimator policy"
        )
    return spec.policy


def _row_count(inputs: Mapping[str, object], primary_role: str) -> int:
    try:
        view = inputs[primary_role]
    except KeyError as exc:
        raise AlgorithmInputError(
            "Joblib estimator input is missing its primary role"
        ) from exc
    if not isinstance(view, MaterializedTabularInputView):
        raise AlgorithmInputError(
            "Joblib estimator requires a materialized tabular primary input"
        )
    if view.row_count < 1:
        raise AlgorithmInputError("Joblib estimator input must not be empty")
    return view.row_count


def _validate_parallelism_contract(
    recipe: JoblibEstimatorRecipe,
    policy: JoblibEstimatorPolicy,
) -> tuple[str, ...]:
    contract = recipe.parallelism_contract()
    if not isinstance(contract, Mapping):
        raise AlgorithmConfigurationError(
            "Joblib parallelism_contract must return a mapping"
        )
    operations = contract.get("fit_operations")
    if not isinstance(operations, (tuple, list)) or any(
        not isinstance(item, str) or not item for item in operations
    ):
        raise AlgorithmConfigurationError(
            "Joblib parallelism contract requires fit_operations"
        )
    normalized = tuple(operations)
    if normalized != policy.fit_operations:
        raise AlgorithmConfigurationError(
            "Joblib recipe fit_operations conflict with DistributionSpec"
        )
    exactness = contract.get("exactness")
    if exactness != policy.exactness.value:
        raise AlgorithmConfigurationError(
            "Joblib recipe exactness conflicts with DistributionSpec"
        )
    return normalized


@ray.remote(num_cpus=0, max_retries=0)
def _fit_estimator(
    plan: ResolvedAlgorithmPlan,
    payload: WorkerInputPayload | WorkerInputPayloadSet,
    run_id: str,
    artifacts: tuple[ArtifactDraft, ...],
) -> _JoblibFitResult:
    from joblib import parallel_backend
    from ray.util.joblib import register_ray

    policy = _policy(plan)
    recipe = load_algorithm(plan, JoblibEstimatorRecipe, artifacts)
    operations = _validate_parallelism_contract(recipe, policy)
    prepared = prepare_input(plan, payload)
    try:
        rows = _row_count(prepared.views, plan.input_bindings.primary_role)
        if rows > policy.max_materialized_rows:
            raise AlgorithmInputError(
                "Joblib estimator input exceeds max_materialized_rows: "
                f"actual={rows}, limit={policy.max_materialized_rows}"
            )
        estimator = recipe.build_estimator(plan.algorithm_config)
        fit = getattr(estimator, "fit", None)
        set_params = getattr(estimator, "set_params", None)
        if not callable(fit) or not callable(set_params):
            raise AlgorithmConfigurationError(
                "Joblib recipe must build an estimator with fit() and set_params()"
            )
        try:
            set_params(**{policy.n_jobs_parameter: plan.runtime.worker_count})
        except Exception as exc:
            raise AlgorithmConfigurationError(
                "Joblib estimator rejected the declared n_jobs parameter"
            ) from exc
        arguments = recipe.fit_arguments(prepared.views, plan.algorithm_config)
        if (
            not isinstance(arguments, tuple)
            or len(arguments) != 2
            or not isinstance(arguments[0], tuple)
            or not isinstance(arguments[1], Mapping)
        ):
            raise AlgorithmConfigurationError(
                "Joblib fit_arguments must return (tuple, mapping)"
            )
        register_ray()
        try:
            with parallel_backend(
                "ray",
                n_jobs=plan.runtime.worker_count,
                ray_remote_args={
                    "num_cpus": plan.runtime.num_cpus,
                    "num_gpus": plan.runtime.num_gpus,
                    "resources": dict(plan.runtime.custom_resources),
                    "scheduling_strategy": "SPREAD",
                },
            ):
                fitted = fit(*arguments[0], **dict(arguments[1]))
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"Joblib estimator fit failed: {type(exc).__name__}"
            ) from exc
        model = recipe.extract_model(fitted)
        _, model_digest = codec_payload(
            recipe.model_codec(),
            model,
            max_bytes=64 * 1024 * 1024,
        )
        identity = runtime_identity()
        execution = execution_result(
            model=model,
            plan=plan,
            run_id=run_id,
            metrics={"row_count": rows},
        )
        return _JoblibFitResult(
            execution=execution,
            actual_versions=actual_versions(plan),
            model_digest=model_digest,
            row_count=rows,
            task_id=identity["task_id"],
            job_id=identity["job_id"],
            fit_operations=operations,
        )
    finally:
        prepared.close()


def _joblib_workers(
    plan: ResolvedAlgorithmPlan,
    result: _JoblibFitResult,
) -> tuple[tuple[WorkerExecutionEvidence, ...], int]:
    from ray.util import state

    deadline = time.monotonic() + 10.0
    matching: list[Any] = []
    while time.monotonic() < deadline:
        tasks = state.list_tasks(
            filters=[("job_id", "=", result.job_id)],
            detail=True,
            limit=10_000,
            timeout=5,
            raise_on_missing_output=False,
        )
        matching = [
            task
            for task in tasks
            if task.parent_task_id == result.task_id
            and task.func_or_class_name == "PoolActor.run_batch"
            and isinstance(task.worker_id, str)
            and task.worker_id
            and isinstance(task.node_id, str)
            and task.node_id
            and task.state not in {"FAILED", "NIL"}
        ]
        if len({task.worker_id for task in matching}) >= plan.runtime.worker_count:
            break
        time.sleep(0.1)
    identities: dict[str, str] = {}
    for task in matching:
        identities.setdefault(task.worker_id, task.node_id)
    if len(identities) != plan.runtime.worker_count:
        raise AlgorithmExecutionError(
            "Ray Joblib evidence did not observe the requested distinct Workers: "
            f"requested={plan.runtime.worker_count}, observed={len(identities)}"
        )
    resources = WorkerResources(
        num_cpus=plan.runtime.num_cpus,
        num_gpus=plan.runtime.num_gpus,
        custom=plan.runtime.custom_resources,
    )
    workers = tuple(
        WorkerExecutionEvidence(
            worker_id=worker_id,
            node_id=node_id,
            rank=rank,
            world_size=plan.runtime.worker_count,
            shard_id=hashlib.sha256(
                f"joblib:{result.task_id}:{worker_id}".encode("utf-8")
            ).hexdigest(),
            resources=resources,
            rows_processed=result.row_count,
            input_rows={plan.input_bindings.primary_role: result.row_count},
        )
        for rank, (worker_id, node_id) in enumerate(sorted(identities.items()))
    )
    return workers, len(matching)


@DeveloperAPI
class RayJoblibEstimatorRuntime:
    """Execute estimator-internal Joblib tasks without Driver data materialization."""

    @property
    def runtime_id(self) -> str:
        """Return the formal runtime identity."""
        return RAY_JOBLIB_ESTIMATOR_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Fit one estimator and bind observed Joblib tasks to the final model."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before Joblib estimator execution"
            )
        if envelope.cancelled:
            raise AlgorithmExecutionError("Joblib estimator execution was cancelled")
        _policy(envelope.plan)
        if len(envelope.input_payloads) != 1:
            raise AlgorithmInputError(
                "Joblib estimator requires one unsplit full-dataset payload"
            )
        try:
            result = ray.get(
                _fit_estimator.remote(
                    envelope.plan,
                    envelope.input_payloads[0],
                    envelope.run_id,
                    envelope.artifacts,
                )
            )
            workers, task_count = _joblib_workers(envelope.plan, result)
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"Ray Joblib estimator execution failed: {type(exc).__name__}"
            ) from exc
        policy = _policy(envelope.plan)
        return WorkerExecutionResult(
            execution=result.execution,
            actual_versions=result.actual_versions,
            worker_metadata={
                "topology": "ray_joblib_estimator",
                "workers": [worker.to_dict() for worker in workers],
                "state": {
                    "coordination": "estimator_internal",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": result.model_digest,
                    "details": {
                        "execution_capability": "estimator_internal_parallel",
                        "fit_operations": ",".join(result.fit_operations),
                        "exactness": policy.exactness.value,
                        "joblib_task_count": task_count,
                        "observed_worker_count": len(workers),
                    },
                },
                "input_complete": True,
                "driver_materialized_training_rows": 0,
            },
        )


__all__ = ["RayJoblibEstimatorRuntime"]
