"""Ray tree-MapReduce Runtime Adapter for bounded model sufficient statistics."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    DistributionStrategy,
    MapReducePolicy,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    StateField,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
    WorkerResources,
)
from tributo.algorithms.core.worker import (
    _actual_environment_versions,
    _load_reference,
    _validate_module_digest,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MapReduceAlgorithm,
    MaterializedTabularInputView,
    PreparedInput,
    RuntimeExecutionEnvelope,
    TabularBatchInputView,
    WorkerInputPayload,
)
from tributo.util.annotations import DeveloperAPI

RAY_MAP_REDUCE_RUNTIME_ID = "tributo.ray_map_reduce"


@dataclass(frozen=True)
class _MapReduceStageResult:
    state: Mapping[str, object]
    state_digest: str
    state_size_bytes: int
    actual_versions: Mapping[str, str]
    workers: tuple[WorkerExecutionEvidence, ...]
    expected_total_rows: int
    reduction_workers: tuple[Mapping[str, str], ...] = ()
    tree_depth: int = 0


class _TrackedBatches:
    def __init__(self, batches: Iterable[Mapping[str, object]]) -> None:
        self._batches = batches
        self.batch_count = 0
        self.row_count = 0
        self._iterated = False
        self.exhausted = False

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        if self._iterated:
            raise AlgorithmInputError(
                "MapReduce input batches may be consumed only once"
            )
        self._iterated = True
        for batch in self._batches:
            if not isinstance(batch, Mapping):
                raise AlgorithmInputError(
                    "MapReduce input yielded a non-columnar batch"
                )
            lengths = {_column_length(value) for value in batch.values()}
            if len(lengths) > 1:
                raise AlgorithmInputError(
                    "MapReduce input batch columns have inconsistent row counts"
                )
            self.batch_count += 1
            self.row_count += next(iter(lengths), 0)
            yield batch
        self.exhausted = True


def _column_length(value: object) -> int:
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return shape[0]
    length_method = getattr(value, "__len__", None)
    if not callable(length_method):
        raise AlgorithmInputError(
            "MapReduce input columns must expose a bounded batch length"
        )
    length = length_method()
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise AlgorithmInputError("MapReduce input column length is invalid")
    return length


def _state_values(state: object) -> Mapping[str, object]:
    if not isinstance(state, Mapping):
        raise AlgorithmExecutionError(
            "MapReduce partial state must be a mapping matching state_schema"
        )
    if any(not isinstance(name, str) or not name for name in state):
        raise AlgorithmExecutionError(
            "MapReduce partial state field names must be non-empty strings"
        )
    return state


def _validate_state(
    state: object,
    schema: tuple[StateField, ...],
    max_bytes: int,
) -> tuple[Mapping[str, object], str, int]:
    import numpy as np

    values = _state_values(state)
    expected_names = tuple(field.name for field in schema)
    if set(values) != set(expected_names):
        raise AlgorithmExecutionError(
            "MapReduce partial state does not match the declared field names"
        )
    digest = hashlib.sha256()
    for field in schema:
        value = values[field.name]
        if not isinstance(value, np.ndarray):
            raise AlgorithmExecutionError(
                f"MapReduce state field {field.name!r} must be a NumPy array"
            )
        if value.dtype.name != field.dtype:
            raise AlgorithmExecutionError(
                f"MapReduce state field {field.name!r} has dtype "
                f"{value.dtype.name!r}; expected {field.dtype!r}"
            )
        if len(value.shape) != len(field.shape) or any(
            declared is not None and actual != declared
            for actual, declared in zip(value.shape, field.shape, strict=True)
        ):
            raise AlgorithmExecutionError(
                f"MapReduce state field {field.name!r} has shape {value.shape!r}; "
                f"expected {field.shape!r}"
            )
        contiguous = np.ascontiguousarray(value)
        digest.update(field.name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    size_bytes = len(pickle.dumps(dict(values), protocol=5))
    if size_bytes > max_bytes:
        raise AlgorithmExecutionError(
            "MapReduce partial state exceeds max_partial_state_bytes: "
            f"actual={size_bytes}, limit={max_bytes}"
        )
    return values, digest.hexdigest(), size_bytes


def _policy(plan: ResolvedAlgorithmPlan) -> MapReducePolicy:
    plan.validate_integrity()
    spec = plan.distribution_spec
    if (
        spec is None
        or spec.strategy is not DistributionStrategy.RAY_MAP_REDUCE
        or not isinstance(spec.policy, MapReducePolicy)
    ):
        raise AlgorithmConfigurationError(
            "RayMapReduceRuntime requires a ray_map_reduce DistributionSpec"
        )
    return spec.policy


def _algorithm(
    plan: ResolvedAlgorithmPlan,
    policy: MapReducePolicy,
) -> MapReduceAlgorithm[Any, Any, Any]:
    _validate_module_digest(
        plan.implementation.implementation_ref,
        plan.implementation.code_digest,
    )
    implementation = _load_reference(plan.implementation.implementation_ref)
    factory = _load_reference(plan.implementation.executable_factory_ref)
    if not callable(factory):
        raise AlgorithmConfigurationError(
            "MapReduce executable factory reference is not callable"
        )
    algorithm = factory(plan=plan, implementation=implementation, artifacts=())
    if not isinstance(algorithm, MapReduceAlgorithm):
        raise AlgorithmConfigurationError(
            "MapReduce executable factory must return MapReduceAlgorithm"
        )
    if algorithm.state_schema() != policy.state_schema:
        raise AlgorithmConfigurationError(
            "MapReduce implementation state_schema conflicts with DistributionSpec"
        )
    declared_reducer = _load_reference(QualifiedReference.parse(policy.reducer_ref))
    declared_finalizer = _load_reference(QualifiedReference.parse(policy.finalizer_ref))
    if declared_reducer is not getattr(type(algorithm), "merge_states", None):
        raise AlgorithmConfigurationError(
            "MapReduce reducer_ref does not match the implementation method"
        )
    if declared_finalizer is not getattr(type(algorithm), "finalize_model", None):
        raise AlgorithmConfigurationError(
            "MapReduce finalizer_ref does not match the implementation method"
        )
    if policy.max_retries > 0:
        raise AlgorithmConfigurationError(
            "MapReduce retries require a replayable input contract; the v1 Ray "
            "streaming shard adapter is single-pass"
        )
    return algorithm


def _runtime_identity() -> dict[str, str]:
    context = ray.get_runtime_context()
    return {
        "node_id": str(context.get_node_id()),
        "worker_id": str(context.get_worker_id()),
        "job_id": str(context.get_job_id()),
    }


def _input_batches(prepared: PreparedInput) -> Iterable[Mapping[str, object]]:
    if len(prepared.views) != 1:
        raise AlgorithmInputError(
            "MapReduce training requires exactly one named tabular input"
        )
    view = next(iter(prepared.views.values()))
    if isinstance(view, TabularBatchInputView):
        return view.iter_batches()
    if isinstance(view, MaterializedTabularInputView):
        return (view.columns(),)
    raise AlgorithmInputError(
        "MapReduce input adapter must expose TabularBatchInputView or "
        "MaterializedTabularInputView"
    )


def _validate_partition_row_count(
    plan: ResolvedAlgorithmPlan,
    row_count: int,
) -> None:
    """Reject an empty map shard when the request claims distributed training."""
    if (
        plan.distribution_spec is not None
        and plan.runtime.worker_count >= plan.distribution_spec.distributed_min_workers
        and row_count < 1
    ):
        raise AlgorithmInputError(
            "distributed MapReduce requires every requested map worker to "
            "consume a non-empty input shard"
        )


def _validate_input_coverage(stage: _MapReduceStageResult) -> int:
    """Cross-check runtime-observed map rows against the Driver count."""
    observed = sum(worker.rows_processed or 0 for worker in stage.workers)
    if observed != stage.expected_total_rows:
        raise AlgorithmExecutionError(
            "MapReduce input coverage mismatch: "
            f"observed={observed}, expected={stage.expected_total_rows}"
        )
    return observed


@ray.remote
def _map_partition(
    plan: ResolvedAlgorithmPlan,
    payload: WorkerInputPayload,
    rank: int,
) -> _MapReduceStageResult:
    policy = _policy(plan)
    algorithm = _algorithm(plan, policy)
    versions = _actual_environment_versions(
        plan.environment.python,
        plan.environment.dependencies,
    )
    input_factory = _load_reference(plan.runtime.worker_input_adapter_ref)
    if not callable(input_factory):
        raise AlgorithmConfigurationError(
            "MapReduce Worker input adapter reference is not callable"
        )
    prepared: PreparedInput | None = None
    try:
        if not isinstance(payload, WorkerInputPayload):
            raise AlgorithmInputError("MapReduce runtime requires a WorkerInputPayload")
        if payload.expected_total_rows is None:
            raise AlgorithmInputError(
                "MapReduce input payload is missing expected total rows"
            )
        prepared_value = input_factory(payload)
        if not isinstance(prepared_value, PreparedInput):
            raise AlgorithmInputError(
                "MapReduce Worker input adapter did not return PreparedInput"
            )
        prepared = prepared_value
        identity = _runtime_identity()
        tracked = _TrackedBatches(_input_batches(prepared))
        context = AlgorithmExecutionContext(
            inputs=prepared.views,
            worker_metadata={
                **identity,
                "world_rank": rank,
                "world_size": plan.runtime.worker_count,
            },
        )
        state = algorithm.map_partition(tracked, context)
        if not tracked.exhausted:
            raise AlgorithmExecutionError(
                "MapReduce map_partition must consume its input shard exactly once"
            )
        _validate_partition_row_count(plan, tracked.row_count)
        values, state_digest, state_size = _validate_state(
            state,
            policy.state_schema,
            policy.max_partial_state_bytes,
        )
        shard_id = hashlib.sha256(
            (
                f"{plan.input_descriptor.binding_digest}:"
                f"{rank}/{plan.runtime.worker_count}"
            ).encode("ascii")
        ).hexdigest()
        assigned = ray.get_runtime_context().get_assigned_resources()
        evidence = WorkerExecutionEvidence(
            worker_id=identity["worker_id"],
            node_id=identity["node_id"],
            rank=rank,
            world_size=plan.runtime.worker_count,
            shard_id=shard_id,
            resources=WorkerResources(
                num_cpus=float(assigned.get("CPU", 0.0)),
                num_gpus=float(assigned.get("GPU", 0.0)),
                custom={
                    str(name): float(value)
                    for name, value in assigned.items()
                    if name not in {"CPU", "GPU", "memory", "object_store_memory"}
                },
            ),
            model_state_digest=state_digest,
            rows_processed=tracked.row_count,
        )
        return _MapReduceStageResult(
            state=values,
            state_digest=state_digest,
            state_size_bytes=state_size,
            actual_versions=versions,
            workers=(evidence,),
            expected_total_rows=payload.expected_total_rows,
        )
    finally:
        if prepared is not None:
            prepared.close()


@ray.remote
def _merge_pair(
    plan: ResolvedAlgorithmPlan,
    left: _MapReduceStageResult,
    right: _MapReduceStageResult,
) -> _MapReduceStageResult:
    policy = _policy(plan)
    algorithm = _algorithm(plan, policy)
    if dict(left.actual_versions) != dict(right.actual_versions):
        raise AlgorithmExecutionError(
            "MapReduce stages loaded inconsistent dependency versions"
        )
    if left.expected_total_rows != right.expected_total_rows:
        raise AlgorithmExecutionError(
            "MapReduce stages disagree on expected total rows"
        )
    state = algorithm.merge_states(left.state, right.state)
    values, state_digest, state_size = _validate_state(
        state,
        policy.state_schema,
        policy.max_partial_state_bytes,
    )
    identity = _runtime_identity()
    return _MapReduceStageResult(
        state=values,
        state_digest=state_digest,
        state_size_bytes=state_size,
        actual_versions=left.actual_versions,
        workers=left.workers + right.workers,
        expected_total_rows=left.expected_total_rows,
        reduction_workers=(
            left.reduction_workers + right.reduction_workers + (identity,)
        ),
        tree_depth=max(left.tree_depth, right.tree_depth) + 1,
    )


@ray.remote(num_cpus=0, max_retries=0)
def _finalize_model(
    plan: ResolvedAlgorithmPlan,
    stage: _MapReduceStageResult,
    run_id: str,
) -> WorkerExecutionResult:
    policy = _policy(plan)
    algorithm = _algorithm(plan, policy)
    observed_input_rows = _validate_input_coverage(stage)
    model = algorithm.finalize_model(stage.state)
    exporter_ref = plan.implementation.exporter_ref
    if exporter_ref is None:
        raise AlgorithmConfigurationError(
            "MapReduce fit requires an explicit exporter reference"
        )
    exporter = _load_reference(exporter_ref)
    if not callable(exporter):
        raise AlgorithmConfigurationError(
            "MapReduce exporter reference is not callable"
        )
    execution = exporter(model=model, plan=plan, run_id=run_id)
    if not isinstance(execution, AlgorithmExecutionResult):
        raise AlgorithmExecutionError(
            "MapReduce exporter must return AlgorithmExecutionResult"
        )
    if (
        plan.distribution_spec is not None
        and plan.distribution_spec.result_policy is ResultPolicy.BUNDLE_REQUIRED
        and not execution.outputs.get("bundle_uri")
    ):
        raise AlgorithmExecutionError(
            "MapReduce fit completed without the required Bundle publication"
        )
    identity = _runtime_identity()
    worker_metadata: dict[str, Any] = {
        "topology": "ray_map_reduce",
        "workers": [item.to_dict() for item in stage.workers],
        "state": {
            "coordination": "associative_reduce",
            "synchronized": True,
            "bounded": True,
            "global_model_digest": stage.state_digest,
            "details": {
                "tree_depth": stage.tree_depth,
                "reduce_task_count": len(stage.reduction_workers),
                "partial_state_bytes": stage.state_size_bytes,
                "max_partial_state_bytes": policy.max_partial_state_bytes,
                "commutative": policy.commutative,
                "expected_input_rows": stage.expected_total_rows,
                "observed_input_rows": observed_input_rows,
            },
        },
        "input_complete": True,
        "driver_materialized_training_rows": 0,
        "reduction_workers": list(stage.reduction_workers),
        "finalizer_worker": identity,
    }
    return WorkerExecutionResult(
        execution=execution,
        actual_versions=stage.actual_versions,
        worker_metadata=worker_metadata,
    )


def _tree_reduce(
    plan: ResolvedAlgorithmPlan,
    references: list[ray.ObjectRef],
    *,
    num_cpus: float,
    num_gpus: float,
    custom_resources: Mapping[str, float],
    max_retries: int,
) -> ray.ObjectRef:
    """Build a balanced pairwise reduction DAG without resolving states on Driver."""
    current = list(references)
    while len(current) > 1:
        next_level: list[ray.ObjectRef] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
                continue
            next_level.append(
                _merge_pair.options(
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    resources=dict(custom_resources),
                    max_retries=max_retries,
                    retry_exceptions=False,
                ).remote(plan, current[index], current[index + 1])
            )
        current = next_level
    return current[0]


@DeveloperAPI
class RayMapReduceRuntime:
    """Execute bounded sufficient-statistics training as a real Ray DAG."""

    @property
    def runtime_id(self) -> str:
        """Return the formal runtime identity."""
        return RAY_MAP_REDUCE_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Submit map, balanced reduce, and finalizer tasks to initialized Ray."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before MapReduce algorithm execution"
            )
        if envelope.cancelled:
            raise AlgorithmExecutionError("MapReduce execution was cancelled")
        policy = _policy(envelope.plan)
        try:
            references = [
                _map_partition.options(
                    num_cpus=envelope.plan.runtime.num_cpus,
                    num_gpus=envelope.plan.runtime.num_gpus,
                    resources=dict(envelope.plan.runtime.custom_resources),
                    scheduling_strategy="SPREAD",
                    max_retries=policy.max_retries,
                    retry_exceptions=False,
                ).remote(envelope.plan, payload, rank)
                for rank, payload in enumerate(envelope.input_payloads)
            ]
            final_state = _tree_reduce(
                envelope.plan,
                references,
                num_cpus=envelope.plan.runtime.num_cpus,
                num_gpus=envelope.plan.runtime.num_gpus,
                custom_resources=envelope.plan.runtime.custom_resources,
                max_retries=policy.max_retries,
            )
            result = ray.get(
                _finalize_model.options(
                    num_cpus=envelope.plan.runtime.num_cpus,
                    num_gpus=envelope.plan.runtime.num_gpus,
                    resources=dict(envelope.plan.runtime.custom_resources),
                ).remote(envelope.plan, final_state, envelope.run_id)
            )
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"Ray MapReduce execution failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(result, WorkerExecutionResult):
            raise AlgorithmExecutionError(
                "Ray MapReduce finalizer returned an invalid result"
            )
        return result


__all__ = ["RayMapReduceRuntime"]
