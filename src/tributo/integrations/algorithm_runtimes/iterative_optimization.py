"""Synchronous Ray Runtime for bounded iterative optimization algorithms."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    DistributionStrategy,
    IterativeOptimizationPolicy,
    ResolvedAlgorithmPlan,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    IterativeOptimizationAlgorithm,
    MaterializedTabularInputView,
    RuntimeExecutionEnvelope,
    TabularBatchInputView,
    WorkerInputPayload,
)
from tributo.integrations.algorithm_runtimes.decomposition import (
    actual_versions,
    assigned_resources,
    codec_payload,
    execution_result,
    load_algorithm,
    prepare_input,
    runtime_identity,
    serialized,
)
from tributo.training.checkpoint import (
    materialize_checkpoint_directory,
    publish_checkpoint_directory,
)
from tributo.util.annotations import DeveloperAPI

RAY_ITERATIVE_OPTIMIZATION_RUNTIME_ID = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION
].runtime_id


@dataclass(frozen=True)
class _UpdateResult:
    update: object
    digest: str
    size_bytes: int
    workers: tuple[WorkerExecutionEvidence, ...]
    actual_versions: Mapping[str, str]
    expected_total_rows: int
    observed_rows: int
    tree_depth: int = 0


@dataclass(frozen=True)
class _RoundSummary:
    round_index: int
    state_digest: str
    state_size_bytes: int
    metrics: Mapping[str, int | float]
    stop: bool
    checkpoint_path: str | None


@dataclass(frozen=True)
class _FinalResult:
    execution: AlgorithmExecutionResult
    state_digest: str
    rounds_completed: int
    metrics: Mapping[str, int | float]


class _TrackedBatches:
    def __init__(self, batches: Iterable[Mapping[str, object]]) -> None:
        self._batches = batches
        self.rows = 0
        self.batch_count = 0
        self.exhausted = False
        self._iterated = False

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        if self._iterated:
            raise AlgorithmInputError(
                "iterative shard batches may be consumed only once per round"
            )
        self._iterated = True
        for batch in self._batches:
            if not isinstance(batch, Mapping):
                raise AlgorithmInputError(
                    "iterative input yielded a non-columnar batch"
                )
            lengths = {_column_length(value) for value in batch.values()}
            if len(lengths) > 1:
                raise AlgorithmInputError(
                    "iterative input batch columns have inconsistent row counts"
                )
            self.rows += next(iter(lengths), 0)
            self.batch_count += 1
            yield batch
        self.exhausted = True


def _column_length(value: object) -> int:
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return shape[0]
    length = getattr(value, "__len__", None)
    if not callable(length):
        raise AlgorithmInputError("iterative input columns must be bounded")
    result = length()
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise AlgorithmInputError("iterative input column length is invalid")
    return result


def _policy(plan: ResolvedAlgorithmPlan) -> IterativeOptimizationPolicy:
    plan.validate_integrity()
    spec = plan.distribution_spec
    if (
        spec is None
        or spec.strategy is not DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION
        or not isinstance(spec.policy, IterativeOptimizationPolicy)
    ):
        raise AlgorithmConfigurationError(
            "RayIterativeOptimizationRuntime requires an iterative policy"
        )
    return spec.policy


def _primary_batches(
    inputs: Mapping[str, object],
    primary_role: str,
) -> Iterable[Mapping[str, object]]:
    try:
        view = inputs[primary_role]
    except KeyError as exc:
        raise AlgorithmInputError(
            "iterative input is missing its primary role"
        ) from exc
    if not isinstance(view, TabularBatchInputView):
        if isinstance(view, MaterializedTabularInputView):
            return (view.columns(),)
        raise AlgorithmInputError(
            "iterative optimization requires a streaming tabular shard"
        )
    return view.iter_batches()


def _checkpoint_directory(plan: ResolvedAlgorithmPlan) -> str | None:
    runtime_config = plan.algorithm_config.get("runtime")
    if runtime_config is None:
        return None
    if not isinstance(runtime_config, Mapping):
        raise AlgorithmConfigurationError("runtime config must be a mapping")
    value = runtime_config.get("checkpoint_dir")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AlgorithmConfigurationError(
            "runtime.checkpoint_dir must be a non-empty path"
        )
    return value


@ray.remote
def _compute_update(
    plan: ResolvedAlgorithmPlan,
    payload: WorkerInputPayload,
    state: object,
    state_digest: str,
    round_index: int,
    rank: int,
    artifacts: tuple[ArtifactDraft, ...],
) -> _UpdateResult:
    policy = _policy(plan)
    algorithm = load_algorithm(plan, IterativeOptimizationAlgorithm, artifacts)
    if policy.max_retries > 0 and not algorithm.retry_safe:
        raise AlgorithmConfigurationError("iterative retries require retry_safe=True")
    prepared = prepare_input(plan, payload)
    try:
        tracked = _TrackedBatches(
            _primary_batches(prepared.views, plan.input_bindings.primary_role)
        )
        identity = runtime_identity()
        context = AlgorithmExecutionContext(
            inputs=prepared.views,
            artifacts=artifacts,
            worker_metadata={
                **identity,
                "world_rank": rank,
                "world_size": plan.runtime.worker_count,
                "round_index": round_index,
                "state_digest": state_digest,
            },
        )
        update = algorithm.compute_partition_update(
            tracked,
            state,
            round_index,
            context,
        )
        if not tracked.exhausted:
            raise AlgorithmExecutionError(
                "compute_partition_update must consume its shard exactly once"
            )
        payload_bytes, update_digest = serialized(
            update,
            max_bytes=policy.max_update_bytes,
            label="iterative local update",
        )
        if payload.expected_total_rows is None:
            raise AlgorithmInputError(
                "iterative shard is missing expected total row evidence"
            )
        evidence = WorkerExecutionEvidence(
            worker_id=identity["worker_id"],
            node_id=identity["node_id"],
            rank=rank,
            world_size=plan.runtime.worker_count,
            shard_id=hashlib.sha256(
                (
                    f"{plan.primary_input_descriptor.binding_digest}:"
                    f"{rank}/{plan.runtime.worker_count}"
                ).encode("ascii")
            ).hexdigest(),
            resources=assigned_resources(),
            model_state_digest=state_digest,
            rows_processed=tracked.rows,
            input_rows={plan.input_bindings.primary_role: tracked.rows},
            batch_count=tracked.batch_count,
        )
        return _UpdateResult(
            update=update,
            digest=update_digest,
            size_bytes=len(payload_bytes),
            workers=(evidence,),
            actual_versions=actual_versions(plan),
            expected_total_rows=payload.expected_total_rows,
            observed_rows=tracked.rows,
        )
    finally:
        prepared.close()


@ray.remote
def _merge_updates(
    plan: ResolvedAlgorithmPlan,
    left: _UpdateResult,
    right: _UpdateResult,
    artifacts: tuple[ArtifactDraft, ...],
) -> _UpdateResult:
    policy = _policy(plan)
    algorithm = load_algorithm(plan, IterativeOptimizationAlgorithm, artifacts)
    if dict(left.actual_versions) != dict(right.actual_versions):
        raise AlgorithmExecutionError(
            "iterative update tasks loaded inconsistent dependencies"
        )
    if left.expected_total_rows != right.expected_total_rows:
        raise AlgorithmExecutionError(
            "iterative shards disagree on expected total rows"
        )
    update = algorithm.merge_updates(left.update, right.update)
    payload, digest = serialized(
        update,
        max_bytes=policy.max_update_bytes,
        label="merged iterative update",
    )
    return _UpdateResult(
        update=update,
        digest=digest,
        size_bytes=len(payload),
        workers=left.workers + right.workers,
        actual_versions=left.actual_versions,
        expected_total_rows=left.expected_total_rows,
        observed_rows=left.observed_rows + right.observed_rows,
        tree_depth=max(left.tree_depth, right.tree_depth) + 1,
    )


@ray.remote(num_cpus=0, max_restarts=0)
class _StateCoordinator:
    def __init__(
        self,
        plan: ResolvedAlgorithmPlan,
        artifacts: tuple[ArtifactDraft, ...],
    ) -> None:
        self._plan = plan
        self._policy = _policy(plan)
        self._algorithm = load_algorithm(
            plan, IterativeOptimizationAlgorithm, artifacts
        )
        if not isinstance(self._algorithm.state_schema(), Mapping):
            raise AlgorithmConfigurationError(
                "iterative state_schema must return a mapping"
            )
        if not isinstance(self._algorithm.update_schema(), Mapping):
            raise AlgorithmConfigurationError(
                "iterative update_schema must return a mapping"
            )
        self._state: object | None = None
        self._state_digest = ""
        self._state_size = 0
        self._rounds_completed = 0
        self._metrics: Mapping[str, int | float] = {}

    def initialize(self) -> tuple[str, int, bool, int]:
        resumed = self._plan.runtime.resume_from is not None
        if resumed:
            self._state = self._load_checkpoint(self._plan.runtime.resume_from or "")
        else:
            config = dict(self._plan.algorithm_config)
            config["_tributo_feature_names"] = tuple(
                self._plan.primary_input_binding.feature_names
            )
            self._state = self._algorithm.initialize_state(
                config,
                self._plan.primary_input_descriptor,
            )
        payload, digest = serialized(
            self._state,
            max_bytes=self._policy.max_state_bytes,
            label="iterative global state",
        )
        self._state_digest = digest
        self._state_size = len(payload)
        return digest, len(payload), resumed, self._rounds_completed

    def state(self) -> object:
        if self._state is None:
            raise AlgorithmExecutionError("iterative state is not initialized")
        return self._state

    def retry_safe(self) -> bool:
        value = self._algorithm.retry_safe
        if not isinstance(value, bool):
            raise AlgorithmConfigurationError("iterative retry_safe must be a boolean")
        return value

    def apply(self, update: _UpdateResult, round_index: int) -> _RoundSummary:
        if self._state is None:
            raise AlgorithmExecutionError("iterative state is not initialized")
        if update.observed_rows != update.expected_total_rows:
            raise AlgorithmExecutionError(
                "iterative round input coverage is incomplete: "
                f"expected={update.expected_total_rows}, "
                f"observed={update.observed_rows}"
            )
        self._state = self._algorithm.apply_update(
            self._state,
            update.update,
            round_index,
        )
        payload, digest = serialized(
            self._state,
            max_bytes=self._policy.max_state_bytes,
            label="iterative global state",
        )
        metrics = self._algorithm.evaluate_round(
            self._state,
            update.update,
            round_index,
        )
        if not isinstance(metrics, Mapping) or any(
            not isinstance(name, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            for name, value in metrics.items()
        ):
            raise AlgorithmExecutionError(
                "evaluate_round must return bounded numeric metrics"
            )
        stop = self._algorithm.should_stop(
            self._state,
            metrics,
            round_index,
        )
        if not isinstance(stop, bool):
            raise AlgorithmExecutionError("should_stop must return bool")
        self._state_digest = digest
        self._state_size = len(payload)
        self._rounds_completed = round_index + 1
        self._metrics = dict(metrics)
        checkpoint_path = None
        if self._rounds_completed % self._policy.checkpoint_interval == 0:
            checkpoint_path = self._write_checkpoint(round_index)
        return _RoundSummary(
            round_index=round_index,
            state_digest=digest,
            state_size_bytes=len(payload),
            metrics=dict(metrics),
            stop=stop,
            checkpoint_path=checkpoint_path,
        )

    def finalize(self, run_id: str) -> _FinalResult:
        if self._state is None:
            raise AlgorithmExecutionError("iterative state is not initialized")
        model = self._algorithm.finalize_model(self._state)
        return _FinalResult(
            execution=execution_result(
                model=model,
                plan=self._plan,
                run_id=run_id,
                metrics=self._metrics,
            ),
            state_digest=self._state_digest,
            rounds_completed=self._rounds_completed,
            metrics=self._metrics,
        )

    def _write_checkpoint(self, round_index: int) -> str | None:
        target = _checkpoint_directory(self._plan)
        if target is None or self._state is None:
            return None
        with tempfile.TemporaryDirectory(prefix="tributo-iterative-checkpoint-") as raw:
            directory = Path(raw)
            codec = self._algorithm.checkpoint_codec()
            payload, digest = codec_payload(
                codec,
                self._state,
                max_bytes=self._policy.max_state_bytes,
            )
            (directory / "state.bin").write_bytes(payload)
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "api_version": 1,
                        "algorithm": self._plan.resolution.algorithm,
                        "implementation_id": self._plan.resolution.implementation_id,
                        "round_index": round_index,
                        "state_sha256": digest,
                        "payload": "state.bin",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return publish_checkpoint_directory(directory, target)

    def _load_checkpoint(self, directory: str | Path) -> object:
        with materialize_checkpoint_directory(directory) as local_directory:
            try:
                manifest = json.loads((local_directory / "manifest.json").read_text())
                payload = (local_directory / "state.bin").read_bytes()
            except (OSError, json.JSONDecodeError) as exc:
                raise AlgorithmExecutionError(
                    "iterative checkpoint could not be read"
                ) from exc
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("api_version") != 1
                or manifest.get("algorithm") != self._plan.resolution.algorithm
                or manifest.get("implementation_id")
                != self._plan.resolution.implementation_id
                or manifest.get("state_sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise AlgorithmExecutionError(
                    "iterative checkpoint is incompatible or corrupted"
                )
            codec = self._algorithm.checkpoint_codec()
            loads = getattr(codec, "loads", None)
            if not callable(loads):
                raise AlgorithmConfigurationError(
                    "iterative checkpoint codec must expose loads(payload)"
                )
            try:
                state = loads(payload)
            except Exception as exc:
                raise AlgorithmExecutionError(
                    "iterative checkpoint codec failed to decode state"
                ) from exc
            round_index = manifest.get("round_index")
            if not isinstance(round_index, int) or isinstance(round_index, bool):
                raise AlgorithmExecutionError(
                    "iterative checkpoint round index is invalid"
                )
            self._rounds_completed = round_index + 1
            return state


def _tree_reduce(
    plan: ResolvedAlgorithmPlan,
    references: list[ray.ObjectRef],
    policy: IterativeOptimizationPolicy,
    artifacts: tuple[ArtifactDraft, ...],
) -> ray.ObjectRef:
    current = list(references)
    while len(current) > 1:
        next_level: list[ray.ObjectRef] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
            else:
                next_level.append(
                    _merge_updates.options(
                        num_cpus=plan.runtime.num_cpus,
                        num_gpus=plan.runtime.num_gpus,
                        memory=plan.runtime.memory_bytes,
                        resources=dict(plan.runtime.custom_resources),
                        max_retries=policy.max_retries,
                        retry_exceptions=True,
                    ).remote(plan, current[index], current[index + 1], artifacts)
                )
        current = next_level
    return current[0]


@DeveloperAPI
class RayIterativeOptimizationRuntime:
    """Own state broadcast, shard updates, barriers, and round checkpoints."""

    @property
    def runtime_id(self) -> str:
        """Return the formal runtime identity."""
        return RAY_ITERATIVE_OPTIMIZATION_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Execute synchronous rounds without loading algorithm code on Driver."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before iterative optimization"
            )
        if envelope.cancelled:
            raise AlgorithmExecutionError("iterative optimization was cancelled")
        payloads: list[WorkerInputPayload] = []
        for payload in envelope.input_payloads:
            if not isinstance(payload, WorkerInputPayload):
                raise AlgorithmInputError(
                    "iterative optimization currently accepts one input role"
                )
            payloads.append(payload)
        policy = _policy(envelope.plan)
        coordinator = cast(Any, _StateCoordinator).remote(
            envelope.plan,
            envelope.artifacts,
        )
        try:
            state_digest, _, resumed, start_round = ray.get(
                coordinator.initialize.remote()
            )
            if policy.max_retries > 0 and not ray.get(coordinator.retry_safe.remote()):
                raise AlgorithmConfigurationError(
                    "iterative policy requests retries for an unsafe algorithm"
                )
            last_update: _UpdateResult | None = None
            last_summary: _RoundSummary | None = None
            for round_index in range(start_round, policy.max_rounds):
                state_ref = coordinator.state.remote()
                references = [
                    _compute_update.options(
                        num_cpus=envelope.plan.runtime.num_cpus,
                        num_gpus=envelope.plan.runtime.num_gpus,
                        memory=envelope.plan.runtime.memory_bytes,
                        resources=dict(envelope.plan.runtime.custom_resources),
                        scheduling_strategy="SPREAD",
                        max_retries=policy.max_retries,
                        retry_exceptions=True,
                    ).remote(
                        envelope.plan,
                        payload,
                        state_ref,
                        state_digest,
                        round_index,
                        rank,
                        envelope.artifacts,
                    )
                    for rank, payload in enumerate(payloads)
                ]
                merged_ref = _tree_reduce(
                    envelope.plan,
                    references,
                    policy,
                    envelope.artifacts,
                )
                last_update = ray.get(merged_ref)
                last_summary = ray.get(
                    coordinator.apply.remote(merged_ref, round_index)
                )
                state_digest = last_summary.state_digest
                if last_summary.stop:
                    break
            if last_update is None or last_summary is None:
                raise AlgorithmExecutionError(
                    "iterative optimization completed no rounds"
                )
            final = ray.get(coordinator.finalize.remote(envelope.run_id))
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"Ray iterative optimization failed: {type(exc).__name__}"
            ) from exc
        workers = tuple(
            replace(worker, model_state_digest=final.state_digest)
            for worker in last_update.workers
        )
        return WorkerExecutionResult(
            execution=final.execution,
            actual_versions=last_update.actual_versions,
            worker_metadata={
                "topology": "ray_iterative_optimization",
                "workers": [worker.to_dict() for worker in workers],
                "state": {
                    "coordination": "iterative_global",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": final.state_digest,
                    "details": {
                        "execution_capability": "single_model_distributed",
                        "rounds_completed": final.rounds_completed,
                        "tree_depth": last_update.tree_depth,
                        "expected_input_rows": last_update.expected_total_rows,
                        "observed_input_rows": last_update.observed_rows,
                        "checkpoint_path": last_summary.checkpoint_path,
                        "resumed": resumed,
                        "exactness": policy.exactness.value,
                    },
                },
                "input_complete": (
                    last_update.observed_rows == last_update.expected_total_rows
                ),
                "driver_materialized_training_rows": 0,
            },
        )


__all__ = ["RayIterativeOptimizationRuntime"]
