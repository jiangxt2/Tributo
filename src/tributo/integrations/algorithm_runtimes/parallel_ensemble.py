"""Ray Runtime for deterministic independent ensemble units."""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    DistributionStrategy,
    ParallelEnsemblePolicy,
    ResolvedAlgorithmPlan,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    EnsembleUnitSpec,
    MaterializedTabularInputView,
    ParallelEnsembleAlgorithm,
    RuntimeExecutionEnvelope,
)
from tributo.integrations.algorithm_runtimes.decomposition import (
    actual_versions,
    assigned_resources,
    configured_seed,
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

RAY_PARALLEL_ENSEMBLE_RUNTIME_ID = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_PARALLEL_ENSEMBLE
].runtime_id


@dataclass(frozen=True)
class _UnitPayload:
    unit_id: str
    payload: bytes
    digest: str


@dataclass(frozen=True)
class _EnsembleWorkerResult:
    units: tuple[_UnitPayload, ...]
    evidence: WorkerExecutionEvidence
    actual_versions: Mapping[str, str]


@dataclass(frozen=True)
class _EnsembleFinalResult:
    execution: AlgorithmExecutionResult
    model_digest: str


def _policy(plan: ResolvedAlgorithmPlan) -> ParallelEnsemblePolicy:
    plan.validate_integrity()
    spec = plan.distribution_spec
    if (
        spec is None
        or spec.strategy is not DistributionStrategy.RAY_PARALLEL_ENSEMBLE
        or not isinstance(spec.policy, ParallelEnsemblePolicy)
    ):
        raise AlgorithmConfigurationError(
            "RayParallelUnitRuntime requires a ray_parallel_ensemble policy"
        )
    return spec.policy


@ray.remote(num_cpus=0, max_retries=0)
def _plan_units(
    plan: ResolvedAlgorithmPlan,
    artifacts: tuple[ArtifactDraft, ...],
) -> tuple[tuple[EnsembleUnitSpec, ...], bool]:
    policy = _policy(plan)
    algorithm = load_algorithm(plan, ParallelEnsembleAlgorithm, artifacts)
    units = algorithm.plan_units(
        plan.algorithm_config,
        plan.primary_input_descriptor,
        configured_seed(plan.algorithm_config),
    )
    if not isinstance(units, tuple) or any(
        not isinstance(unit, EnsembleUnitSpec) for unit in units
    ):
        raise AlgorithmConfigurationError(
            "Parallel Ensemble plan_units must return EnsembleUnitSpec tuple"
        )
    if not units:
        raise AlgorithmConfigurationError(
            "Parallel Ensemble requires at least one unit"
        )
    if len(units) > policy.max_units:
        raise AlgorithmConfigurationError(
            "Parallel Ensemble unit count exceeds max_units: "
            f"actual={len(units)}, limit={policy.max_units}"
        )
    identities = tuple(unit.unit_id for unit in units)
    if len(set(identities)) != len(identities):
        raise AlgorithmConfigurationError(
            "Parallel Ensemble unit identities must be unique"
        )
    schema = algorithm.unit_schema()
    if not isinstance(schema, Mapping) or not schema:
        raise AlgorithmConfigurationError(
            "Parallel Ensemble unit_schema must be a non-empty mapping"
        )
    return units, algorithm.retry_safe


def _primary_rows(inputs: Mapping[str, object], role: str) -> int:
    try:
        primary = inputs[role]
    except KeyError as exc:
        raise AlgorithmInputError(
            "Parallel Ensemble input is missing the primary role"
        ) from exc
    if not isinstance(primary, MaterializedTabularInputView):
        raise AlgorithmInputError(
            "Parallel Ensemble requires materialized tabular input"
        )
    if primary.row_count < 1:
        raise AlgorithmInputError("Parallel Ensemble input must not be empty")
    return primary.row_count


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


def _write_checkpoint(
    plan: ResolvedAlgorithmPlan,
    expected_ids: tuple[str, ...],
    results: Mapping[int, _EnsembleWorkerResult],
) -> str | None:
    target = _checkpoint_directory(plan)
    if target is None:
        return None
    with tempfile.TemporaryDirectory(prefix="tributo-ensemble-checkpoint-") as raw:
        directory = Path(raw)
        workers = directory / "workers"
        workers.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        for rank, result in sorted(results.items()):
            payload = pickle.dumps(result, protocol=5)
            digest = hashlib.sha256(payload).hexdigest()
            relative_path = f"workers/rank-{rank}.bin"
            (directory / relative_path).write_bytes(payload)
            entries.append(
                {
                    "rank": rank,
                    "path": relative_path,
                    "sha256": digest,
                    "unit_ids": [unit.unit_id for unit in result.units],
                }
            )
        manifest = {
            "api_version": 1,
            "algorithm": plan.resolution.algorithm,
            "implementation_id": plan.resolution.implementation_id,
            "worker_count": plan.runtime.worker_count,
            "expected_unit_ids": list(expected_ids),
            "workers": entries,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return publish_checkpoint_directory(directory, target)


def _load_checkpoint(
    plan: ResolvedAlgorithmPlan,
    expected_ids: tuple[str, ...],
    assignments: tuple[tuple[EnsembleUnitSpec, ...], ...],
) -> dict[int, _EnsembleWorkerResult]:
    if plan.runtime.resume_from is None:
        return {}
    target = plan.runtime.resume_from
    with materialize_checkpoint_directory(target) as directory:
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AlgorithmExecutionError(
                "ensemble checkpoint could not be read"
            ) from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("api_version") != 1
            or manifest.get("algorithm") != plan.resolution.algorithm
            or manifest.get("implementation_id") != plan.resolution.implementation_id
            or manifest.get("worker_count") != plan.runtime.worker_count
            or manifest.get("expected_unit_ids") != list(expected_ids)
            or not isinstance(manifest.get("workers"), list)
        ):
            raise AlgorithmExecutionError(
                "ensemble checkpoint is incompatible or corrupted"
            )
        restored: dict[int, _EnsembleWorkerResult] = {}
        for entry in manifest["workers"]:
            if not isinstance(entry, Mapping):
                raise AlgorithmExecutionError("ensemble checkpoint entry is invalid")
            rank = entry.get("rank")
            path = entry.get("path")
            digest = entry.get("sha256")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 0
                or rank >= len(assignments)
                or rank in restored
                or not isinstance(path, str)
                or not path.startswith("workers/rank-")
                or not isinstance(digest, str)
            ):
                raise AlgorithmExecutionError("ensemble checkpoint entry is invalid")
            try:
                payload = (directory / path).read_bytes()
            except OSError as exc:
                raise AlgorithmExecutionError(
                    "ensemble checkpoint payload could not be read"
                ) from exc
            if hashlib.sha256(payload).hexdigest() != digest:
                raise AlgorithmExecutionError(
                    "ensemble checkpoint payload digest mismatch"
                )
            try:
                result = pickle.loads(payload)
            except Exception as exc:
                raise AlgorithmExecutionError(
                    "ensemble checkpoint payload could not be decoded"
                ) from exc
            expected_assignment = tuple(unit.unit_id for unit in assignments[rank])
            if (
                not isinstance(result, _EnsembleWorkerResult)
                or result.evidence.rank != rank
                or result.evidence.world_size != plan.runtime.worker_count
                or tuple(unit.unit_id for unit in result.units) != expected_assignment
                or entry.get("unit_ids") != list(expected_assignment)
            ):
                raise AlgorithmExecutionError(
                    "ensemble checkpoint worker state is incompatible"
                )
            restored[rank] = result
        return restored


@ray.remote
def _fit_units(
    plan: ResolvedAlgorithmPlan,
    payload: object,
    units: tuple[EnsembleUnitSpec, ...],
    rank: int,
    artifacts: tuple[ArtifactDraft, ...],
) -> _EnsembleWorkerResult:
    policy = _policy(plan)
    algorithm = load_algorithm(plan, ParallelEnsembleAlgorithm, artifacts)
    if policy.max_retries > 0 and not algorithm.retry_safe:
        raise AlgorithmConfigurationError(
            "Parallel Ensemble retries require retry_safe=True"
        )
    prepared = prepare_input(plan, payload)
    try:
        rows = _primary_rows(prepared.views, plan.input_bindings.primary_role)
        identity = runtime_identity()
        context = AlgorithmExecutionContext(
            inputs=prepared.views,
            artifacts=artifacts,
            worker_metadata={
                **identity,
                "world_rank": rank,
                "world_size": plan.runtime.worker_count,
            },
        )
        fitted: list[_UnitPayload] = []
        digest = hashlib.sha256()
        for unit in units:
            model = algorithm.fit_unit(unit, prepared.views, context)
            payload_bytes, unit_digest = serialized(
                model,
                max_bytes=policy.max_unit_model_bytes,
                label=f"ensemble unit {unit.unit_id!r}",
            )
            fitted.append(
                _UnitPayload(
                    unit_id=unit.unit_id,
                    payload=payload_bytes,
                    digest=unit_digest,
                )
            )
            digest.update(unit.unit_id.encode("utf-8"))
            digest.update(unit_digest.encode("ascii"))
        evidence = WorkerExecutionEvidence(
            worker_id=identity["worker_id"],
            node_id=identity["node_id"],
            rank=rank,
            world_size=plan.runtime.worker_count,
            shard_id=hashlib.sha256(
                ",".join(unit.unit_id for unit in units).encode("utf-8")
            ).hexdigest(),
            resources=assigned_resources(),
            model_state_digest=digest.hexdigest(),
            rows_processed=rows,
            input_rows={plan.input_bindings.primary_role: rows},
        )
        return _EnsembleWorkerResult(
            units=tuple(fitted),
            evidence=evidence,
            actual_versions=actual_versions(plan),
        )
    finally:
        prepared.close()


@ray.remote(num_cpus=0, max_retries=0)
def _finalize_ensemble(
    plan: ResolvedAlgorithmPlan,
    ordered_units: tuple[_UnitPayload, ...],
    run_id: str,
    artifacts: tuple[ArtifactDraft, ...],
) -> _EnsembleFinalResult:
    policy = _policy(plan)
    algorithm = load_algorithm(plan, ParallelEnsembleAlgorithm, artifacts)
    models: list[object] = []
    for unit in ordered_units:
        if hashlib.sha256(unit.payload).hexdigest() != unit.digest:
            raise AlgorithmExecutionError(
                f"ensemble unit {unit.unit_id!r} failed digest validation"
            )
        try:
            models.append(pickle.loads(unit.payload))
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"ensemble unit {unit.unit_id!r} could not be decoded"
            ) from exc
    merged = algorithm.merge_units(tuple(models))
    model = algorithm.finalize_ensemble(merged)
    max_model_bytes = policy.max_unit_model_bytes * max(1, len(models))
    _, model_digest = serialized(
        model,
        max_bytes=max_model_bytes,
        label="final ensemble model",
    )
    return _EnsembleFinalResult(
        execution=execution_result(model=model, plan=plan, run_id=run_id),
        model_digest=model_digest,
    )


def _assign_units(
    units: tuple[EnsembleUnitSpec, ...],
    worker_count: int,
) -> tuple[tuple[EnsembleUnitSpec, ...], ...]:
    if len(units) < worker_count:
        raise AlgorithmConfigurationError(
            "Parallel Ensemble requires at least one unit per requested Worker"
        )
    assignments: list[list[EnsembleUnitSpec]] = [[] for _ in range(worker_count)]
    for index, unit in enumerate(units):
        assignments[index % worker_count].append(unit)
    return tuple(tuple(group) for group in assignments)


@DeveloperAPI
class RayParallelUnitRuntime:
    """Execute deterministic ensemble units on a SPREAD Ray Worker group."""

    @property
    def runtime_id(self) -> str:
        """Return the formal runtime identity."""
        return RAY_PARALLEL_ENSEMBLE_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Plan units, execute them in parallel, then finalize outside Driver."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before Parallel Ensemble execution"
            )
        if envelope.cancelled:
            raise AlgorithmExecutionError("Parallel Ensemble execution was cancelled")
        if len(envelope.input_payloads) != 1:
            raise AlgorithmInputError(
                "Parallel Ensemble requires one unsplit full-dataset payload"
            )
        policy = _policy(envelope.plan)
        try:
            units, retry_safe = ray.get(
                _plan_units.remote(envelope.plan, envelope.artifacts)
            )
            if policy.max_retries > 0 and not retry_safe:
                raise AlgorithmConfigurationError(
                    "Parallel Ensemble policy requests retries for an unsafe algorithm"
                )
            assignments = _assign_units(units, envelope.plan.runtime.worker_count)
            expected_ids = tuple(unit.unit_id for unit in units)
            by_rank = _load_checkpoint(envelope.plan, expected_ids, assignments)
            restored_unit_count = sum(len(result.units) for result in by_rank.values())
            pending = {
                _fit_units.options(
                    num_cpus=envelope.plan.runtime.num_cpus,
                    num_gpus=envelope.plan.runtime.num_gpus,
                    resources=dict(envelope.plan.runtime.custom_resources),
                    scheduling_strategy="SPREAD",
                    max_retries=policy.max_retries,
                    retry_exceptions=True,
                ).remote(
                    envelope.plan,
                    envelope.input_payloads[0],
                    assignment,
                    rank,
                    envelope.artifacts,
                ): rank
                for rank, assignment in enumerate(assignments)
                if rank not in by_rank
            }
            checkpoint_path = _checkpoint_directory(envelope.plan)
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                reference = ready[0]
                rank = pending.pop(reference)
                by_rank[rank] = ray.get(reference)
                completed_units = sum(len(result.units) for result in by_rank.values())
                if completed_units % policy.checkpoint_interval == 0 or not pending:
                    checkpoint_path = _write_checkpoint(
                        envelope.plan,
                        expected_ids,
                        by_rank,
                    )
            worker_results = tuple(by_rank[rank] for rank in range(len(assignments)))
            versions = dict(worker_results[0].actual_versions)
            if any(
                dict(result.actual_versions) != versions for result in worker_results
            ):
                raise AlgorithmExecutionError(
                    "Parallel Ensemble Workers loaded inconsistent dependencies"
                )
            by_id = {
                unit.unit_id: unit for result in worker_results for unit in result.units
            }
            if len(by_id) != len(units) or set(by_id) != set(expected_ids):
                raise AlgorithmExecutionError(
                    "Parallel Ensemble unit coverage is incomplete or duplicated"
                )
            ordered = tuple(by_id[unit_id] for unit_id in expected_ids)
            final = ray.get(
                _finalize_ensemble.remote(
                    envelope.plan,
                    ordered,
                    envelope.run_id,
                    envelope.artifacts,
                )
            )
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"Ray Parallel Ensemble execution failed: {type(exc).__name__}"
            ) from exc
        unit_digest = hashlib.sha256(
            "\n".join(expected_ids).encode("utf-8")
        ).hexdigest()
        workers = tuple(result.evidence for result in worker_results)
        return WorkerExecutionResult(
            execution=final.execution,
            actual_versions=versions,
            worker_metadata={
                "topology": "ray_parallel_ensemble",
                "workers": [worker.to_dict() for worker in workers],
                "state": {
                    "coordination": "ordered_ensemble",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": final.model_digest,
                    "details": {
                        "execution_capability": "single_model_distributed",
                        "unit_count": len(units),
                        "unit_ids_digest": unit_digest,
                        "worker_count": len(workers),
                        "exactness": policy.exactness.value,
                        "checkpoint_path": checkpoint_path,
                        "resumed": envelope.plan.runtime.resume_from is not None,
                        "restored_unit_count": restored_unit_count,
                    },
                },
                "input_complete": True,
                "driver_materialized_training_rows": 0,
            },
        )


__all__ = ["RayParallelUnitRuntime"]
