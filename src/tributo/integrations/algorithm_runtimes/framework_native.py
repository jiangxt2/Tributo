"""Runtime Adapter for framework-owned distributed training implementations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    DistributionStrategy,
    FrameworkNativePolicy,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    StateCoordination,
    StateCoordinationEvidence,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.core.worker import (
    _actual_environment_versions,
    _load_reference,
    _validate_module_digest,
)
from tributo.algorithms.spi import (
    FrameworkNativeAlgorithm,
    PreparedInput,
    RuntimeExecutionEnvelope,
)
from tributo.exceptions import BundleExportError
from tributo.integrations.algorithm_runtimes.portable_metrics import (
    portable_fit_only_metrics,
)
from tributo.util.annotations import DeveloperAPI

FRAMEWORK_NATIVE_RUNTIME_ID = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.FRAMEWORK_NATIVE
].runtime_id


def _bundle_export_failure_message(error: BundleExportError) -> str:
    """Keep required Bundle node failures visible across a Ray Job boundary."""
    execution = error.execution_result
    if execution is None:
        return "framework-native Bundle export failed without an execution result"
    failures: list[dict[str, object]] = []
    for node in getattr(execution, "node_results", ()):
        status = getattr(node, "status", None)
        if status == "succeeded":
            continue
        failure = getattr(node, "failure", None)
        failures.append(
            {
                "node_id": getattr(node, "node_id", None),
                "status": status,
                "exporter_id": getattr(node, "exporter_id", None),
                "code": getattr(failure, "code", None),
                "category": getattr(failure, "category", None),
                "message": str(getattr(failure, "message", ""))[:4096],
            }
        )
    return "framework-native Bundle export failed: " + json.dumps(
        failures, sort_keys=True, separators=(",", ":")
    )


def _framework_execution_result(
    *,
    algorithm: FrameworkNativeAlgorithm,
    result: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    """Apply result delivery only after framework evidence has been validated."""
    distribution = plan.distribution_spec
    if distribution is not None and distribution.result_policy is ResultPolicy.FIT_ONLY:
        raw_metrics = getattr(result, "metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise AlgorithmExecutionError(
                "framework-native FIT_ONLY result metrics must be a mapping"
            )
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=portable_fit_only_metrics(raw_metrics),
        )
    checkpoint = algorithm.checkpoint_source(result)
    exporter_ref = plan.implementation.exporter_ref
    if exporter_ref is None:
        raise AlgorithmConfigurationError(
            "framework-native bundle_required fit requires an exporter"
        )
    exporter = _load_reference(exporter_ref)
    if not callable(exporter):
        raise AlgorithmConfigurationError("framework-native exporter is not callable")
    execution = exporter(
        result=result,
        checkpoint=checkpoint,
        plan=plan,
        run_id=run_id,
    )
    if not isinstance(execution, AlgorithmExecutionResult):
        raise AlgorithmExecutionError(
            "framework-native exporter must return AlgorithmExecutionResult"
        )
    if not execution.outputs.get("bundle_uri"):
        raise AlgorithmExecutionError(
            "framework-native fit completed without the required Bundle publication"
        )
    return execution


def _validated_framework_evidence(
    evidence: Mapping[str, Any],
    *,
    worker_count: int,
    resources_per_worker: WorkerResources,
    expected_training_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate observed framework facts before any Bundle is published."""
    raw_workers = evidence.get("workers")
    raw_state = evidence.get("state")
    if not isinstance(raw_workers, (list, tuple)) or not isinstance(raw_state, Mapping):
        raise AlgorithmExecutionError(
            "framework-native evidence is missing workers or state"
        )
    if len(raw_workers) != worker_count:
        raise AlgorithmExecutionError(
            "framework-native evidence did not report every requested worker"
        )
    workers: list[WorkerExecutionEvidence] = []
    try:
        for value in raw_workers:
            if not isinstance(value, Mapping):
                raise TypeError("worker evidence must be a mapping")
            workers.append(WorkerExecutionEvidence.from_dict(value))
        state = StateCoordinationEvidence.from_dict(raw_state)
    except (AlgorithmConfigurationError, KeyError, TypeError, ValueError) as exc:
        raise AlgorithmExecutionError(
            "framework-native execution evidence is malformed"
        ) from exc
    if (
        {worker.rank for worker in workers} != set(range(worker_count))
        or any(worker.world_size != worker_count for worker in workers)
        or len({worker.worker_id for worker in workers}) != worker_count
        or len({worker.shard_id for worker in workers}) != worker_count
    ):
        raise AlgorithmExecutionError(
            "framework-native evidence does not prove unique workers and shards"
        )
    if any(
        worker.resources.num_cpus < resources_per_worker.num_cpus
        or worker.resources.num_gpus < resources_per_worker.num_gpus
        or (
            resources_per_worker.memory_bytes is not None
            and (
                worker.resources.memory_bytes is None
                or worker.resources.memory_bytes < resources_per_worker.memory_bytes
            )
        )
        or any(
            worker.resources.custom.get(name, 0.0) < amount
            for name, amount in resources_per_worker.custom.items()
        )
        or not worker.rows_processed
        for worker in workers
    ):
        raise AlgorithmExecutionError(
            "framework-native evidence does not satisfy requested resources and input"
        )
    if sum(worker.rows_processed or 0 for worker in workers) != expected_training_rows:
        raise AlgorithmExecutionError(
            "framework-native evidence does not prove complete training input coverage"
        )
    if (
        state.coordination is not StateCoordination.FRAMEWORK_NATIVE
        or not state.synchronized
        or not state.bounded
        or state.global_model_digest is None
        or {worker.model_state_digest for worker in workers}
        != {state.global_model_digest}
    ):
        raise AlgorithmExecutionError(
            "framework-native evidence does not prove one synchronized model state"
        )
    if evidence.get("input_complete") is not True:
        raise AlgorithmExecutionError(
            "framework-native evidence does not prove complete input coverage"
        )
    return (
        [worker.to_dict() for worker in sorted(workers, key=lambda item: item.rank)],
        {
            "coordination": state.coordination.value,
            "synchronized": state.synchronized,
            "bounded": state.bounded,
            "global_model_digest": state.global_model_digest,
            "details": dict(state.details),
        },
    )


def _validated_staged_framework_evidence(
    evidence: Mapping[str, Any],
    *,
    component_stages: tuple[str, ...],
    worker_count: int,
    resources_per_worker: WorkerResources,
    expected_training_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a bounded, explicitly declared sequence of framework fits."""
    raw_stages = evidence.get("stages")
    if not isinstance(raw_stages, Mapping):
        raise AlgorithmExecutionError(
            "staged framework evidence requires a stages mapping"
        )
    if set(raw_stages) != set(component_stages):
        raise AlgorithmExecutionError(
            "staged framework evidence does not match declared component stages"
        )
    composition_digest = evidence.get("composition_digest")
    if (
        not isinstance(composition_digest, str)
        or len(composition_digest) != 64
        or any(character not in "0123456789abcdef" for character in composition_digest)
    ):
        raise AlgorithmExecutionError(
            "staged framework evidence requires a canonical composition digest"
        )
    validated: dict[str, tuple[list[dict[str, Any]], dict[str, Any], int]] = {}
    for stage in component_stages:
        payload = raw_stages[stage]
        if not isinstance(payload, Mapping):
            raise AlgorithmExecutionError(
                f"framework stage {stage!r} evidence must be a mapping"
            )
        cross_fit_folds = payload.get("cross_fit_folds")
        if isinstance(cross_fit_folds, list):
            if len(cross_fit_folds) < 2:
                raise AlgorithmExecutionError(
                    f"framework stage {stage!r} requires at least two cross-fit folds"
                )
            fold_records: list[
                tuple[list[dict[str, Any]], dict[str, Any], int, int]
            ] = []
            heldout_total = 0
            for fold in cross_fit_folds:
                if not isinstance(fold, Mapping):
                    raise AlgorithmExecutionError(
                        f"framework stage {stage!r} cross-fit fold is malformed"
                    )
                fold_rows = fold.get("expected_training_rows")
                heldout_rows = fold.get("heldout_rows")
                if (
                    not isinstance(fold_rows, int)
                    or isinstance(fold_rows, bool)
                    or fold_rows < 1
                    or not isinstance(heldout_rows, int)
                    or isinstance(heldout_rows, bool)
                    or heldout_rows < 1
                ):
                    raise AlgorithmExecutionError(
                        f"framework stage {stage!r} cross-fit coverage is malformed"
                    )
                fold_workers, fold_state = _validated_framework_evidence(
                    fold,
                    worker_count=worker_count,
                    resources_per_worker=resources_per_worker,
                    expected_training_rows=fold_rows,
                )
                fold_records.append((fold_workers, fold_state, fold_rows, heldout_rows))
                heldout_total += heldout_rows
            if heldout_total != expected_training_rows:
                raise AlgorithmExecutionError(
                    f"framework stage {stage!r} cross-fit heldout coverage does not "
                    "partition the input"
                )
            rank_rows = [0] * worker_count
            for workers, _, _, _ in fold_records:
                for worker in workers:
                    rank = int(worker["rank"])
                    rank_rows[rank] += int(worker["rows_processed"])
            digest = hashlib.sha256(
                json.dumps(
                    [state["global_model_digest"] for _, state, _, _ in fold_records],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            workers = [dict(worker) for worker in fold_records[0][0]]
            for worker in workers:
                worker["rows_processed"] = rank_rows[int(worker["rank"])]
                worker["model_state_digest"] = digest
            validated[stage] = (
                workers,
                {
                    "coordination": StateCoordination.FRAMEWORK_NATIVE.value,
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": digest,
                    "details": {
                        "framework": "staged_cross_fit",
                        "cross_fit_folds": len(fold_records),
                        "heldout_rows": heldout_total,
                        "training_rows_per_fold": json.dumps(
                            [item[2] for item in fold_records], separators=(",", ":")
                        ),
                    },
                },
                expected_training_rows,
            )
            continue
        stage_rows = payload.get("expected_training_rows")
        if (
            not isinstance(stage_rows, int)
            or isinstance(stage_rows, bool)
            or stage_rows < 1
            or stage_rows > expected_training_rows
        ):
            raise AlgorithmExecutionError(
                f"framework stage {stage!r} has invalid expected training rows"
            )
        workers, state = _validated_framework_evidence(
            payload,
            worker_count=worker_count,
            resources_per_worker=resources_per_worker,
            expected_training_rows=stage_rows,
        )
        validated[stage] = (workers, state, stage_rows)

    full_coverage_stages = [
        stage
        for stage, (_, _, rows) in validated.items()
        if rows == expected_training_rows
    ]
    if not full_coverage_stages:
        raise AlgorithmExecutionError(
            "staged framework evidence has no full-input anchor stage"
        )
    anchor = max(
        full_coverage_stages,
        key=lambda stage: (
            len({item["node_id"] for item in validated[stage][0]}),
            stage,
        ),
    )
    digest_payload = {
        "composition_digest": composition_digest,
        "stages": {
            stage: {
                "digest": validated[stage][1]["global_model_digest"],
                "rows": validated[stage][2],
            }
            for stage in component_stages
        },
    }
    composite_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    details: dict[str, str | int | float | bool | None] = {
        "framework": "staged_composite",
        "component_stage_count": len(component_stages),
        "component_stages": ",".join(component_stages),
        "anchor_stage": anchor,
        "composition_digest": composition_digest,
    }
    for stage in component_stages:
        details[f"stage.{stage}.digest"] = str(
            validated[stage][1]["global_model_digest"]
        )
        details[f"stage.{stage}.rows"] = validated[stage][2]
        details[f"stage.{stage}.workers"] = len(validated[stage][0])
        details[f"stage.{stage}.nodes"] = len(
            {item["node_id"] for item in validated[stage][0]}
        )
    return validated[anchor][0], {
        "coordination": StateCoordination.FRAMEWORK_NATIVE.value,
        "synchronized": True,
        "bounded": True,
        "global_model_digest": composite_digest,
        "details": details,
    }


def _algorithm(envelope: RuntimeExecutionEnvelope) -> FrameworkNativeAlgorithm:
    plan = envelope.plan
    spec = plan.distribution_spec
    if spec is None or spec.strategy is not DistributionStrategy.FRAMEWORK_NATIVE:
        raise AlgorithmConfigurationError(
            "framework runtime requires a framework_native DistributionSpec"
        )
    _validate_module_digest(
        plan.implementation.implementation_ref,
        plan.implementation.code_digest,
    )
    implementation = _load_reference(plan.implementation.implementation_ref)
    factory = _load_reference(plan.implementation.executable_factory_ref)
    if not callable(factory):
        raise AlgorithmConfigurationError(
            "framework-native executable factory is not callable"
        )
    value = factory(
        plan=plan,
        implementation=implementation,
        artifacts=envelope.artifacts,
    )
    if not isinstance(value, FrameworkNativeAlgorithm):
        raise AlgorithmConfigurationError(
            "framework-native factory must return FrameworkNativeAlgorithm"
        )
    if not isinstance(spec.policy, FrameworkNativePolicy):
        raise AlgorithmConfigurationError(
            "framework-native DistributionSpec requires FrameworkNativePolicy"
        )
    declared_collector = _load_reference(
        QualifiedReference.parse(spec.policy.evidence_collector_ref)
    )
    if declared_collector is not getattr(type(value), "collect_evidence", None):
        raise AlgorithmConfigurationError(
            "framework-native evidence_collector_ref does not match the "
            "implementation method"
        )
    return value


@DeveloperAPI
class FrameworkNativeRuntime:
    """Run an installed framework trainer and require framework evidence."""

    @property
    def runtime_id(self) -> str:
        """Return the formal runtime identity."""
        return FRAMEWORK_NATIVE_RUNTIME_ID

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        """Execute one framework-owned trainer without materializing Driver data."""
        if not ray.is_initialized():
            raise AlgorithmExecutionError(
                "Ray must be initialized before framework-native execution"
            )
        if envelope.cancelled:
            raise AlgorithmExecutionError("framework-native execution was cancelled")
        algorithm = _algorithm(envelope)
        algorithm.validate_environment()
        input_adapter = _load_reference(envelope.plan.runtime.worker_input_adapter_ref)
        if not callable(input_adapter):
            raise AlgorithmConfigurationError(
                "framework-native input adapter is not callable"
            )
        prepared = input_adapter(envelope.input_payloads[0])
        if not isinstance(prepared, PreparedInput):
            raise AlgorithmConfigurationError(
                "framework-native input adapter must return PreparedInput"
            )
        try:
            datasets = algorithm.bind_datasets(prepared.views)
            if not isinstance(datasets, Mapping):
                raise AlgorithmConfigurationError(
                    "framework-native bind_datasets must return a mapping"
                )
            training_dataset = datasets.get("train")
            count_training_rows = getattr(training_dataset, "count", None)
            if not callable(count_training_rows):
                raise AlgorithmConfigurationError(
                    "framework-native fit requires a named 'train' Ray Dataset"
                )
            expected_training_rows = int(count_training_rows())
            if expected_training_rows < 1:
                raise AlgorithmConfigurationError(
                    "framework-native training Dataset must be non-empty"
                )
            trainer = algorithm.build_trainer(
                envelope.plan.algorithm_config,
                datasets,
            )
            fit = getattr(trainer, "fit", None)
            if not callable(fit):
                raise AlgorithmConfigurationError(
                    "framework-native build_trainer must return a fit-capable object"
                )
            result = fit()
            evidence = algorithm.collect_evidence(result)
            if not isinstance(evidence, Mapping):
                raise AlgorithmExecutionError(
                    "framework-native collect_evidence must return a mapping"
                )
            plan = envelope.plan
            resources = WorkerResources(
                num_cpus=plan.runtime.num_cpus,
                num_gpus=plan.runtime.num_gpus,
                memory_bytes=plan.runtime.memory_bytes,
                custom=plan.runtime.custom_resources,
            )
            distribution = plan.distribution_spec
            if distribution is None:
                raise AlgorithmConfigurationError(
                    "framework-native execution lost its DistributionSpec"
                )
            policy = distribution.policy
            assert isinstance(policy, FrameworkNativePolicy)
            if policy.component_stages:
                workers, state = _validated_staged_framework_evidence(
                    evidence,
                    component_stages=policy.component_stages,
                    worker_count=plan.runtime.worker_count,
                    resources_per_worker=resources,
                    expected_training_rows=expected_training_rows,
                )
            else:
                workers, state = _validated_framework_evidence(
                    evidence,
                    worker_count=plan.runtime.worker_count,
                    resources_per_worker=resources,
                    expected_training_rows=expected_training_rows,
                )
            try:
                execution = _framework_execution_result(
                    algorithm=algorithm,
                    result=result,
                    plan=envelope.plan,
                    run_id=envelope.run_id,
                )
            except BundleExportError as exc:
                raise AlgorithmExecutionError(
                    _bundle_export_failure_message(exc)
                ) from exc
            versions = _actual_environment_versions(
                envelope.plan.environment.python,
                envelope.plan.environment.dependencies,
            )
            return WorkerExecutionResult(
                execution=execution,
                actual_versions=versions,
                worker_metadata={
                    "topology": "framework_native",
                    "workers": workers,
                    "state": state,
                    "input_complete": True,
                    "driver_materialized_training_rows": 0,
                },
            )
        except ray.exceptions.RayError as exc:
            raise AlgorithmExecutionError(
                f"framework-native execution failed: {type(exc).__name__}"
            ) from exc
        finally:
            prepared.close()


__all__ = ["FrameworkNativeRuntime"]
