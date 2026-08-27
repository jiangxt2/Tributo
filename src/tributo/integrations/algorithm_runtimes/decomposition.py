"""Shared fail-closed helpers for Core-owned decomposition runtimes."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from typing import Any, cast

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    WorkerResources,
)
from tributo.algorithms.core.worker import (
    _actual_environment_versions,
    _load_reference,
    _validate_module_digest,
)
from tributo.algorithms.spi import PreparedInput


def load_algorithm(
    plan: ResolvedAlgorithmPlan,
    expected_type: type[Any],
    artifacts: tuple[ArtifactDraft, ...] = (),
) -> Any:
    """Load one algorithm through its declared factory on a Ray process."""
    _validate_module_digest(
        plan.implementation.implementation_ref,
        plan.implementation.code_digest,
    )
    implementation = _load_reference(plan.implementation.implementation_ref)
    factory = _load_reference(plan.implementation.executable_factory_ref)
    if not callable(factory):
        raise AlgorithmConfigurationError(
            "decomposition executable factory reference is not callable"
        )
    algorithm = factory(plan=plan, implementation=implementation, artifacts=artifacts)
    if not isinstance(algorithm, expected_type):
        raise AlgorithmConfigurationError(
            f"decomposition factory must return {expected_type.__name__}"
        )
    return algorithm


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[ArtifactDraft, ...],
) -> object:
    """Instantiate one no-argument algorithm Hook object on a Ray process."""
    del plan, artifacts
    if isinstance(implementation, type):
        try:
            return implementation()
        except TypeError as exc:
            raise AlgorithmConfigurationError(
                "decomposition algorithm classes require a no-argument constructor"
            ) from exc
    if callable(implementation):
        try:
            return implementation()
        except TypeError as exc:
            raise AlgorithmConfigurationError(
                "decomposition algorithm factories require no arguments"
            ) from exc
    return implementation


def prepare_input(plan: ResolvedAlgorithmPlan, payload: object) -> PreparedInput:
    """Invoke the selected Worker adapter and validate its bounded result."""
    input_factory = _load_reference(plan.runtime.worker_input_adapter_ref)
    if not callable(input_factory):
        raise AlgorithmConfigurationError(
            "decomposition Worker input adapter reference is not callable"
        )
    prepared = input_factory(payload)
    if not isinstance(prepared, PreparedInput):
        raise AlgorithmConfigurationError(
            "decomposition Worker input adapter must return PreparedInput"
        )
    return prepared


def actual_versions(plan: ResolvedAlgorithmPlan) -> Mapping[str, str]:
    """Validate and record the execution environment on one Ray process."""
    return _actual_environment_versions(
        plan.environment.python,
        plan.environment.dependencies,
    )


def runtime_identity() -> dict[str, str]:
    """Return stable Ray job, node, task, and Worker identities."""
    context = ray.get_runtime_context()
    return {
        "job_id": str(context.get_job_id()),
        "node_id": str(context.get_node_id()),
        "task_id": str(context.get_task_id()),
        "worker_id": str(context.get_worker_id()),
    }


def assigned_resources() -> WorkerResources:
    """Convert actual Ray-assigned resources into receipt evidence."""
    assigned = ray.get_runtime_context().get_assigned_resources()
    return WorkerResources(
        num_cpus=float(assigned.get("CPU", 0.0)),
        num_gpus=float(assigned.get("GPU", 0.0)),
        custom={
            str(name): float(value)
            for name, value in assigned.items()
            if name not in {"CPU", "GPU", "memory", "object_store_memory"}
        },
    )


def serialized(value: object, *, max_bytes: int, label: str) -> tuple[bytes, str]:
    """Serialize bounded state and return its content digest."""
    try:
        payload = pickle.dumps(value, protocol=5)
    except Exception as exc:
        raise AlgorithmExecutionError(f"{label} is not serializable") from exc
    if len(payload) > max_bytes:
        raise AlgorithmExecutionError(
            f"{label} exceeds the declared byte limit: "
            f"actual={len(payload)}, limit={max_bytes}"
        )
    return payload, hashlib.sha256(payload).hexdigest()


def codec_payload(codec: object, value: object, *, max_bytes: int) -> tuple[bytes, str]:
    """Encode model/checkpoint state through an algorithm-owned bounded codec."""
    dumps = getattr(codec, "dumps", None)
    if not callable(dumps):
        raise AlgorithmConfigurationError("algorithm codec must expose dumps(value)")
    try:
        payload = dumps(value)
    except Exception as exc:
        raise AlgorithmExecutionError("algorithm codec failed to encode state") from exc
    if not isinstance(payload, bytes):
        raise AlgorithmExecutionError("algorithm codec dumps() must return bytes")
    if len(payload) > max_bytes:
        raise AlgorithmExecutionError(
            "algorithm codec payload exceeds the declared byte limit: "
            f"actual={len(payload)}, limit={max_bytes}"
        )
    return payload, hashlib.sha256(payload).hexdigest()


def execution_result(
    *,
    model: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
    metrics: Mapping[str, int | float] | None = None,
) -> AlgorithmExecutionResult:
    """Apply FIT_ONLY or the declared Bundle exporter to one final model."""
    distribution = plan.distribution_spec
    if distribution is None:
        raise AlgorithmConfigurationError(
            "decomposition runtime requires a DistributionSpec"
        )
    if distribution.result_policy is ResultPolicy.FIT_ONLY:
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=dict(metrics or {}),
        )
    exporter_ref = plan.implementation.exporter_ref
    if exporter_ref is None:
        raise AlgorithmConfigurationError(
            "bundle_required decomposition fit requires an exporter"
        )
    exporter = _load_reference(exporter_ref)
    if not callable(exporter):
        raise AlgorithmConfigurationError(
            "decomposition exporter reference is not callable"
        )
    result = exporter(model=model, plan=plan, run_id=run_id)
    if not isinstance(result, AlgorithmExecutionResult):
        raise AlgorithmExecutionError(
            "decomposition exporter must return AlgorithmExecutionResult"
        )
    if not result.outputs.get("bundle_uri"):
        raise AlgorithmExecutionError(
            "decomposition fit completed without required Bundle publication"
        )
    if metrics:
        merged_metrics = dict(result.metrics)
        merged_metrics.update(metrics)
        result = AlgorithmExecutionResult(
            status=result.status,
            metrics=merged_metrics,
            outputs=result.outputs,
            artifacts=cast(tuple[ArtifactDraft, ...], result.artifacts),
            failure_category=result.failure_category,
            error_type=result.error_type,
            error_message=result.error_message,
        )
    return result


def strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    """Validate an uncoerced integer used by a mathematical runtime."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AlgorithmConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def configured_seed(config: Mapping[str, Any]) -> int:
    """Resolve the common deterministic seed without algorithm-name branching."""
    direct = config.get("seed")
    training = config.get("training")
    nested = training.get("seed") if isinstance(training, Mapping) else None
    value = direct if direct is not None else (nested if nested is not None else 0)
    return strict_int(value, name="seed", minimum=0)


__all__: list[str] = []
