"""Framework-neutral entry points for standalone and inline inference."""

from __future__ import annotations

from tributo.inference.contracts import (
    InferenceExecutor,
    InferenceRequest,
    InferenceResult,
    LanceResultSinkRequest,
    ResolvedInference,
    ResultSink,
)
from tributo.inference.executor import RayMapBatchesExecutor
from tributo.inference.resolver import InferenceResolver
from tributo.integrations.sinks import LanceResultSink, ParquetResultSink
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
def resolve_inference(
    request: InferenceRequest,
    *,
    resolver: InferenceResolver | None = None,
) -> ResolvedInference:
    """Resolve one strict request into an immutable executable plan."""
    validated = InferenceRequest.model_validate(request.model_dump(mode="python"))
    return (resolver or InferenceResolver()).resolve(validated)


@PublicAPI(stability="alpha")
def run_resolved_inference(
    plan: ResolvedInference,
    *,
    executor: InferenceExecutor | None = None,
    sink: ResultSink | None = None,
) -> InferenceResult:
    """Execute one already-pinned plan without reinterpreting user intent."""
    plan = ResolvedInference.model_validate(plan.model_dump(mode="python"))
    if sink is None:
        sink = (
            LanceResultSink()
            if isinstance(plan.result_sink, LanceResultSinkRequest)
            else ParquetResultSink()
        )
    result = (executor or RayMapBatchesExecutor()).execute(plan, sink)
    # Extension implementations cross back into the public domain through a
    # fresh contract validation, including receipt/metrics credential gates.
    return InferenceResult.model_validate(result.model_dump(mode="python"))


@PublicAPI(stability="alpha")
def run_inference(
    request: InferenceRequest,
    *,
    resolver: InferenceResolver | None = None,
    executor: InferenceExecutor | None = None,
    sink: ResultSink | None = None,
) -> InferenceResult:
    """Resolve and execute one standalone or post-training request."""
    plan = resolve_inference(request, resolver=resolver)
    return run_resolved_inference(plan, executor=executor, sink=sink)


__all__ = ["resolve_inference", "run_inference", "run_resolved_inference"]
