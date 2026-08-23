"""Framework-neutral entry points for standalone and inline inference."""

from __future__ import annotations

from tributo.inference.contracts import (
    BoundResultSink,
    InferenceExecutor,
    InferenceRequest,
    InferenceResult,
    PreparedInferencePlan,
    ResolvedInference,
    ResultSink,
    ResultSinkProvider,
)
from tributo.inference.executor import RayMapBatchesExecutor, prepared_inference_plan
from tributo.inference.input_resolver import OpenedInferenceInput
from tributo.inference.kernel import ModelKernelProvider, PredictionKernelFactory
from tributo.inference.resolver import InferenceResolver
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
    bound_sink: BoundResultSink | None = None,
    sink_provider: ResultSinkProvider | None = None,
    kernel_provider: ModelKernelProvider | None = None,
) -> InferenceResult:
    """Execute one already-pinned plan without reinterpreting user intent."""
    plan = ResolvedInference.model_validate(plan.model_dump(mode="python"))
    if sink is not None and bound_sink is not None:
        raise ValueError("sink and bound_sink are mutually exclusive")
    if executor is None:
        kernel_factory = (
            kernel_provider or _default_model_kernel_provider()
        ).prediction_factory(plan.model)
        executor = RayMapBatchesExecutor(kernel_factory=kernel_factory)
        if bound_sink is not None:
            result = executor.execute_bound(plan, bound_sink)
        elif sink is not None:
            result = executor.execute(plan, sink)
        else:
            provider = sink_provider or _default_result_sink_provider()
            result = executor.execute_bound(plan, provider.bind(plan.result_sink))
    else:
        if bound_sink is not None:
            raise ValueError("custom InferenceExecutor requires the legacy sink port")
        if sink is None:
            provider = sink_provider or _default_result_sink_provider()
            sink = provider.create(plan.result_sink)
        result = executor.execute(plan, sink)
    # Extension implementations cross back into the public domain through a
    # fresh contract validation, including receipt/metrics credential gates.
    return InferenceResult.model_validate(result.model_dump(mode="python"))


@PublicAPI(stability="alpha")
def run_prepared_inference(
    plan: PreparedInferencePlan,
    *,
    opened_input: OpenedInferenceInput,
    kernel_factory: PredictionKernelFactory,
    sink: BoundResultSink,
    executor: RayMapBatchesExecutor | None = None,
) -> InferenceResult:
    """Execute already-prepared data, model kernel, and output ports."""
    validated = PreparedInferencePlan.model_validate(plan.model_dump(mode="python"))
    runtime = executor or RayMapBatchesExecutor(kernel_factory=kernel_factory)
    result = runtime.execute_prepared(
        validated,
        sink,
        opened_input,
        kernel_factory=kernel_factory,
    )
    return InferenceResult.model_validate(result.model_dump(mode="python"))


@PublicAPI(stability="alpha")
def run_inference(
    request: InferenceRequest,
    *,
    resolver: InferenceResolver | None = None,
    executor: InferenceExecutor | None = None,
    sink: ResultSink | None = None,
    bound_sink: BoundResultSink | None = None,
    sink_provider: ResultSinkProvider | None = None,
    kernel_provider: ModelKernelProvider | None = None,
) -> InferenceResult:
    """Resolve and execute one standalone or post-training request."""
    plan = resolve_inference(request, resolver=resolver)
    return run_resolved_inference(
        plan,
        executor=executor,
        sink=sink,
        bound_sink=bound_sink,
        sink_provider=sink_provider,
        kernel_provider=kernel_provider,
    )


def _default_model_kernel_provider() -> ModelKernelProvider:
    """Resolve the default model runtime through top-level composition."""
    from tributo.runtime import default_model_kernel_provider

    return default_model_kernel_provider()


def _default_result_sink_provider() -> ResultSinkProvider:
    """Resolve the default composition root without importing integrations."""
    from tributo.runtime import default_result_sink_provider

    return default_result_sink_provider()


__all__ = [
    "resolve_inference",
    "prepared_inference_plan",
    "run_inference",
    "run_prepared_inference",
    "run_resolved_inference",
]
