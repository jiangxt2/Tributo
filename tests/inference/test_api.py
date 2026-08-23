"""Tests for framework-neutral inference entry points."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.inference.test_executor import _plan
from tributo.inference.api import (
    prepared_inference_plan,
    run_inference,
    run_prepared_inference,
    run_resolved_inference,
)
from tributo.inference.contracts import (
    BundleModelReference,
    FailureDiagnostic,
    InferenceRequest,
    InferenceResult,
)


def _failed_result() -> InferenceResult:
    plan = _plan()
    return InferenceResult(
        run_id=plan.run_id,
        attempt_id=plan.attempt_id,
        submission_id=plan.submission_id,
        parent_run_id=plan.parent_run_id,
        plan_digest=plan.plan_digest,
        bundle_id=plan.model.bundle_ref.bundle_id,
        manifest_sha256=plan.model.bundle_ref.manifest_sha256,
        role=plan.model.role,
        flavor_id=plan.model.flavor_id,
        source_ref_id=plan.input.descriptor.source_ref,
        status="failed",
        failure=FailureDiagnostic(
            phase="execution",
            code="inference_execution_failed",
            error_type="ConformanceError",
        ),
    )


def _request() -> InferenceRequest:
    plan = _plan()
    return InferenceRequest(
        model=BundleModelReference(uri=plan.model.bundle_ref.canonical_uri),
        input=plan.input.request.model_copy(update={"binding_id": None}),
        input_binding=plan.input_binding,
        output_binding=plan.output_binding,
        result_sink=plan.result_sink,
        execution=plan.execution,
        run_id=plan.run_id,
    )


def test_run_inference_resolves_once_then_executes_same_plan() -> None:
    request = _request()
    plan = _plan()
    result = _failed_result()
    resolver = MagicMock()
    resolver.resolve.return_value = plan
    executor = MagicMock()
    executor.execute.return_value = result
    sink = object()

    actual = run_inference(
        request,
        resolver=resolver,
        executor=executor,
        sink=sink,
    )

    assert actual == result
    assert actual is not result
    resolved_request = resolver.resolve.call_args.args[0]
    assert resolved_request == request
    assert resolved_request is not request
    executed_plan = executor.execute.call_args.args[0]
    assert executed_plan == plan
    assert executed_plan is not plan
    assert executor.execute.call_args.args[1] is sink


def test_run_resolved_inference_does_not_resolve_again() -> None:
    plan = _plan()
    result = _failed_result()
    executor = MagicMock()
    executor.execute.return_value = result
    sink = object()

    actual = run_resolved_inference(plan, executor=executor, sink=sink)

    assert actual == result
    assert actual is not result
    executed_plan = executor.execute.call_args.args[0]
    assert executed_plan == plan
    assert executed_plan is not plan
    assert executor.execute.call_args.args[1] is sink


def test_run_prepared_inference_uses_only_injected_ports() -> None:
    plan = prepared_inference_plan(_plan())
    result = _failed_result()
    executor = MagicMock()
    executor.execute_prepared.return_value = result
    opened_input = object()
    kernel_factory = object()
    sink = object()

    actual = run_prepared_inference(
        plan,
        opened_input=opened_input,
        kernel_factory=kernel_factory,
        sink=sink,
        executor=executor,
    )

    assert actual == result
    args = executor.execute_prepared.call_args.args
    assert args[0] == plan
    assert args[0] is not plan
    assert args[1] is sink
    assert args[2] is opened_input
    assert (
        executor.execute_prepared.call_args.kwargs["kernel_factory"] is kernel_factory
    )


def test_default_executor_receives_injected_model_kernel_factory() -> None:
    plan = _plan()
    result = _failed_result()
    provider = MagicMock()
    factory = object()
    provider.prediction_factory.return_value = factory
    runtime = MagicMock()
    runtime.execute.return_value = result

    with patch(
        "tributo.inference.api.RayMapBatchesExecutor",
        return_value=runtime,
    ) as executor_type:
        actual = run_resolved_inference(
            plan,
            sink=object(),
            kernel_provider=provider,
        )

    assert actual == result
    provider.prediction_factory.assert_called_once_with(plan.model)
    executor_type.assert_called_once_with(kernel_factory=factory)


def test_default_execution_binds_output_before_entering_core() -> None:
    plan = _plan()
    result = _failed_result()
    sink_provider = MagicMock()
    bound_sink = object()
    sink_provider.bind.return_value = bound_sink
    kernel_provider = MagicMock()
    kernel_provider.prediction_factory.return_value = object()
    runtime = MagicMock()
    runtime.execute_bound.return_value = result

    with patch(
        "tributo.inference.api.RayMapBatchesExecutor",
        return_value=runtime,
    ):
        actual = run_resolved_inference(
            plan,
            sink_provider=sink_provider,
            kernel_provider=kernel_provider,
        )

    assert actual == result
    sink_provider.bind.assert_called_once_with(plan.result_sink)
    runtime.execute_bound.assert_called_once()
    assert runtime.execute_bound.call_args.args[1] is bound_sink


def test_inference_api_does_not_import_concrete_sink_integrations() -> None:
    path = Path(__file__).parents[2] / "src/tributo/inference/api.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("tributo.integrations") for module in imports)
