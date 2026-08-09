"""Unit tests for the managed scikit-learn Worker runtime."""

from __future__ import annotations

from dataclasses import replace

from tributo.algorithms.api import (
    AlgorithmOperation,
    QualifiedReference,
    WorkerExecutionResult,
)
from tributo.algorithms.core.worker import worker_bootstrap
from tributo.algorithms.input import FakeInputInvocation, FakeTabularPayload
from tributo.algorithms.spi import InputExecutionContext, RuntimeExecutionEnvelope

from .conftest import dispatcher_for, request_for, sklearn_registration


class DirectWorkerRuntime:
    """Run WorkerBootstrap in-process for fast runtime contract tests."""

    @property
    def runtime_id(self) -> str:
        return "tributo.ray_task"

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        return worker_bootstrap(
            envelope.worker_envelope(0),
            {
                "worker_id": "direct-worker",
                "node_id": "direct-node",
                "world_rank": 0,
                "world_size": 1,
            },
        )


def _context(
    columns: dict[str, tuple[object, ...]],
) -> InputExecutionContext:
    return InputExecutionContext(
        {"binary-fixture": FakeInputInvocation(FakeTabularPayload(columns))}
    )


def test_logistic_fit_evaluate_and_predict(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(sklearn_registration(), DirectWorkerRuntime())
    fit_result = dispatcher.execute(
        request_for(
            "external_sklearn",
            AlgorithmOperation.FIT,
            config={"C": 1.0, "max_iter": 200},
        ),
        _context(binary_columns),
    )

    assert fit_result.execution.status == "succeeded"
    assert fit_result.execution.metrics["accuracy"] == 1.0
    assert fit_result.execution.metrics["row_count"] == 8
    assert len(fit_result.execution.outputs["predictions"]) == 8
    assert fit_result.actual_versions["scikit-learn"]
    model = fit_result.execution.artifacts[0]
    assert model.kind == "model"
    assert model.trusted

    evaluate_result = dispatcher.execute(
        request_for("external_sklearn", AlgorithmOperation.EVALUATE),
        _context(binary_columns),
        artifacts=(model,),
    )
    assert evaluate_result.execution.status == "succeeded"
    assert evaluate_result.execution.metrics["accuracy"] == 1.0

    predict_result = dispatcher.execute(
        request_for("external_sklearn", AlgorithmOperation.PREDICT),
        _context(binary_columns),
        artifacts=(model,),
    )
    assert predict_result.execution.status == "succeeded"
    assert predict_result.execution.outputs["predictions"] == (
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    )


def test_sklearn_pipeline_is_cloned_and_persisted(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        sklearn_registration(pipeline=True), DirectWorkerRuntime()
    )
    result = dispatcher.execute(
        request_for(
            "external_sklearn",
            AlgorithmOperation.FIT,
            config={"model__C": 0.5},
        ),
        _context(binary_columns),
    )

    assert result.execution.status == "succeeded"
    assert result.execution.metrics["accuracy"] == 1.0
    assert result.execution.artifacts[0].sha256


def test_predict_rejects_missing_or_untrusted_model(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(sklearn_registration(), DirectWorkerRuntime())
    result = dispatcher.execute(
        request_for("external_sklearn", AlgorithmOperation.PREDICT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.failure_category == "execution"
    assert result.execution.error_message is not None
    assert "requires one model artifact" in result.execution.error_message

    fit = dispatcher.execute(
        request_for("external_sklearn", AlgorithmOperation.FIT),
        _context(binary_columns),
    )
    untrusted = replace(fit.execution.artifacts[0], trusted=False)
    result = dispatcher.execute(
        request_for("external_sklearn", AlgorithmOperation.PREDICT),
        _context(binary_columns),
        artifacts=(untrusted,),
    )
    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "untrusted pickle" in result.execution.error_message


def test_sklearn_input_and_target_errors_are_normalized(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(sklearn_registration(), DirectWorkerRuntime())
    request = request_for("external_sklearn", AlgorithmOperation.FIT)
    request_without_label = replace(
        request,
        input_binding=replace(request.input_binding, label_name=None),
    )

    missing_label = dispatcher.execute(
        request_without_label,
        _context(binary_columns),
    )
    assert missing_label.execution.status == "failed"
    assert missing_label.execution.failure_category == "input"
    assert missing_label.execution.error_message is not None
    assert "requires a label_name" in missing_label.execution.error_message

    continuous_target = dict(binary_columns)
    continuous_target["label"] = tuple(index / 10 for index in range(8))
    incompatible_target = dispatcher.execute(
        request,
        _context(continuous_target),
    )
    assert incompatible_target.execution.status == "failed"
    assert incompatible_target.execution.failure_category == "execution"
    assert incompatible_target.execution.error_type == "AlgorithmExecutionError"


def test_framework_managed_rejects_unbounded_n_jobs(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    registration = sklearn_registration(framework_managed=True)
    registration = replace(
        registration,
        implementation=replace(
            registration.implementation,
            implementation_ref=QualifiedReference.parse(
                "tests.support.portable_algorithms:unbounded_joblib_factory"
            ),
        ),
    )
    dispatcher = dispatcher_for(registration, DirectWorkerRuntime())

    result = dispatcher.execute(
        request_for("external_sklearn", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "n_jobs must be positive" in result.execution.error_message
