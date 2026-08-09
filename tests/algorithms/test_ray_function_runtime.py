"""Unit tests for restricted module-qualified user function execution."""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest

from tributo.algorithms.api import AlgorithmOperation, WorkerExecutionResult
from tributo.algorithms.core.worker import worker_bootstrap
from tributo.algorithms.input import FakeInputInvocation, FakeTabularPayload
from tributo.algorithms.spi import InputExecutionContext, RuntimeExecutionEnvelope

from .conftest import dispatcher_for, function_registration, request_for


class DirectWorkerRuntime:
    @property
    def runtime_id(self) -> str:
        return "tributo.ray_task"

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult:
        return worker_bootstrap(
            envelope.worker_envelope(0),
            {
                "worker_id": "direct-user-worker",
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


def test_user_function_reports_only_through_restricted_context(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(function_registration(), DirectWorkerRuntime())
    result = dispatcher.execute(
        request_for(
            "external_function",
            AlgorithmOperation.FIT,
            config={"threshold": 0.75},
        ),
        _context(binary_columns),
    )

    assert result.execution.status == "succeeded"
    assert result.execution.metrics == {"row_count": 8, "positive_rate": 0.5}
    assert result.execution.outputs["threshold"] == 0.75
    assert result.execution.outputs["worker_id"] == "direct-user-worker"
    assert [artifact.kind for artifact in result.execution.artifacts] == [
        "report",
        "checkpoint",
    ]


def test_user_exception_has_one_stable_failure_category(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        function_registration(
            "tests.support.portable_algorithms:failing_training_fragment"
        ),
        DirectWorkerRuntime(),
    )
    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.failure_category == "execution"
    assert result.execution.error_type == "AlgorithmExecutionError"
    assert result.execution.error_message is not None
    assert "ValueError: user-visible failure" in result.execution.error_message


def test_user_error_diagnostics_are_sanitized(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        function_registration(
            "tests.support.portable_algorithms:sensitive_failure_fragment"
        ),
        DirectWorkerRuntime(),
    )
    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "hunter2" not in result.execution.error_message
    assert "abc123" not in result.execution.error_message
    assert "alice:private" not in result.execution.error_message
    assert result.execution.error_message.count("<redacted>") == 3


def test_user_function_receives_cancellation_snapshot(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        function_registration(
            "tests.support.portable_algorithms:cancellation_aware_fragment"
        ),
        DirectWorkerRuntime(),
    )
    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
        cancelled=True,
    )

    assert result.execution.status == "succeeded"
    assert result.execution.outputs["cancelled"] is True


def test_user_return_value_cannot_bypass_reporting(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        function_registration(
            "tests.support.portable_algorithms:invalid_returning_fragment"
        ),
        DirectWorkerRuntime(),
    )
    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "must report through UserExecutionContext" in result.execution.error_message


def test_invalid_user_metrics_use_the_execution_failure_taxonomy(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        function_registration(
            "tests.support.portable_algorithms:invalid_reporting_fragment"
        ),
        DirectWorkerRuntime(),
    )
    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.failure_category == "execution"
    assert result.execution.error_message is not None
    assert "finite portable JSON values" in result.execution.error_message


@pytest.mark.parametrize(
    "reference",
    [
        "tests.support.portable_algorithms:callable_fragment",
        "tests.support.portable_algorithms:lambda_fragment",
        "tests.support.portable_algorithms:closure_fragment",
    ],
)
def test_user_channel_rejects_non_module_level_function_shapes(
    binary_columns: dict[str, tuple[object, ...]],
    reference: str,
) -> None:
    dispatcher = dispatcher_for(
        function_registration(reference),
        DirectWorkerRuntime(),
    )

    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "module-level function" in result.execution.error_message


def test_code_digest_is_verified_before_user_module_import(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    module = "tests.support.portable_algorithms"
    sys.modules.pop(module, None)
    registration = function_registration()
    registration = replace(
        registration,
        implementation=replace(
            registration.implementation,
            code_digest="0" * 64,
        ),
    )
    dispatcher = dispatcher_for(registration, DirectWorkerRuntime())

    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.failure_category == "dependency"
    assert result.execution.error_message is not None
    assert "code digest mismatch" in result.execution.error_message
    assert module not in sys.modules
