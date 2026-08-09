"""Unit tests for data-parallel Worker-group validation and reduction."""

from __future__ import annotations

from tributo.algorithms.api import (
    AlgorithmExecutionResult,
    AlgorithmOperation,
    ResolvedAlgorithmPlan,
    WorkerExecutionResult,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.core.worker import reduce_worker_group
from tributo.algorithms.input import FakeInputResolver

from .conftest import function_registration, request_for


def _plan() -> ResolvedAlgorithmPlan:
    registry = AlgorithmRegistrationRegistry()
    registry.register(function_registration(data_parallel=True))
    resolver = FakeInputResolver()
    return AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for("external_function", AlgorithmOperation.FIT)
    )


def _result(
    rank: int,
    *,
    worker_id: str,
    python_version: str = "3.12.12",
) -> WorkerExecutionResult:
    return WorkerExecutionResult(
        execution=AlgorithmExecutionResult(status="succeeded"),
        actual_versions={"python": python_version},
        worker_metadata={
            "world_rank": rank,
            "world_size": 2,
            "worker_id": worker_id,
        },
    )


def test_worker_group_rejects_duplicate_worker_identity_before_reducer() -> None:
    result = reduce_worker_group(
        _plan(),
        (
            _result(0, worker_id="same-worker"),
            _result(1, worker_id="same-worker"),
        ),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "distinct Ray Workers" in result.execution.error_message


def test_worker_group_rejects_inconsistent_versions_before_reducer() -> None:
    result = reduce_worker_group(
        _plan(),
        (
            _result(0, worker_id="worker-0"),
            _result(1, worker_id="worker-1", python_version="3.13.0"),
        ),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "inconsistent dependency versions" in result.execution.error_message
