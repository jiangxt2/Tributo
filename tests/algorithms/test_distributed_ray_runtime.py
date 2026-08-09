"""Local-Ray functional tests for distributed portable execution."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import ray

from tributo.algorithms.api import AlgorithmOperation, QualifiedReference
from tributo.algorithms.input import FakeInputInvocation, FakeTabularPayload
from tributo.algorithms.spi import InputExecutionContext
from tributo.integrations.algorithm_runtimes.ray_task import RayTaskRuntime

from .conftest import (
    dispatcher_for,
    function_registration,
    request_for,
    sklearn_registration,
)

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


@pytest.fixture(scope="module", autouse=True)
def local_ray() -> Iterator[None]:
    """Use a small local Ray runtime for fast multi-Worker semantics."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tip: In future versions of Ray.*",
            category=FutureWarning,
            module="ray._private.worker",
        )
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def _context(binary_columns: dict[str, tuple[object, ...]]) -> InputExecutionContext:
    return InputExecutionContext(
        {"binary-fixture": FakeInputInvocation(FakeTabularPayload(binary_columns))}
    )


def test_custom_function_covers_disjoint_shards_once(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        function_registration(data_parallel=True),
        RayTaskRuntime(),
    )

    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    expected_values = list(binary_columns["x0"])
    assert result.execution.status == "succeeded"
    assert result.execution.metrics == {"row_count": 8, "positive_rate": 0.5}
    assert result.execution.outputs["ranks"] == (0, 1)
    assert sorted(result.execution.outputs["shard_values"]) == expected_values
    assert len(set(result.execution.outputs["worker_ids"])) == 2
    workers = result.worker_metadata["workers"]
    assert [worker["world_rank"] for worker in workers] == [0, 1]
    assert all(worker["world_size"] == 2 for worker in workers)
    assert result.worker_metadata["reducer_worker"]["worker_id"]


def test_custom_function_group_failure_skips_reducer(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    registration = function_registration(
        "tests.support.portable_algorithms:fail_rank_one_fragment",
        data_parallel=True,
    )
    registration = replace(
        registration,
        runtime=replace(
            registration.runtime,
            result_reducer_ref=QualifiedReference.parse(
                "tests.support.portable_algorithms:reducer_must_not_run"
            ),
        ),
    )
    dispatcher = dispatcher_for(registration, RayTaskRuntime())

    result = dispatcher.execute(
        request_for("external_function", AlgorithmOperation.FIT),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "rank-one failure" in result.execution.error_message
    assert "reducer was called" not in result.execution.error_message


def test_sklearn_framework_managed_uses_ray_joblib_workers(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        sklearn_registration(framework_managed=True),
        RayTaskRuntime(),
    )

    result = dispatcher.execute(
        request_for(
            "external_sklearn",
            AlgorithmOperation.FIT,
            config={"n_jobs": 2, "task_count": 6},
        ),
        _context(binary_columns),
    )

    assert result.execution.status == "succeeded"
    estimator: Any = pickle.loads(result.execution.artifacts[0].payload)
    evidence = estimator.worker_evidence_
    assert len(evidence) == 6
    assert len({item["worker_id"] for item in evidence}) >= 2
    assert result.worker_metadata["framework_parallelism"] == 2
    assert result.worker_metadata["topology"] == "framework_managed"


def test_sklearn_framework_parallelism_fails_closed(
    binary_columns: dict[str, tuple[object, ...]],
) -> None:
    dispatcher = dispatcher_for(
        sklearn_registration(framework_managed=True),
        RayTaskRuntime(),
    )

    result = dispatcher.execute(
        request_for(
            "external_sklearn",
            AlgorithmOperation.FIT,
            config={"n_jobs": 3, "task_count": 6},
        ),
        _context(binary_columns),
    )

    assert result.execution.status == "failed"
    assert result.execution.error_message is not None
    assert "declared framework parallelism" in result.execution.error_message
