"""Fail-closed evidence tests for the Ray Train collective runtime."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    ResolvedAlgorithmPlan,
    ResultPolicy,
)
from tributo.integrations.algorithm_runtimes.collective import (
    _collective_execution_result,
    _worker_evidence,
)


def _worker(rank: int, *, digest: str = "a" * 64) -> dict[str, object]:
    return {
        "worker_id": f"worker-{rank}",
        "node_id": f"node-{rank}",
        "rank": rank,
        "world_size": 2,
        "shard_id": f"shard-{rank}",
        "rows_processed": 4,
        "input_rows": {"train": 4},
        "batch_count": 1,
        "collective_steps": 1,
        "model_state_digest": digest,
        "resources": {"num_cpus": 1.0, "num_gpus": 0.0, "custom": {}},
    }


def test_collective_evidence_requires_unique_complete_synchronized_workers() -> None:
    workers, digest = _worker_evidence(
        {
            "execution_workers": [_worker(1), _worker(0)],
            "model_state_digest": "a" * 64,
        },
        worker_count=2,
        num_cpus=1,
        num_gpus=0,
        custom_resources={},
        expected_input_rows={"train": 8},
    )

    assert [worker["rank"] for worker in workers] == [0, 1]
    assert digest == "a" * 64


@pytest.mark.parametrize(
    "workers,digest,match",
    [
        ([_worker(0)], "a" * 64, "every requested worker"),
        ([_worker(0), _worker(0)], "a" * 64, "unique ranks"),
        ([_worker(0), _worker(1, digest="b" * 64)], "a" * 64, "synchronized"),
    ],
)
def test_collective_evidence_rejects_unproved_distribution(
    workers: list[dict[str, object]],
    digest: str,
    match: str,
) -> None:
    with pytest.raises(AlgorithmExecutionError, match=match):
        _worker_evidence(
            {"execution_workers": workers, "model_state_digest": digest},
            worker_count=2,
            num_cpus=1,
            num_gpus=0,
            custom_resources={},
            expected_input_rows={"train": 8},
        )


def test_collective_evidence_requires_declared_custom_resources() -> None:
    workers = [_worker(0), _worker(1)]

    with pytest.raises(AlgorithmExecutionError, match="custom resources"):
        _worker_evidence(
            {
                "execution_workers": workers,
                "model_state_digest": "a" * 64,
            },
            worker_count=2,
            num_cpus=1,
            num_gpus=0,
            custom_resources={"accelerator_type_a": 0.25},
            expected_input_rows={"train": 8},
        )


def test_collective_evidence_rejects_input_facts_that_cannot_prove_coverage() -> None:
    workers = [_worker(0), _worker(1)]
    workers[0]["input_rows"] = {}

    with pytest.raises(AlgorithmExecutionError, match="input batches"):
        _worker_evidence(
            {
                "execution_workers": workers,
                "model_state_digest": "a" * 64,
            },
            worker_count=2,
            num_cpus=1,
            num_gpus=0,
            custom_resources={},
            expected_input_rows={"train": 8},
        )


def test_collective_evidence_rejects_incomplete_global_row_coverage() -> None:
    workers = [_worker(0), _worker(1)]
    workers[1]["input_rows"] = {"train": 3}

    with pytest.raises(AlgorithmExecutionError, match="complete input coverage"):
        _worker_evidence(
            {
                "execution_workers": workers,
                "model_state_digest": "a" * 64,
            },
            worker_count=2,
            num_cpus=1,
            num_gpus=0,
            custom_resources={},
            expected_input_rows={"train": 8},
        )


def test_collective_fit_only_skips_exporter_and_keeps_portable_metrics() -> None:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            distribution_spec=SimpleNamespace(result_policy=ResultPolicy.FIT_ONLY),
            implementation=SimpleNamespace(exporter_ref=None),
        ),
    )

    execution = _collective_execution_result(
        result=object(),
        metrics={
            "loss": 0.25,
            "execution_workers": [_worker(0), _worker(1)],
            "model_state_digest": "a" * 64,
            "world_size": 2,
            "state_coordination": "all_reduce",
            "collective_backend": "gloo",
            "checkpoint_owner_rank": 0,
            "metric_reducers": {"loss": "weighted_mean"},
        },
        plan=plan,
        run_id="run-1",
    )

    assert execution.status == "succeeded"
    assert execution.metrics == {"loss": 0.25}
    assert execution.outputs == {}
    assert execution.artifacts == ()


def test_collective_fit_only_rejects_nonportable_user_metrics() -> None:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            distribution_spec=SimpleNamespace(result_policy=ResultPolicy.FIT_ONLY),
            implementation=SimpleNamespace(exporter_ref=None),
        ),
    )

    with pytest.raises(AlgorithmExecutionError, match="FIT_ONLY user metrics"):
        _collective_execution_result(
            result=object(),
            metrics={"loss": object()},
            plan=plan,
            run_id="run-1",
        )


def test_collective_fit_only_rejects_non_string_metric_names() -> None:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            distribution_spec=SimpleNamespace(result_policy=ResultPolicy.FIT_ONLY),
            implementation=SimpleNamespace(exporter_ref=None),
        ),
    )

    with pytest.raises(AlgorithmExecutionError, match="metric names must be strings"):
        _collective_execution_result(
            result=object(),
            metrics={1: 0.25},
            plan=plan,
            run_id="run-1",
        )
