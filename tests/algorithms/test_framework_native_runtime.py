"""Pre-publication evidence tests for framework-native algorithms."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tributo.algorithms.api import AlgorithmExecutionError, WorkerResources
from tributo.integrations.algorithm_runtimes.framework_native import (
    _validated_framework_evidence,
)


def _worker(rank: int, *, digest: str = "a" * 64) -> dict[str, object]:
    return {
        "worker_id": f"worker-{rank}",
        "node_id": f"node-{rank}",
        "rank": rank,
        "world_size": 2,
        "shard_id": f"shard-{rank}",
        "rows_processed": 4,
        "model_state_digest": digest,
        "resources": {"num_cpus": 1.0, "num_gpus": 0.0, "custom": {}},
    }


def _evidence() -> dict[str, object]:
    return {
        "workers": [_worker(1), _worker(0)],
        "state": {
            "coordination": "framework_native",
            "synchronized": True,
            "bounded": True,
            "global_model_digest": "a" * 64,
            "details": {"framework": "example"},
        },
        "input_complete": True,
    }


def test_framework_evidence_is_normalized_before_export() -> None:
    workers, state = _validated_framework_evidence(
        _evidence(),
        worker_count=2,
        resources_per_worker=WorkerResources(),
        expected_training_rows=8,
    )

    assert [worker["rank"] for worker in workers] == [0, 1]
    assert state["global_model_digest"] == "a" * 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(input_complete=False), "input coverage"),
        (
            lambda value: value["workers"][1].update(model_state_digest="b" * 64),
            "synchronized model",
        ),
        (
            lambda value: value["workers"][1].update(rows_processed=0),
            "resources and input",
        ),
        (
            lambda value: value["workers"][1].update(rows_processed=5),
            "complete training input coverage",
        ),
    ],
)
def test_framework_evidence_fails_before_bundle_publication(
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    evidence = _evidence()
    mutation(evidence)

    with pytest.raises(AlgorithmExecutionError, match=match):
        _validated_framework_evidence(
            evidence,
            worker_count=2,
            resources_per_worker=WorkerResources(),
            expected_training_rows=8,
        )
