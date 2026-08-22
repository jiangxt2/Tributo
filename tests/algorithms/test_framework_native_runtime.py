"""Pre-publication evidence tests for framework-native algorithms."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    WorkerResources,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo.integrations.algorithm_runtimes.framework_native import (
    _framework_execution_result,
    _validated_framework_evidence,
    _validated_staged_framework_evidence,
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


def test_staged_framework_evidence_validates_each_component_and_composes_digest() -> (
    None
):
    stages = {}
    for name, digest in (
        ("mu0", "a" * 64),
        ("mu1", "b" * 64),
        ("propensity", "c" * 64),
    ):
        payload = _evidence()
        payload["workers"] = [_worker(0, digest=digest), _worker(1, digest=digest)]
        payload["state"]["global_model_digest"] = digest
        payload["expected_training_rows"] = 8
        stages[name] = payload

    workers, state = _validated_staged_framework_evidence(
        {"stages": stages, "composition_digest": "d" * 64},
        component_stages=("mu0", "mu1", "propensity"),
        worker_count=2,
        resources_per_worker=WorkerResources(),
        expected_training_rows=8,
    )

    assert len(workers) == 2
    assert len(state["global_model_digest"]) == 64
    assert state["details"]["component_stages"] == "mu0,mu1,propensity"
    assert state["details"]["composition_digest"] == "d" * 64


def test_staged_framework_evidence_rejects_undeclared_or_missing_stage() -> None:
    with pytest.raises(AlgorithmExecutionError, match="declared component stages"):
        _validated_staged_framework_evidence(
            {"stages": {"mu0": {}}, "composition_digest": "d" * 64},
            component_stages=("mu0", "mu1"),
            worker_count=2,
            resources_per_worker=WorkerResources(),
            expected_training_rows=8,
        )


def test_staged_framework_evidence_requires_composition_digest() -> None:
    with pytest.raises(AlgorithmExecutionError, match="composition digest"):
        _validated_staged_framework_evidence(
            {"stages": {}},
            component_stages=(),
            worker_count=2,
            resources_per_worker=WorkerResources(),
            expected_training_rows=8,
        )


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


def test_framework_native_fit_only_skips_checkpoint_and_exporter() -> None:
    class CheckpointMustNotRun:
        def checkpoint_source(self, _result: object) -> object:
            raise AssertionError("checkpoint publication source was requested")

    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            distribution_spec=SimpleNamespace(result_policy=ResultPolicy.FIT_ONLY),
            implementation=SimpleNamespace(exporter_ref=None),
        ),
    )
    result = SimpleNamespace(
        metrics={
            "score": 0.75,
            "execution_workers": [_worker(0), _worker(1)],
            "model_state_digest": "a" * 64,
            "world_size": 2,
            "state_coordination": "framework_native",
        }
    )

    execution = _framework_execution_result(
        algorithm=cast(FrameworkNativeAlgorithm, CheckpointMustNotRun()),
        result=result,
        plan=plan,
        run_id="run-1",
    )

    assert execution.status == "succeeded"
    assert execution.metrics == {"score": 0.75}
    assert execution.outputs == {}
    assert execution.artifacts == ()


def test_framework_native_fit_only_rejects_nonportable_user_metrics() -> None:
    class CheckpointMustNotRun:
        def checkpoint_source(self, _result: object) -> object:
            raise AssertionError("checkpoint publication source was requested")

    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            distribution_spec=SimpleNamespace(result_policy=ResultPolicy.FIT_ONLY),
            implementation=SimpleNamespace(exporter_ref=None),
        ),
    )

    with pytest.raises(AlgorithmExecutionError, match="FIT_ONLY user metrics"):
        _framework_execution_result(
            algorithm=cast(FrameworkNativeAlgorithm, CheckpointMustNotRun()),
            result=SimpleNamespace(metrics={"score": object()}),
            plan=plan,
            run_id="run-1",
        )


def test_framework_native_fit_only_rejects_non_mapping_metrics() -> None:
    class CheckpointMustNotRun:
        def checkpoint_source(self, _result: object) -> object:
            raise AssertionError("checkpoint publication source was requested")

    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            distribution_spec=SimpleNamespace(result_policy=ResultPolicy.FIT_ONLY),
            implementation=SimpleNamespace(exporter_ref=None),
        ),
    )

    with pytest.raises(AlgorithmExecutionError, match="must be a mapping"):
        _framework_execution_result(
            algorithm=cast(FrameworkNativeAlgorithm, CheckpointMustNotRun()),
            result=SimpleNamespace(metrics=object()),
            plan=plan,
            run_id="run-1",
        )
