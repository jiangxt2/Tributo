"""Pre-publication evidence tests for framework-native algorithms."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

import tributo.integrations.algorithm_runtimes.framework_native as framework_native_module
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    FrameworkNativeInputRoute,
    FrameworkNativePolicy,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    WorkerResources,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm, PreparedInput
from tributo.exceptions import BundleExportError
from tributo.integrations.algorithm_runtimes.framework_native import (
    FrameworkNativeRuntime,
    _bundle_export_failure_message,
    _framework_execution_result,
    _framework_input_coverage_snapshot,
    _validated_framework_evidence,
    _validated_staged_framework_evidence,
)


def _worker(
    rank: int,
    *,
    digest: str = "a" * 64,
    rows_processed: int = 4,
    input_rows: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "worker_id": f"worker-{rank}",
        "node_id": f"node-{rank}",
        "rank": rank,
        "world_size": 2,
        "shard_id": f"shard-{rank}",
        "rows_processed": rows_processed,
        "input_rows": dict(input_rows or {}),
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


def test_framework_evidence_validates_partitioned_and_replicated_input_roles() -> None:
    evidence = _evidence()
    evidence["workers"] = [
        _worker(0, rows_processed=4, input_rows={"train": 4, "validation": 5}),
        _worker(1, rows_processed=4, input_rows={"train": 4, "validation": 5}),
    ]

    workers, _ = _validated_framework_evidence(
        evidence,
        worker_count=2,
        resources_per_worker=WorkerResources(),
        expected_training_rows=8,
        input_role_modes={"train": "partitioned", "validation": "replicated"},
        expected_input_role_rows={"train": 8, "validation": 5},
    )

    assert [worker["input_rows"]["validation"] for worker in workers] == [5, 5]


def test_framework_evidence_rejects_role_mismatch_before_export() -> None:
    evidence = _evidence()
    evidence["workers"] = [
        _worker(0, rows_processed=4, input_rows={"train": 4, "validation": 5}),
        _worker(1, rows_processed=4, input_rows={"train": 4, "validation": 4}),
    ]

    with pytest.raises(AlgorithmExecutionError, match="replicated input role"):
        _validated_framework_evidence(
            evidence,
            worker_count=2,
            resources_per_worker=WorkerResources(),
            expected_training_rows=8,
            input_role_modes={"train": "partitioned", "validation": "replicated"},
            expected_input_role_rows={"train": 8, "validation": 5},
        )


def test_framework_native_input_snapshot_counts_each_role_once() -> None:
    class CountedDataset:
        def __init__(self, role: str, rows: int, events: list[str]) -> None:
            self.role = role
            self.rows = rows
            self.events = events
            self.count_calls = 0

        def count(self) -> int:
            self.count_calls += 1
            self.events.append(f"count:{self.role}")
            return self.rows

    events: list[str] = []
    train = CountedDataset("train", 7, events)
    validation = CountedDataset("validation", 3, events)
    plan = SimpleNamespace(
        input_bindings=SimpleNamespace(
            bindings=(
                SimpleNamespace(name="train"),
                SimpleNamespace(name="validation"),
            )
        )
    )
    policy = FrameworkNativePolicy(
        framework="test-framework",
        evidence_collector_ref="tests.example:collect",
        input_routes=(
            FrameworkNativeInputRoute("train", "partitioned"),
            FrameworkNativeInputRoute("validation", "partitioned"),
        ),
    )

    expected_train, modes, expected_roles = _framework_input_coverage_snapshot(
        plan=cast(ResolvedAlgorithmPlan, plan),
        prepared_views={"train": object(), "validation": object()},
        datasets={"train": train, "validation": validation},
        policy=policy,
    )

    assert expected_train == 7
    assert modes == {"train": "partitioned", "validation": "partitioned"}
    assert expected_roles == {"train": 7, "validation": 3}
    assert train.count_calls == validation.count_calls == 1
    assert len(events) == 2


def test_framework_native_input_snapshot_preserves_legacy_train_only_route() -> None:
    class TrainingDataset:
        def count(self) -> int:
            return 7

    expected_rows, modes, role_rows = _framework_input_coverage_snapshot(
        plan=cast(
            ResolvedAlgorithmPlan,
            SimpleNamespace(
                input_bindings=SimpleNamespace(
                    bindings=(SimpleNamespace(name="train"),)
                )
            ),
        ),
        prepared_views={"train": object()},
        datasets={"train": TrainingDataset()},
        policy=FrameworkNativePolicy(
            framework="test-framework",
            evidence_collector_ref="tests.example:collect",
        ),
    )

    assert expected_rows == 7
    assert modes == {}
    assert role_rows == {}


def test_framework_native_multirole_input_requires_route_declarations() -> None:
    class TrainingDataset:
        def count(self) -> int:
            raise AssertionError("multi-role inputs must fail before count")

    with pytest.raises(AlgorithmConfigurationError, match="require declared"):
        _framework_input_coverage_snapshot(
            plan=cast(
                ResolvedAlgorithmPlan,
                SimpleNamespace(
                    input_bindings=SimpleNamespace(
                        bindings=(
                            SimpleNamespace(name="train"),
                            SimpleNamespace(name="validation"),
                        )
                    )
                ),
            ),
            prepared_views={"train": object(), "validation": object()},
            datasets={
                "train": TrainingDataset(),
                "validation": object(),
            },
            policy=FrameworkNativePolicy(
                framework="test-framework",
                evidence_collector_ref="tests.example:collect",
            ),
        )


def test_framework_native_input_snapshot_rejects_missing_required_role_before_count() -> (
    None
):
    class CountMustNotRun:
        def count(self) -> int:
            raise AssertionError("count must not run for an invalid binding")

    policy = FrameworkNativePolicy(
        framework="test-framework",
        evidence_collector_ref="tests.example:collect",
        input_routes=(
            FrameworkNativeInputRoute("train", "partitioned"),
            FrameworkNativeInputRoute("validation", "partitioned"),
        ),
    )

    with pytest.raises(AlgorithmConfigurationError, match="missing"):
        _framework_input_coverage_snapshot(
            plan=cast(
                ResolvedAlgorithmPlan,
                SimpleNamespace(
                    input_bindings=SimpleNamespace(
                        bindings=(
                            SimpleNamespace(name="train"),
                            SimpleNamespace(name="validation"),
                        )
                    )
                ),
            ),
            prepared_views={"train": object()},
            datasets={"train": CountMustNotRun()},
            policy=policy,
        )


def test_framework_native_runtime_counts_routes_before_trainer_and_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class CountedDataset:
        def __init__(self, role: str, rows: int) -> None:
            self.role = role
            self.rows = rows
            self.count_calls = 0

        def count(self) -> int:
            self.count_calls += 1
            events.append(f"count:{self.role}")
            return self.rows

    datasets = {
        "train": CountedDataset("train", 5),
        "validation": CountedDataset("validation", 5),
    }
    prepared = PreparedInput({role: object() for role in datasets})
    policy = FrameworkNativePolicy(
        framework="test-framework",
        evidence_collector_ref="tests.example:collect",
        input_routes=(
            FrameworkNativeInputRoute("train", "partitioned"),
            FrameworkNativeInputRoute("validation", "partitioned"),
        ),
    )

    class Trainer:
        def fit(self) -> object:
            events.append("fit")
            return SimpleNamespace(metrics={"score": 0.75})

    class Algorithm:
        def validate_environment(self) -> None:
            events.append("validate")

        def bind_datasets(self, _views: object) -> dict[str, CountedDataset]:
            events.append("bind")
            return datasets

        def build_trainer(self, _config: object, _datasets: object) -> Trainer:
            assert datasets["train"].count_calls == 1
            assert datasets["validation"].count_calls == 1
            events.append("build")
            return Trainer()

        def collect_evidence(self, _result: object) -> dict[str, object]:
            workers = [
                _worker(0, rows_processed=3, input_rows={"train": 3, "validation": 2}),
                _worker(1, rows_processed=2, input_rows={"train": 2, "validation": 3}),
            ]
            return {
                "workers": workers,
                "state": _evidence()["state"],
                "input_complete": True,
            }

        def checkpoint_source(self, _result: object) -> object:
            raise AssertionError("FIT_ONLY must not request a checkpoint")

    plan = SimpleNamespace(
        input_bindings=SimpleNamespace(
            bindings=(
                SimpleNamespace(name="train"),
                SimpleNamespace(name="validation"),
            )
        ),
        distribution_spec=SimpleNamespace(
            policy=policy,
            result_policy=ResultPolicy.FIT_ONLY,
        ),
        runtime=SimpleNamespace(
            worker_input_adapter_ref=object(),
            worker_count=2,
            num_cpus=1.0,
            num_gpus=0.0,
            memory_bytes=None,
            custom_resources={},
        ),
        algorithm_config={},
        environment=SimpleNamespace(python="3.12", dependencies=()),
        implementation=SimpleNamespace(exporter_ref=None),
    )
    envelope = SimpleNamespace(
        cancelled=False,
        plan=plan,
        input_payloads=(object(),),
        artifacts=(),
        run_id="coverage-run",
    )
    algorithm = Algorithm()
    monkeypatch.setattr(framework_native_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(framework_native_module, "_algorithm", lambda _env: algorithm)
    monkeypatch.setattr(
        framework_native_module,
        "_load_reference",
        lambda _reference: lambda _payload: prepared,
    )
    monkeypatch.setattr(
        framework_native_module, "_actual_environment_versions", lambda *_args: {}
    )

    result = FrameworkNativeRuntime().execute(envelope)

    assert events.index("build") > max(
        events.index("count:train"), events.index("count:validation")
    )
    assert events.index("fit") > events.index("build")
    assert result.execution.status == "succeeded"
    assert result.worker_metadata["input_complete"] is True
    assert prepared.closed


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


def test_staged_framework_evidence_checks_role_coverage_for_every_fold() -> None:
    folds = []
    for train_rows in (3, 3):
        fold = _evidence()
        fold["workers"] = [
            _worker(
                0,
                rows_processed=train_rows,
                input_rows={"train": train_rows, "validation": 5},
            ),
            _worker(
                1,
                rows_processed=train_rows,
                input_rows={"train": train_rows, "validation": 5},
            ),
        ]
        fold["expected_training_rows"] = 6
        fold["heldout_rows"] = 4
        folds.append(fold)
    evidence = {
        "stages": {"model": {"cross_fit_folds": folds}},
        "composition_digest": "d" * 64,
    }

    workers, state = _validated_staged_framework_evidence(
        evidence,
        component_stages=("model",),
        worker_count=2,
        resources_per_worker=WorkerResources(),
        expected_training_rows=8,
        input_role_modes={"train": "partitioned", "validation": "replicated"},
        expected_input_role_rows={"train": 8, "validation": 5},
    )

    assert [worker["rows_processed"] for worker in workers] == [6, 6]
    assert [worker["input_rows"]["train"] for worker in workers] == [6, 6]
    assert [worker["input_rows"]["validation"] for worker in workers] == [10, 10]
    assert state["details"]["component_stages"] == "model"
    assert state["details"]["stage.model.input_rows_scope"] == "cumulative_across_folds"


def test_staged_framework_evidence_rejects_bad_role_coverage_in_any_fold() -> None:
    folds = []
    for validation_rows in (5, 4):
        fold = _evidence()
        fold["workers"] = [
            _worker(0, rows_processed=3, input_rows={"train": 3, "validation": 5}),
            _worker(
                1,
                rows_processed=3,
                input_rows={"train": 3, "validation": validation_rows},
            ),
        ]
        fold["expected_training_rows"] = 6
        fold["heldout_rows"] = 4
        folds.append(fold)
    evidence = {
        "stages": {"model": {"cross_fit_folds": folds}},
        "composition_digest": "d" * 64,
    }

    with pytest.raises(AlgorithmExecutionError, match="replicated input role"):
        _validated_staged_framework_evidence(
            evidence,
            component_stages=("model",),
            worker_count=2,
            resources_per_worker=WorkerResources(),
            expected_training_rows=8,
            input_role_modes={"train": "partitioned", "validation": "replicated"},
            expected_input_role_rows={"train": 8, "validation": 5},
        )


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
            "coverage rows do not partition input",
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


def test_bundle_failure_preserves_failed_node_diagnostics() -> None:
    error = BundleExportError(
        "Bundle export failed: failed",
        execution_result=SimpleNamespace(
            node_results=(
                SimpleNamespace(status="succeeded"),
                SimpleNamespace(
                    node_id="x-learner-onnx",
                    status="failed",
                    exporter_id="official-x-learner-onnx-v1",
                    failure=SimpleNamespace(
                        code="INVALID_GRAPH",
                        category="validation",
                        message="ONNX Runtime rejected the graph",
                    ),
                ),
            )
        ),
    )

    message = _bundle_export_failure_message(error)

    assert "x-learner-onnx" in message
    assert "INVALID_GRAPH" in message
    assert "ONNX Runtime rejected the graph" in message


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
