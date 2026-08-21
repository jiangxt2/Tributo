"""Training execution controls stay independent from the public Broker API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tributo.integrations.broker as broker_contract
from tributo.training.execution_context import (
    ExecutionContext,
    TrainingCancelledError,
    TrainingControlSpec,
)
from tributo.training.progress import TrainingPhase, TrainingProgress
from tributo.training.xgboost_trainer import (
    RoundMetricsReporter,
    _raise_if_training_cancelled,
    _resolve_worker_event_reporting,
)


def test_execution_context_rebuilds_controls_from_explicit_factory_refs() -> None:
    class Checker:
        def is_cancelled(self, job_id: str) -> bool:
            return False

    class Reporter:
        def report_phase(self, job_id: str, phase: str) -> None:
            return None

        def report_metrics(self, job_id, metrics, progress=None) -> None:
            return None

    checker = Checker()
    reporter = Reporter()
    module = SimpleNamespace(
        checker=lambda *, job_id, options: checker,
        reporter=lambda *, job_id, options: reporter,
    )
    context = ExecutionContext(
        cancellation=TrainingControlSpec("provider.controls:checker", "job-1"),
        event_reporter=TrainingControlSpec("provider.controls:reporter", "job-1"),
    )

    with patch(
        "tributo.training.execution_context.importlib.import_module",
        return_value=module,
    ):
        assert context.build_cancellation_checker() is checker
        assert context.build_event_reporter() is reporter

    assert not hasattr(broker_contract, "EventReporter")
    assert not hasattr(broker_contract, "CancellationChecker")


def test_execution_context_rejects_inline_credentials() -> None:
    with pytest.raises(ValueError, match="credential"):
        TrainingControlSpec(
            "provider.controls:checker",
            "job-1",
            {"redis_url": "redis://user:password@redis:6379/0"},
        )


def test_phase_checks_cancellation_before_reporting() -> None:
    calls: list[str] = []
    checker = MagicMock()
    checker.is_cancelled.side_effect = lambda job_id: calls.append("cancel") or False
    reporter = MagicMock()
    reporter.report_phase.side_effect = lambda job_id, phase: calls.append(phase)
    progress = TrainingProgress("job-1", reporter=reporter, checker=checker)

    progress.report_phase(TrainingPhase.LOADING_DATA)

    assert calls == ["cancel", "LOADING_DATA"]


def test_confirmed_cancel_raises_before_nonterminal_event() -> None:
    checker = MagicMock()
    checker.is_cancelled.return_value = True
    reporter = MagicMock()
    progress = TrainingProgress("job-1", reporter=reporter, checker=checker)

    with pytest.raises(TrainingCancelledError, match="job-1"):
        progress.report_phase(TrainingPhase.TRAINING)

    reporter.report_phase.assert_not_called()


def test_round_metrics_are_sampled_and_early_stop_final_is_kept() -> None:
    reporter = MagicMock()
    rounds = RoundMetricsReporter("job-1", reporter, total_rounds=250)

    for epoch in range(7):
        rounds.after_iteration(
            epoch,
            {"train": {"logloss": [0.9 - 0.1 * i for i in range(epoch + 1)]}},
        )
    rounds.after_training()

    assert [
        call.args[1]["round"] for call in reporter.report_metrics.call_args_list
    ] == [1, 3, 6, 7]
    assert reporter.report_metrics.call_args_list[-1].args[2] == pytest.approx(7 / 250)


def test_resume_metrics_keep_absolute_round_numbers() -> None:
    reporter = MagicMock()
    rounds = RoundMetricsReporter("job-1", reporter, total_rounds=10, start_round=7)

    rounds.after_iteration(0, {"train": {"loss": [0.2]}})
    rounds.after_training(8, {"train": {"loss": [0.2]}})

    assert reporter.report_metrics.call_args.args[1]["round"] == 8


def test_worker_cancel_is_dedicated_error_and_nonzero_rank_has_no_reporter() -> None:
    checker = MagicMock()
    checker.is_cancelled.return_value = True
    with pytest.raises(TrainingCancelledError, match="job-1"):
        _raise_if_training_cancelled("job-1", checker)

    with patch(
        "tributo.training.xgboost_trainer.ExecutionContext.from_environment"
    ) as from_environment:
        assert _resolve_worker_event_reporting(1) == (None, None)
    from_environment.assert_not_called()


def test_worker_callback_checks_each_round_reports_rank_zero_and_enters_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the callbacks through the real worker-loop boundary wiring."""
    pytest.importorskip("xgboost")
    import pyarrow as pa
    import ray.train
    import ray.train.xgboost
    import xgboost

    from tributo.training.xgboost_trainer import train_loop_per_worker

    reporter = MagicMock()
    checker = MagicMock()
    checker.is_cancelled.return_value = False
    monkeypatch.setattr(
        "tributo.training.xgboost_trainer._resolve_worker_cancellation",
        lambda: ("job-1", checker),
    )
    monkeypatch.setattr(
        "tributo.training.xgboost_trainer._resolve_worker_event_reporting",
        lambda _rank: ("job-1", reporter),
    )
    monkeypatch.setattr(
        ray.train,
        "get_context",
        lambda: SimpleNamespace(get_world_rank=lambda: 0, get_world_size=lambda: 1),
    )
    monkeypatch.setattr(ray.train, "report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ray.train.xgboost,
        "XGBoostCheckpoint",
        SimpleNamespace(from_model=lambda _model: None),
    )

    class FakeDMatrix:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def num_col(self) -> int:
            return 1

    class FakeBooster:
        def get_score(self, importance_type: str = "gain") -> dict[str, float]:
            return {}

        def num_boosted_rounds(self) -> int:
            return 2

    def fake_train(*_args: Any, **kwargs: Any) -> FakeBooster:
        log = {"train": {"logloss": []}}
        for epoch, value in enumerate((0.6, 0.4)):
            log["train"]["logloss"].append(value)
            for callback in kwargs["callbacks"]:
                callback.after_iteration(FakeBooster(), epoch, log)
        kwargs["evals_result"].update(log)
        return FakeBooster()

    table = pa.table({"feature": pa.array([1, 2]), "label": pa.array([0, 1])})

    class FakeShard:
        def schema(self) -> Any:
            return table.schema

        def iter_batches(self, **_kwargs: Any):
            yield table

    monkeypatch.setattr(xgboost, "QuantileDMatrix", FakeDMatrix)
    monkeypatch.setattr(xgboost, "train", fake_train)
    monkeypatch.setattr(
        ray.train,
        "get_dataset_shard",
        lambda key: FakeShard() if key == "train" else None,
    )

    train_loop_per_worker(
        {
            "label_col": "label",
            "xgb_params": {"objective": "binary:logistic"},
            "num_rounds": 2,
            "resource": {},
        }
    )

    assert [
        call.args[1]["round"] for call in reporter.report_metrics.call_args_list
    ] == [1, 2]
    reporter.report_phase.assert_called_once_with("job-1", "EVALUATING")
    assert checker.is_cancelled.call_count >= 4
