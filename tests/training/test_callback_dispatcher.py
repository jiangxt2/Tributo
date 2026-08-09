"""Unit tests for training/callbacks.py — CallbackDispatcher."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tributo.training.base import BaseTrainer
from tributo.training.callbacks import CallbackDispatcher


class _FakeTrainer(BaseTrainer):
    def setup(self) -> None:
        pass

    def training_loop(self) -> Any:
        return "checkpoint"


class _RecordingCallback:
    """Implements every lifecycle hook and records invocations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, *args))

    def on_setup_start(self, trainer: BaseTrainer) -> None:
        self._record("on_setup_start", trainer, None)

    def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
        self._record("on_training_end", trainer, result)

    def on_export_end(self, trainer: BaseTrainer, output_path: str) -> None:
        self._record("on_export_end", trainer, output_path)

    def on_artifacts_exported(self, trainer: BaseTrainer, output_path: str) -> None:
        self._record("on_artifacts_exported", trainer, output_path)

    def on_run_complete(self, trainer: BaseTrainer, summary: dict[str, Any]) -> None:
        self._record("on_run_complete", trainer, summary)

    def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None:
        self._record("on_run_error", trainer, error)


class _LegacyCallback(_RecordingCallback):
    """Implements only the legacy hooks.

    ``on_artifacts_exported`` lives on the parent class, so the
    dispatcher's ``type(cb).__dict__`` check treats this as a legacy
    callback that falls back to ``on_export_end``.
    """


def _trainer() -> _FakeTrainer:
    return _FakeTrainer(datasets={}, config={})


class TestEventDispatch:
    def test_events_fire_in_order_with_args(self) -> None:
        cb = _RecordingCallback()
        dispatcher = CallbackDispatcher([cb])
        trainer = _trainer()

        dispatcher.on_setup_start(trainer)
        dispatcher.on_training_end(trainer, "ckpt")
        dispatcher.on_artifacts_exported(trainer, "/tmp/out")
        dispatcher.on_run_complete(trainer, {"status": "succeeded"})

        assert [c[0] for c in cb.calls] == [
            "on_setup_start",
            "on_training_end",
            "on_artifacts_exported",
            "on_run_complete",
        ]
        assert cb.calls[1][2] == "ckpt"
        assert cb.calls[2][2] == "/tmp/out"

    def test_empty_dispatcher_is_noop(self) -> None:
        CallbackDispatcher([]).on_setup_start(_trainer())

    def test_unspecified_setup_policy_is_best_effort(self) -> None:
        class _AbortingCallback(_RecordingCallback):
            def on_setup_start(self, trainer: BaseTrainer) -> None:
                raise RuntimeError("abort")

        CallbackDispatcher([_AbortingCallback()]).on_setup_start(_trainer())

    def test_explicit_best_effort_setup_error_is_swallowed(self) -> None:
        class _BestEffortCallback(_RecordingCallback):
            failure_policy = "best_effort"

            def on_setup_start(self, trainer: BaseTrainer) -> None:
                raise RuntimeError("abort")

        CallbackDispatcher([_BestEffortCallback()]).on_setup_start(_trainer())

    def test_required_policy_propagates_at_setup_and_training_end(self) -> None:
        class _RequiredCallback(_RecordingCallback):
            failure_policy = "required"

            def on_setup_start(self, trainer: BaseTrainer) -> None:
                raise RuntimeError("setup abort")

        with pytest.raises(RuntimeError, match="setup abort"):
            CallbackDispatcher([_RequiredCallback()]).on_setup_start(_trainer())

        class _RequiredTrainingCallback(_RecordingCallback):
            failure_policy = "required"

            def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
                raise RuntimeError("training abort")

        with pytest.raises(RuntimeError, match="training abort"):
            CallbackDispatcher([_RequiredTrainingCallback()]).on_training_end(
                _trainer(), "ckpt"
            )

    def test_on_training_end_swallows_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _BrokenCallback(_RecordingCallback):
            def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
                raise RuntimeError("broken")

        with caplog.at_level(logging.WARNING):
            CallbackDispatcher([_BrokenCallback()]).on_training_end(_trainer(), "ckpt")

        assert "Callback on_training_end failed" in caplog.text


class TestArtifactsExportFallback:
    def test_new_hook_called_when_implemented(self) -> None:
        cb = _RecordingCallback()
        CallbackDispatcher([cb]).on_artifacts_exported(_trainer(), "/tmp/out")

        assert [c[0] for c in cb.calls] == ["on_artifacts_exported"]
        assert "on_export_end" not in [c[0] for c in cb.calls]

    def test_legacy_callback_falls_back_to_on_export_end(self) -> None:
        cb = _LegacyCallback()
        CallbackDispatcher([cb]).on_artifacts_exported(_trainer(), "/tmp/out")

        assert [c[0] for c in cb.calls] == ["on_export_end"]
        assert cb.calls[0][2] == "/tmp/out"

    def test_on_artifacts_exported_swallows_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _BrokenCallback(_RecordingCallback):
            def on_artifacts_exported(
                self, trainer: BaseTrainer, output_path: str
            ) -> None:
                raise RuntimeError("broken")

        with caplog.at_level(logging.WARNING):
            CallbackDispatcher([_BrokenCallback()]).on_artifacts_exported(
                _trainer(), "/tmp/out"
            )

        assert "Callback on_artifacts_exported failed" in caplog.text

    def test_on_run_complete_swallows_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _BrokenCallback(_RecordingCallback):
            def on_run_complete(
                self, trainer: BaseTrainer, summary: dict[str, Any]
            ) -> None:
                raise RuntimeError("broken")

        with caplog.at_level(logging.WARNING):
            CallbackDispatcher([_BrokenCallback()]).on_run_complete(
                _trainer(), {"status": "succeeded"}
            )

        assert "Callback on_run_complete failed" in caplog.text


class TestErrorEvent:
    def test_on_run_error_returns_first_callback_error(self) -> None:
        class _RaisingCallback(_RecordingCallback):
            def on_run_error(self, trainer: BaseTrainer, error: Exception) -> None:
                raise ValueError("callback blew up")

        error = RuntimeError("training failed")
        result = CallbackDispatcher([_RaisingCallback()]).on_run_error(
            _trainer(), error
        )

        assert isinstance(result, ValueError)
        assert str(result) == "callback blew up"

    def test_on_run_error_returns_none_when_clean(self) -> None:
        cb = _RecordingCallback()
        result = CallbackDispatcher([cb]).on_run_error(_trainer(), RuntimeError("x"))

        assert result is None
        assert cb.calls[0][0] == "on_run_error"
