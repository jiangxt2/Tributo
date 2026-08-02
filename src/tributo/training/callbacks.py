"""Callback event dispatch for trainers.

``CallbackDispatcher`` centralises the event dispatch that used to be
inlined in ``BaseTrainer.run``: uniform error handling per event type and
backward-compatible fallback between the newer ``on_artifacts_exported``
and the legacy ``on_export_end`` hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from tributo.training.base import BaseTrainer, TrainerCallback

logger = logging.getLogger(__name__)


class CallbackDispatcher:
    """Dispatch trainer lifecycle events to registered callbacks.

    Error-handling policy:
    - ``on_setup_start`` propagates exceptions so a callback can abort
      training early (backward-compatible with ``BaseTrainer.run``).
    - ``on_run_error`` swallows per-callback errors, but collects and
      returns the first callback error so the lifecycle can re-raise it
      chained to the original training error.
    - Every other event swallows per-callback errors and logs a warning,
      so a failing callback does not block the training pipeline.
    """

    def __init__(self, callbacks: Sequence[TrainerCallback]) -> None:
        self._callbacks = list(callbacks)

    def on_setup_start(self, trainer: BaseTrainer) -> None:
        """Fire ``on_setup_start``; exceptions propagate to abort early."""
        for cb in self._callbacks:
            cb.on_setup_start(trainer)

    def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
        """Fire ``on_training_end``; per-callback errors are swallowed."""
        for cb in self._callbacks:
            try:
                cb.on_training_end(trainer, result)
            except Exception as e:
                logger.warning("Callback on_training_end failed: %s", e)

    def on_artifacts_exported(self, trainer: BaseTrainer, output_path: str) -> None:
        """Fire artifact-export events with backward-compatible fallback.

        Callbacks implementing the newer ``on_artifacts_exported`` hook are
        called directly; others fall back to the legacy ``on_export_end``.
        This avoids double-firing when a callback delegates
        ``on_artifacts_exported`` → ``on_export_end``.
        """
        for cb in self._callbacks:
            has_new_hook = "on_artifacts_exported" in type(cb).__dict__
            event = "on_artifacts_exported" if has_new_hook else "on_export_end"
            try:
                if has_new_hook:
                    cb.on_artifacts_exported(trainer, output_path)
                else:
                    cb.on_export_end(trainer, output_path)
            except Exception as e:
                logger.warning("Callback %s failed: %s", event, e)

    def on_run_complete(self, trainer: BaseTrainer, summary: dict[str, Any]) -> None:
        """Fire ``on_run_complete``; per-callback errors are swallowed."""
        for cb in self._callbacks:
            try:
                cb.on_run_complete(trainer, summary)
            except Exception as e:
                logger.warning("Callback on_run_complete failed: %s", e)

    def on_run_error(self, trainer: BaseTrainer, error: Exception) -> Exception | None:
        """Fire ``on_run_error``; return the first callback error, if any."""
        callback_error: Exception | None = None
        for cb in self._callbacks:
            try:
                cb.on_run_error(trainer, error)
            except Exception as cb_err:
                logger.warning("Callback on_run_error failed: %s", cb_err)
                if callback_error is None:
                    callback_error = cb_err
        return callback_error
