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

    A callback may expose ``failure_policy == "required"``.  Required
    callbacks propagate failures at every normal lifecycle phase; best-effort
    callbacks log and continue.  Error-handler failures are returned to the
    lifecycle as diagnostics and never replace the original training error.
    """

    def __init__(self, callbacks: Sequence[TrainerCallback]) -> None:
        self._callbacks = list(callbacks)

    def on_setup_start(self, trainer: BaseTrainer) -> None:
        """Fire ``on_setup_start`` under each callback's failure policy."""
        for cb in self._callbacks:
            self._invoke(cb, "on_setup_start", trainer)

    def on_training_end(self, trainer: BaseTrainer, result: Any) -> None:
        """Fire ``on_training_end`` under each callback's failure policy."""
        for cb in self._callbacks:
            self._invoke(cb, "on_training_end", trainer, result)

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
            self._invoke(cb, event, trainer, output_path)

    def on_run_complete(self, trainer: BaseTrainer, summary: dict[str, Any]) -> None:
        """Fire ``on_run_complete`` under each callback's failure policy."""
        for cb in self._callbacks:
            self._invoke(cb, "on_run_complete", trainer, summary)

    def on_run_error(
        self, trainer: BaseTrainer, error: BaseException
    ) -> Exception | None:
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

    @staticmethod
    def _invoke(
        cb: TrainerCallback,
        event: str,
        *args: Any,
        default_policy: str = "best_effort",
    ) -> None:
        try:
            getattr(cb, event)(*args)
        except Exception as exc:
            logger.warning("Callback %s failed: %s", event, exc)
            if getattr(cb, "failure_policy", default_policy) == "required":
                raise
