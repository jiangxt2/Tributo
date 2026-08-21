"""Broker-neutral progress reporting for training orchestration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tributo.training.execution_context import (
    CancellationChecker,
    ExecutionContext,
    TrainingCancelledError,
    TrainingEventReporter,
)

logger = logging.getLogger(__name__)


class TrainingPhase(str, Enum):
    """Non-terminal phases emitted by Core at real execution boundaries."""

    LOADING_DATA = "LOADING_DATA"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    DATA_SPLITTING = "DATA_SPLITTING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"


@dataclass
class TrainingProgress:
    """Cancellation-aware, best-effort non-terminal event publisher."""

    job_id: str | None = None
    reporter: TrainingEventReporter | None = None
    checker: CancellationChecker | None = None
    reporter_job_id: str | None = None
    cancellation_job_id: str | None = None

    def __post_init__(self) -> None:
        """Map the legacy shared ``job_id`` onto both independent controls."""
        if self.reporter_job_id is None:
            self.reporter_job_id = self.job_id
        if self.cancellation_job_id is None:
            self.cancellation_job_id = self.job_id

    @classmethod
    def from_environment(cls) -> TrainingProgress:
        """Rebuild configured worker/driver controls from the opaque context."""
        context = ExecutionContext.from_environment()
        reporter_job_id = (
            context.event_reporter.job_id
            if context.event_reporter is not None
            else None
        )
        cancellation_job_id = (
            context.cancellation.job_id if context.cancellation is not None else None
        )
        checker: CancellationChecker | None = None
        reporter: TrainingEventReporter | None = None
        try:
            checker = context.build_cancellation_checker()
        except Exception:
            logger.warning("Failed to create cancellation checker", exc_info=True)
        try:
            reporter = context.build_event_reporter()
        except Exception:
            logger.warning("Failed to create training event reporter", exc_info=True)
        return cls(
            job_id=reporter_job_id or cancellation_job_id,
            reporter=reporter,
            checker=checker,
            reporter_job_id=reporter_job_id,
            cancellation_job_id=cancellation_job_id,
        )

    def check_cancelled(self) -> None:
        """Fail open on checker transport faults, but raise on a confirmed cancel."""
        if self.cancellation_job_id is None or self.checker is None:
            return
        try:
            cancelled = self.checker.is_cancelled(self.cancellation_job_id)
        except Exception:
            logger.warning(
                "Cancellation check failed for job %s",
                self.cancellation_job_id,
                exc_info=True,
            )
            return
        if cancelled:
            raise TrainingCancelledError(
                f"Training job {self.cancellation_job_id!r} was cancelled"
            )

    def report_phase(self, phase: TrainingPhase) -> None:
        """Check cancellation, then publish a non-terminal phase best-effort."""
        self.check_cancelled()
        if self.reporter_job_id is None or self.reporter is None:
            return
        try:
            self.reporter.report_phase(self.reporter_job_id, phase.value)
        except Exception:
            logger.warning(
                "Failed to report training phase %s for job %s",
                phase.value,
                self.reporter_job_id,
                exc_info=True,
            )

    def report_metrics(
        self, metrics: Mapping[str, Any], progress: float | None = None
    ) -> None:
        """Check cancellation, then publish live metrics best-effort."""
        self.check_cancelled()
        if self.reporter_job_id is None or self.reporter is None:
            return
        try:
            self.reporter.report_metrics(self.reporter_job_id, metrics, progress)
        except Exception:
            logger.warning(
                "Failed to report training metrics for job %s",
                self.reporter_job_id,
                exc_info=True,
            )
