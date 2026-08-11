"""Closed training outcome contract for training and Bundle publication."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tributo.exporting.models import HookStatus
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class TrainingStatus(StrEnum):
    """Terminal state of the training phase."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@PublicAPI(stability="beta")
class BundleStatus(StrEnum):
    """State of the required Bundle publication phase."""

    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


@PublicAPI(stability="beta")
class TrainingHookStatus(StrEnum):
    """Aggregate state of configured post-publication hooks."""

    NOT_CONFIGURED = "not_configured"
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    FAILED = "failed"


@PublicAPI(stability="beta")
class TrainingResult(BaseModel):
    """Stable result shared by first-party and compatible legacy trainers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_uri: str | None = None
    bundle_uri: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    legacy_artifact_uri: str | None = None
    training_status: TrainingStatus
    bundle_status: BundleStatus
    hook_status: TrainingHookStatus
    execution_id: str | None = None

    @model_validator(mode="after")
    def _validate_state_combination(self) -> TrainingResult:
        if self.training_status in {
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        }:
            if self.bundle_status != BundleStatus.NOT_STARTED:
                raise ValueError(
                    "failed or cancelled training requires bundle_status=not_started"
                )
            if self.hook_status != TrainingHookStatus.NOT_CONFIGURED:
                raise ValueError(
                    "failed or cancelled training requires hook_status=not_configured"
                )

        if self.bundle_status in {BundleStatus.NOT_STARTED, BundleStatus.FAILED}:
            if self.hook_status != TrainingHookStatus.NOT_CONFIGURED:
                raise ValueError(
                    "hooks cannot run before a Bundle is successfully committed"
                )
        elif self.training_status != TrainingStatus.SUCCEEDED:
            raise ValueError("a committed Bundle requires succeeded training")

        if self.bundle_status in {BundleStatus.SUCCEEDED, BundleStatus.PARTIAL}:
            if self.bundle_uri is None or self.execution_id is None:
                raise ValueError(
                    "a committed Bundle requires bundle_uri and execution_id"
                )
        elif self.bundle_uri is not None:
            raise ValueError("bundle_uri is only valid for a committed Bundle")

        return self


def aggregate_hook_status(receipts: Iterable[Any]) -> TrainingHookStatus:
    """Aggregate immutable hook receipts according to the architecture contract."""
    statuses = [HookStatus(receipt.status) for receipt in receipts]
    if not statuses:
        return TrainingHookStatus.NOT_CONFIGURED
    if HookStatus.ACCEPTED in statuses:
        return TrainingHookStatus.PENDING
    if all(status == HookStatus.SKIPPED for status in statuses):
        return TrainingHookStatus.SKIPPED

    failures = {
        HookStatus.RETRYABLE_FAILED,
        HookStatus.TERMINAL_FAILED,
    }
    has_failure = any(status in failures for status in statuses)
    has_non_failure = any(
        status in {HookStatus.SUCCEEDED, HookStatus.SKIPPED} for status in statuses
    )
    if has_failure and has_non_failure:
        return TrainingHookStatus.PARTIAL
    if has_failure:
        return TrainingHookStatus.FAILED
    return TrainingHookStatus.SUCCEEDED


__all__ = [
    "BundleStatus",
    "TrainingHookStatus",
    "TrainingResult",
    "TrainingStatus",
]
