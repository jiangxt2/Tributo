"""Inline dispatcher for explicitly configured publication hooks."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from tributo.exceptions import JobConfigurationError, PostPublishCallbackError
from tributo.exporting.events import OperationEvent
from tributo.exporting.hooks import (
    ArtifactAccessor,
    HookOutcome,
    PublicationHook,
)
from tributo.exporting.models import (
    BundleResult,
    HookBinding,
    HookReceipt,
    HookStatus,
)
from tributo.exporting.records import InMemoryOperationStore, OperationStore
from tributo.plugin import resolve_hook_plugin
from tributo.util.annotations import DeveloperAPI, PublicAPI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedHook:
    """Validated adapter instance and its delivery policy."""

    adapter: PublicationHook
    options: BaseModel
    required: bool


@runtime_checkable
@DeveloperAPI
class HookDispatcher(Protocol):
    """Execution-policy boundary for publication hook adapters."""

    def preflight(
        self, bindings: tuple[HookBinding, ...]
    ) -> tuple[PreparedHook, ...]: ...

    def dispatch(
        self,
        *,
        event: OperationEvent,
        bundle_result: BundleResult,
        bundle_digest: str,
        prepared_hooks: tuple[PreparedHook, ...],
        artifacts: ArtifactAccessor,
    ) -> BundleResult: ...


@PublicAPI(stability="beta")
class InlineHookDispatcher:
    """Claim, execute, and complete publication hooks in the caller process."""

    def __init__(self, operation_store: OperationStore | None = None) -> None:
        self._store = (
            operation_store if operation_store is not None else InMemoryOperationStore()
        )

    def preflight(self, bindings: tuple[HookBinding, ...]) -> tuple[PreparedHook, ...]:
        """Resolve only requested plugins and validate their options."""
        prepared: list[PreparedHook] = []
        for binding in bindings:
            hook_cls = resolve_hook_plugin(binding.hook_id)
            try:
                options = hook_cls.options_model.model_validate(binding.options)
            except ValidationError as exc:
                details = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors(include_url=False, include_input=False)
                )
                raise JobConfigurationError(
                    f"Invalid options for hook {binding.hook_id!r}: {details}"
                ) from exc
            try:
                adapter = hook_cls()
            except Exception as exc:
                raise JobConfigurationError(
                    f"Failed to initialize hook {binding.hook_id!r} "
                    f"({type(exc).__name__})"
                ) from exc
            prepared.append(
                PreparedHook(
                    adapter=adapter,
                    options=options,
                    required=binding.required,
                )
            )
        return tuple(prepared)

    def dispatch(
        self,
        *,
        event: OperationEvent,
        bundle_result: BundleResult,
        bundle_digest: str,
        prepared_hooks: tuple[PreparedHook, ...],
        artifacts: ArtifactAccessor,
    ) -> BundleResult:
        """Synchronously deliver an event and return receipts on the result."""
        return self.dispatch_prepared(
            event=event,
            bundle_result=bundle_result,
            bundle_digest=bundle_digest,
            prepared_hooks=prepared_hooks,
            artifacts=artifacts,
        )

    def dispatch_prepared(
        self,
        *,
        event: OperationEvent,
        bundle_result: BundleResult,
        bundle_digest: str,
        prepared_hooks: tuple[PreparedHook, ...],
        artifacts: ArtifactAccessor,
    ) -> BundleResult:
        """Delivery core, separated so tests can supply an accessor."""
        receipts: list[HookReceipt] = []
        for prepared in prepared_hooks:
            adapter = prepared.adapter
            key_error: Exception | None = None
            try:
                key = adapter.idempotency_key(event, prepared.options)
            except Exception as exc:
                logger.error(
                    "Hook %s failed to build an idempotency key (%s)",
                    adapter.hook_id,
                    type(exc).__name__,
                )
                key_error = exc
                fallback = f"{event.event_id}/{adapter.hook_id}/key-error"
                key = hashlib.sha256(fallback.encode("utf-8")).hexdigest()
            try:
                claim = self._store.claim_delivery(
                    event_id=event.event_id,
                    bundle_digest=bundle_digest,
                    hook_id=adapter.hook_id,
                    idempotency_key=key,
                )
            except Exception as exc:
                logger.error(
                    "OperationStore claim failed for Hook %s (%s)",
                    adapter.hook_id,
                    type(exc).__name__,
                )
                now = datetime.now(timezone.utc)
                receipt = HookReceipt(
                    event_id=event.event_id,
                    hook_id=adapter.hook_id,
                    delivery_id=f"delivery-unrecorded-{uuid.uuid4().hex}",
                    idempotency_key=key,
                    status=HookStatus.RETRYABLE_FAILED,
                    started_at=now,
                    completed_at=now,
                    error_code="operation_store_claim_failed",
                    error_summary=(
                        f"OperationStore claim failed ({type(exc).__name__})"
                    ),
                )
                receipts.append(receipt)
                current_result = bundle_result.model_copy(
                    update={"hook_receipts": tuple(receipts)}
                )
                if prepared.required:
                    raise PostPublishCallbackError(
                        f"Required hook {adapter.hook_id!r} could not be claimed",
                        bundle_result=current_result,
                        receipts=tuple(receipts),
                    ) from exc
                continue

            if claim.disposition != "acquired":
                receipt = self._receipt_from_existing(
                    event,
                    claim.disposition,
                    claim.record,
                    required=prepared.required,
                )
            else:
                if key_error is not None:
                    outcome = HookOutcome(
                        status=HookStatus.TERMINAL_FAILED,
                        error_code="idempotency_key_error",
                        error_summary=(
                            f"Hook idempotency key failed ({type(key_error).__name__})"
                        ),
                    )
                else:
                    try:
                        outcome = adapter.deliver(event, artifacts, prepared.options)
                        if outcome.status is HookStatus.ACCEPTED:
                            outcome = HookOutcome(
                                status=HookStatus.TERMINAL_FAILED,
                                error_code="invalid_inline_outcome",
                                error_summary=(
                                    "Inline adapters must return a completed status"
                                ),
                            )
                    except Exception as exc:
                        logger.error(
                            "Hook %s raised unexpectedly (%s)",
                            adapter.hook_id,
                            type(exc).__name__,
                        )
                        outcome = HookOutcome(
                            status=HookStatus.RETRYABLE_FAILED,
                            error_code="hook_exception",
                            error_summary=(
                                f"Hook raised an unexpected {type(exc).__name__}"
                            ),
                        )
                try:
                    completed = self._store.complete_delivery(
                        claim.record.delivery_id,
                        status=outcome.status,
                        error_code=outcome.error_code,
                        error_summary=outcome.error_summary,
                        external_references=outcome.external_references,
                    )
                    if completed.completed_at is None:
                        raise ValueError(
                            "OperationStore returned a completed delivery without "
                            "completed_at"
                        )
                except Exception as exc:
                    logger.error(
                        "OperationStore completion failed for Hook %s (%s)",
                        adapter.hook_id,
                        type(exc).__name__,
                    )
                    receipt = HookReceipt(
                        event_id=event.event_id,
                        hook_id=adapter.hook_id,
                        delivery_id=claim.record.delivery_id,
                        idempotency_key=key,
                        status=(
                            outcome.status
                            if outcome.status
                            in (
                                HookStatus.RETRYABLE_FAILED,
                                HookStatus.TERMINAL_FAILED,
                            )
                            else HookStatus.RETRYABLE_FAILED
                        ),
                        started_at=claim.record.started_at,
                        completed_at=datetime.now(timezone.utc),
                        error_code="operation_store_complete_failed",
                        error_summary=(
                            f"OperationStore completion failed ({type(exc).__name__})"
                        ),
                        external_references=outcome.external_references,
                    )
                else:
                    receipt = HookReceipt(
                        event_id=event.event_id,
                        hook_id=adapter.hook_id,
                        delivery_id=completed.delivery_id,
                        idempotency_key=key,
                        status=completed.status,
                        started_at=completed.started_at,
                        completed_at=completed.completed_at,
                        error_code=completed.error_code,
                        error_summary=completed.error_summary,
                        external_references=completed.external_references,
                    )

            receipts.append(receipt)
            current_result = bundle_result.model_copy(
                update={"hook_receipts": tuple(receipts)}
            )
            if prepared.required and receipt.status in (
                HookStatus.RETRYABLE_FAILED,
                HookStatus.TERMINAL_FAILED,
            ):
                raise PostPublishCallbackError(
                    f"Required hook {adapter.hook_id!r} failed: "
                    f"{receipt.error_summary or receipt.error_code}",
                    bundle_result=current_result,
                    receipts=tuple(receipts),
                )

        return bundle_result.model_copy(update={"hook_receipts": tuple(receipts)})

    @staticmethod
    def _receipt_from_existing(
        event: OperationEvent,
        disposition: str,
        record: object,
        *,
        required: bool,
    ) -> HookReceipt:
        from tributo.exporting.records import DeliveryRecord

        assert isinstance(record, DeliveryRecord)
        completed_at = record.completed_at or datetime.now(timezone.utc)
        if disposition == "terminal_failed":
            status = HookStatus.TERMINAL_FAILED
            error_code = record.error_code
            error_summary = record.error_summary
        elif disposition == "in_progress" and required:
            status = HookStatus.RETRYABLE_FAILED
            error_code = "in_progress"
            error_summary = "Required delivery is still in progress"
        else:
            status = HookStatus.SKIPPED
            error_code = disposition
            error_summary = {
                "already_succeeded": "Delivery already completed successfully",
                "in_progress": "Delivery is already in progress",
            }.get(disposition)
        return HookReceipt(
            event_id=event.event_id,
            hook_id=record.hook_id,
            delivery_id=record.delivery_id,
            idempotency_key=record.idempotency_key,
            status=status,
            started_at=record.started_at,
            completed_at=completed_at,
            error_code=error_code,
            error_summary=error_summary,
            external_references=record.external_references,
        )
