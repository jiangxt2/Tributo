"""Tests for inline Hook Dispatcher delivery policy."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from tributo.exceptions import PostPublishCallbackError
from tributo.exporting.dispatch import InlineHookDispatcher, PreparedHook
from tributo.exporting.events import OperationEvent
from tributo.exporting.hooks import HookOutcome, PublicationRunner
from tributo.exporting.models import BundleResult, HookStatus
from tributo.exporting.records import InMemoryOperationStore


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Accessor:
    def read_manifest(self) -> Any:
        return None

    @contextmanager
    def materialize_manifest(self) -> Any:
        yield Path("/committed/manifest.json")

    @contextmanager
    def materialize_bundle(self) -> Any:
        yield Path("/committed")


class _Hook:
    api_version: ClassVar[int] = 1
    hook_id: ClassVar[str] = "test-hook-v1"
    options_model: ClassVar[type[BaseModel]] = _Options

    def __init__(self, status: HookStatus = HookStatus.SUCCEEDED) -> None:
        self.status = status
        self.calls = 0

    def deliver(self, event: Any, artifacts: Any, options: Any) -> HookOutcome:
        self.calls += 1
        return HookOutcome(
            status=self.status,
            error_code="failed" if "failed" in self.status.value else None,
        )

    def idempotency_key(self, event: Any, options: Any) -> str:
        return "stable-key"


def _event() -> OperationEvent:
    return OperationEvent.bundle_published(
        occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        bundle_id="bundle-1",
        canonical_uri="file:///bundle-1",
        manifest_sha256="a" * 64,
    )


def _result() -> BundleResult:
    return BundleResult(
        bundle_id="bundle-1",
        canonical_uri="file:///bundle-1",
        manifest_uri="file:///bundle-1/manifest.json",
        manifest_sha256="a" * 64,
        status="succeeded",
    )


def _dispatch(
    dispatcher: InlineHookDispatcher, hook: _Hook, *, required: bool = False
) -> BundleResult:
    return dispatcher.dispatch_prepared(
        event=_event(),
        bundle_result=_result(),
        bundle_digest="b" * 64,
        prepared_hooks=(
            PreparedHook(adapter=hook, options=_Options(), required=required),
        ),
        artifacts=_Accessor(),
    )


def test_success_receipt_and_duplicate_skip_share_delivery_fact() -> None:
    dispatcher = InlineHookDispatcher(InMemoryOperationStore())
    hook = _Hook()
    first = _dispatch(dispatcher, hook)
    second = _dispatch(dispatcher, hook)

    assert first.hook_receipts[0].status is HookStatus.SUCCEEDED
    assert second.hook_receipts[0].status is HookStatus.SKIPPED
    assert second.hook_receipts[0].error_code == "already_succeeded"
    assert hook.calls == 1


def test_publication_runner_reuses_its_default_delivery_store() -> None:
    hook = _Hook()
    runner = PublicationRunner([(hook, _Options(), False)])

    first = runner.run(
        event=_event(),
        artifacts=_Accessor(),
        bundle_result=_result(),
        bundle_digest="b" * 64,
    )
    second = runner.run(
        event=_event(),
        artifacts=_Accessor(),
        bundle_result=_result(),
        bundle_digest="b" * 64,
    )

    assert first.hook_receipts[0].status is HookStatus.SUCCEEDED
    assert second.hook_receipts[0].status is HookStatus.SKIPPED
    assert hook.calls == 1


def test_optional_failure_is_returned_without_invalidating_bundle() -> None:
    result = _dispatch(InlineHookDispatcher(), _Hook(HookStatus.RETRYABLE_FAILED))
    assert result.status == "succeeded"
    assert result.hook_receipts[0].status is HookStatus.RETRYABLE_FAILED


def test_required_failure_carries_committed_result_and_receipts() -> None:
    with pytest.raises(PostPublishCallbackError) as exc_info:
        _dispatch(
            InlineHookDispatcher(),
            _Hook(HookStatus.TERMINAL_FAILED),
            required=True,
        )
    assert exc_info.value.bundle_result.status == "succeeded"
    assert exc_info.value.receipts[0].status is HookStatus.TERMINAL_FAILED


def test_required_delivery_does_not_treat_in_progress_as_success() -> None:
    store = InMemoryOperationStore()
    store.claim_delivery(
        event_id=_event().event_id,
        bundle_digest="b" * 64,
        hook_id=_Hook.hook_id,
        idempotency_key="stable-key",
    )
    with pytest.raises(PostPublishCallbackError) as exc_info:
        _dispatch(InlineHookDispatcher(store), _Hook(), required=True)
    assert exc_info.value.receipts[0].status is HookStatus.RETRYABLE_FAILED
    assert exc_info.value.receipts[0].error_code == "in_progress"


def test_unexpected_hook_exception_is_retryable() -> None:
    hook = _Hook()

    def raise_error(*args: Any) -> HookOutcome:
        raise RuntimeError("secret-token-value")

    hook.deliver = raise_error
    result = _dispatch(InlineHookDispatcher(), hook)
    assert result.hook_receipts[0].status is HookStatus.RETRYABLE_FAILED
    assert result.hook_receipts[0].error_code == "hook_exception"
    assert "secret-token-value" not in (result.hook_receipts[0].error_summary or "")


def test_idempotency_key_exception_is_terminal_for_optional_hook() -> None:
    hook = _Hook()

    def raise_error(*args: Any) -> str:
        raise RuntimeError("secret-key-value")

    hook.idempotency_key = raise_error
    result = _dispatch(InlineHookDispatcher(), hook)
    assert result.status == "succeeded"
    assert result.hook_receipts[0].status is HookStatus.TERMINAL_FAILED
    assert result.hook_receipts[0].error_code == "idempotency_key_error"
    assert "secret-key-value" not in (result.hook_receipts[0].error_summary or "")


def test_operation_store_claim_failure_is_an_optional_receipt() -> None:
    class _ClaimFailingStore(InMemoryOperationStore):
        def claim_delivery(self, **kwargs: Any) -> Any:
            raise RuntimeError("secret-store-value")

    hook = _Hook()
    result = _dispatch(InlineHookDispatcher(_ClaimFailingStore()), hook)
    receipt = result.hook_receipts[0]
    assert receipt.status is HookStatus.RETRYABLE_FAILED
    assert receipt.error_code == "operation_store_claim_failed"
    assert "secret-store-value" not in (receipt.error_summary or "")
    assert hook.calls == 0


def test_operation_store_completion_failure_preserves_external_outcome() -> None:
    class _CompleteFailingStore(InMemoryOperationStore):
        def complete_delivery(self, delivery_id: str, **kwargs: Any) -> Any:
            raise RuntimeError("secret-store-value")

    result = _dispatch(InlineHookDispatcher(_CompleteFailingStore()), _Hook())
    receipt = result.hook_receipts[0]
    assert receipt.status is HookStatus.RETRYABLE_FAILED
    assert receipt.error_code == "operation_store_complete_failed"
    assert "secret-store-value" not in (receipt.error_summary or "")


def test_operation_store_completion_requires_completed_at() -> None:
    class _InvalidCompleteStore(InMemoryOperationStore):
        def complete_delivery(self, delivery_id: str, **kwargs: Any) -> Any:
            completed = super().complete_delivery(delivery_id, **kwargs)
            return completed.model_copy(update={"completed_at": None})

    result = _dispatch(InlineHookDispatcher(_InvalidCompleteStore()), _Hook())
    receipt = result.hook_receipts[0]
    assert receipt.status is HookStatus.RETRYABLE_FAILED
    assert receipt.error_code == "operation_store_complete_failed"
