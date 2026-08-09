"""Tests for append-only execution attempt records."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tributo.exporting.models import HookStatus
from tributo.exporting.records import (
    DeliveryClaim,
    ExecutionRecord,
    InMemoryOperationStore,
    OperationStore,
)
from tributo.integrations.storage.json_operation_store import JsonFileOperationStore


def _record(attempt_id: str) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="exec-1",
        bundle_id="bundle-1",
        attempt_id=attempt_id,
        status="succeeded",
    )


def test_in_memory_store_retains_all_execution_attempts() -> None:
    store = InMemoryOperationStore()
    store.record_execution(_record("attempt-1"))
    store.record_execution(_record("attempt-2"))

    assert [r.attempt_id for r in store.list_executions("exec-1")] == [
        "attempt-1",
        "attempt-2",
    ]
    latest = store.get_execution("exec-1")
    assert latest is not None
    assert latest.attempt_id == "attempt-2"


def test_json_store_retains_all_execution_attempts(tmp_path: Path) -> None:
    store = JsonFileOperationStore(tmp_path)
    store.record_execution(_record("attempt-1"))
    store.record_execution(_record("attempt-2"))

    fresh_store = JsonFileOperationStore(tmp_path)
    assert [r.attempt_id for r in fresh_store.list_executions("exec-1")] == [
        "attempt-1",
        "attempt-2",
    ]


def _claim(store: OperationStore) -> DeliveryClaim:
    return store.claim_delivery(
        event_id="event-1",
        bundle_digest="b" * 64,
        hook_id="hook-v1",
        idempotency_key="key-1",
        lease_seconds=30,
    )


def test_claim_is_atomic_within_one_process() -> None:
    store = InMemoryOperationStore()
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: _claim(store), range(8)))
    assert sum(claim.disposition == "acquired" for claim in claims) == 1
    assert sum(claim.disposition == "in_progress" for claim in claims) == 7


def test_same_weak_key_is_scoped_to_one_event_and_bundle() -> None:
    store = InMemoryOperationStore()
    first = _claim(store)
    store.complete_delivery(first.record.delivery_id, status=HookStatus.SUCCEEDED)

    second = store.claim_delivery(
        event_id="event-2",
        bundle_digest="c" * 64,
        hook_id="hook-v1",
        idempotency_key="key-1",
        lease_seconds=30,
    )

    assert second.disposition == "acquired"
    assert second.record.event_id == "event-2"
    assert second.record.bundle_digest == "c" * 64


def test_retryable_failure_and_expired_lease_can_be_reclaimed() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    current = [now]
    store = InMemoryOperationStore(clock=lambda: current[0])
    first = _claim(store)
    store.complete_delivery(
        first.record.delivery_id, status=HookStatus.RETRYABLE_FAILED
    )
    assert _claim(store).disposition == "acquired"

    current[0] += timedelta(seconds=31)
    assert _claim(store).disposition == "acquired"


def test_late_success_from_expired_lease_wins_future_claims() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    current = [now]
    store = InMemoryOperationStore(clock=lambda: current[0])
    first = _claim(store)
    current[0] += timedelta(seconds=31)
    second = _claim(store)
    assert second.disposition == "acquired"

    store.complete_delivery(first.record.delivery_id, status=HookStatus.SUCCEEDED)
    replay = _claim(store)
    assert replay.disposition == "already_succeeded"
    assert replay.record.delivery_id == first.record.delivery_id


def test_succeeded_and_terminal_deliveries_are_not_reexecuted() -> None:
    succeeded_store = InMemoryOperationStore()
    succeeded = _claim(succeeded_store)
    succeeded_store.complete_delivery(
        succeeded.record.delivery_id, status=HookStatus.SUCCEEDED
    )
    assert _claim(succeeded_store).disposition == "already_succeeded"

    terminal_store = InMemoryOperationStore()
    terminal = _claim(terminal_store)
    terminal_store.complete_delivery(
        terminal.record.delivery_id, status=HookStatus.TERMINAL_FAILED
    )
    assert _claim(terminal_store).disposition == "terminal_failed"


def test_json_store_restores_delivery_claim_state(tmp_path: Path) -> None:
    store = JsonFileOperationStore(tmp_path)
    claim = _claim(store)
    store.complete_delivery(claim.record.delivery_id, status=HookStatus.SUCCEEDED)

    fresh_store = JsonFileOperationStore(tmp_path)
    replay = _claim(fresh_store)
    assert replay.disposition == "already_succeeded"
    assert replay.record.delivery_id == claim.record.delivery_id


def test_json_claim_persistence_failure_rolls_back_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = JsonFileOperationStore(tmp_path)
    original = store._persist_delivery

    def fail_persist(record: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_persist_delivery", fail_persist)
    with pytest.raises(OSError, match="disk full"):
        _claim(store)

    monkeypatch.setattr(store, "_persist_delivery", original)
    assert _claim(store).disposition == "acquired"


def test_json_completion_persistence_failure_restores_accepted_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = JsonFileOperationStore(tmp_path)
    claim = _claim(store)

    def fail_persist(record: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_persist_delivery", fail_persist)
    with pytest.raises(OSError, match="disk full"):
        store.complete_delivery(
            claim.record.delivery_id,
            status=HookStatus.SUCCEEDED,
        )

    current = store.list_deliveries("b" * 64, "hook-v1")
    assert current == [claim.record]
    fresh = JsonFileOperationStore(tmp_path)
    assert fresh.list_deliveries("b" * 64, "hook-v1") == [claim.record]
