"""OperationStore persistence tests for explainability records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tributo.explainability.contracts import ExplainabilityOperationRecord
from tributo.exporting.records import InMemoryOperationStore
from tributo.integrations.storage.json_operation_store import JsonFileOperationStore


def _record() -> ExplainabilityOperationRecord:
    return ExplainabilityOperationRecord(
        operation_id="operation-1",
        request_id="request-1",
        bundle_digest="a" * 64,
        result_uri="/results/explanations",
        status="succeeded",
    )


def test_in_memory_store_keeps_explainability_history_separate() -> None:
    store = InMemoryOperationStore()
    record = _record()
    store.record_explainability(record)
    assert store.get_explainability("operation-1") == record
    assert store.get_execution("operation-1") is None


def test_json_store_round_trips_explainability_record(tmp_path) -> None:
    store = JsonFileOperationStore(tmp_path / "operations")
    record = _record()
    store.record_explainability(record)
    restored = JsonFileOperationStore(tmp_path / "operations")
    assert restored.get_explainability("operation-1") == record


def test_operation_store_enforces_idempotent_state_transitions() -> None:
    store = InMemoryOperationStore()
    key = "b" * 64
    running = _record().model_copy(update={"idempotency_key": key, "status": "running"})
    store.record_explainability(running)
    with pytest.raises(ValueError, match="already running"):
        store.record_explainability(running)

    failed = running.model_copy(update={"status": "failed", "retryable": True})
    store.record_explainability(failed)
    with pytest.raises(ValueError, match="invalid explainability state transition"):
        store.record_explainability(failed.model_copy(update={"status": "succeeded"}))

    retry = failed.model_copy(update={"status": "running", "retries": 1})
    store.record_explainability(retry)
    succeeded = retry.model_copy(update={"status": "succeeded"})
    store.record_explainability(succeeded)
    store.record_explainability(succeeded)
    with pytest.raises(ValueError, match="already terminal"):
        store.record_explainability(
            succeeded.model_copy(update={"result_uri": "/other"})
        )


def test_expired_explainability_lease_can_be_reclaimed() -> None:
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    store = InMemoryOperationStore(clock=lambda: now)
    key = "c" * 64
    running = _record().model_copy(
        update={
            "idempotency_key": key,
            "status": "running",
            "lease_expires_at": now - timedelta(seconds=1),
        }
    )
    store.record_explainability(running)
    reclaimed = running.model_copy(
        update={
            "lease_token": "new-token",
            "lease_expires_at": now + timedelta(minutes=5),
            "retries": 1,
        }
    )
    store.record_explainability(reclaimed)
    assert store.get_explainability("operation-1") == reclaimed


def test_json_store_persists_explainability_lease(tmp_path) -> None:
    store_dir = tmp_path / "operations"
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    record = _record().model_copy(
        update={
            "idempotency_key": "d" * 64,
            "status": "running",
            "lease_expires_at": expires,
        }
    )
    JsonFileOperationStore(store_dir).record_explainability(record)
    restored = JsonFileOperationStore(store_dir).get_explainability("operation-1")
    assert restored is not None
    assert restored.lease_expires_at == record.lease_expires_at


def test_explainability_lease_can_be_renewed_only_by_current_token() -> None:
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    store = InMemoryOperationStore(clock=lambda: now)
    running = _record().model_copy(
        update={
            "idempotency_key": "e" * 64,
            "status": "running",
            "lease_token": "token-1",
            "lease_expires_at": now + timedelta(seconds=30),
        }
    )
    store.record_explainability(running)
    renewed = store.renew_explainability(
        "operation-1",
        idempotency_key="e" * 64,
        lease_token="token-1",
        lease_expires_at=now + timedelta(minutes=5),
    )
    assert renewed.lease_expires_at == now + timedelta(minutes=5)
    with pytest.raises(ValueError, match="lease token"):
        store.renew_explainability(
            "operation-1",
            idempotency_key="e" * 64,
            lease_token="stale-token",
            lease_expires_at=now + timedelta(minutes=5),
        )


def test_json_store_replays_lease_renewal_snapshots(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    store_dir = tmp_path / "operations"
    store = JsonFileOperationStore(store_dir)
    running = _record().model_copy(
        update={
            "idempotency_key": "f" * 64,
            "status": "running",
            "lease_token": "token-2",
            "lease_expires_at": now + timedelta(seconds=30),
        }
    )
    store.record_explainability(running)
    store.renew_explainability(
        "operation-1",
        idempotency_key="f" * 64,
        lease_token="token-2",
        lease_expires_at=now + timedelta(minutes=5),
    )

    restored = JsonFileOperationStore(store_dir).list_explainability("operation-1")
    assert len(restored) == 2
    assert restored[-1].lease_expires_at == now + timedelta(minutes=5)
