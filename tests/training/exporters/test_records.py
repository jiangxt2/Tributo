"""Tests for append-only execution attempt records."""

from __future__ import annotations

from pathlib import Path

from tributo.exporting.records import ExecutionRecord, InMemoryOperationStore
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
