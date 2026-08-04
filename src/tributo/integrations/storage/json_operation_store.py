"""JSON file-backed operation store — persists ExecutionRecord and PublicationAttempt.

Uses atomic per-file writes (write-to-tmp-then-rename) for crash safety.
Production deployments should prefer a database-backed implementation.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from tributo.exporting.records import (
    ExecutionRecord,
    InMemoryOperationStore,
    PublicationAttempt,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class JsonFileOperationStore:
    """File-backed ``OperationStore`` that persists records as JSON files.

    Each ``ExecutionRecord`` and ``PublicationAttempt`` is written to a
    separate JSON file under the store root.  File writes use atomic
    ``write-to-tmp-then-rename``.

    Args:
        store_dir: Root directory for operation records.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self._store_dir = Path(store_dir)
        self._executions_dir = self._store_dir / "executions"
        self._attempts_dir = self._store_dir / "attempts"
        self._lock = threading.Lock()
        self._mem = InMemoryOperationStore()

    # ── Persistence ──────────────────────────────────────────────────────────

    def record_execution(self, record: ExecutionRecord) -> None:
        """Persist *record* atomically."""
        self._mem.record_execution(record)
        self._executions_dir.mkdir(parents=True, exist_ok=True)
        fpath = self._executions_dir / (
            f"{record.execution_id}--{uuid.uuid4().hex}.json"
        )
        _atomic_write_json(fpath, record.model_dump(mode="json"), self._lock)

    def record_publication_attempt(self, attempt: PublicationAttempt) -> None:
        """Persist *attempt* atomically."""
        self._mem.record_publication_attempt(attempt)
        self._attempts_dir.mkdir(parents=True, exist_ok=True)
        fpath = self._attempts_dir / f"{attempt.attempt_id}.json"
        _atomic_write_json(fpath, attempt.model_dump(mode="json"), self._lock)

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Return the latest attempt for an execution ID."""
        records = self.list_executions(execution_id)
        return records[-1] if records else None

    def list_executions(self, execution_id: str) -> list[ExecutionRecord]:
        """Return all attempts for an execution ID in start order."""
        cached = self._mem.list_executions(execution_id)
        if cached:
            return cached
        if not self._executions_dir.exists():
            return []

        results: list[ExecutionRecord] = []
        candidates = [
            self._executions_dir / f"{execution_id}.json",
            *self._executions_dir.glob(f"{execution_id}--*.json"),
        ]
        for fpath in candidates:
            raw = _read_json(fpath)
            if raw and raw.get("execution_id") == execution_id:
                results.append(ExecutionRecord(**raw))
        results.sort(key=lambda record: record.started_at)
        for record in results:
            self._mem.record_execution(record)
        return results

    def list_attempts(
        self, bundle_digest: str, hook_id: str
    ) -> list[PublicationAttempt]:
        """List all attempts for a (bundle_digest, hook_id) pair."""
        cached = self._mem.list_attempts(bundle_digest, hook_id)
        if cached:
            return cached
        # Scan disk for matching attempts.
        if not self._attempts_dir.exists():
            return []
        results: list[PublicationAttempt] = []
        for fpath in sorted(self._attempts_dir.glob("*.json")):
            raw = _read_json(fpath)
            if (
                raw
                and raw.get("bundle_digest") == bundle_digest
                and raw.get("hook_id") == hook_id
            ):
                results.append(PublicationAttempt(**raw))
        return results

    # ── Maintenance ──────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all persisted records (for testing)."""
        with self._lock:
            self._mem = InMemoryOperationStore()
            for d in (self._executions_dir, self._attempts_dir):
                if d.exists():
                    for f in d.glob("*.json"):
                        f.unlink()


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _atomic_write_json(fpath: Path, data: dict[str, Any], lock: threading.Lock) -> None:
    """Write JSON data atomically via tmp-file + rename."""
    tmp_path = fpath.with_suffix(".tmp")
    payload = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)
    with lock:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(str(tmp_path), str(fpath))


def _read_json(fpath: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning None if it doesn't exist."""
    if not fpath.is_file():
        return None
    raw = json.loads(fpath.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    return raw
