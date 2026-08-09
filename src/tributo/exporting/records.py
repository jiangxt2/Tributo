"""Operation store — execution and publication record keeping.

Separates mutable operation state (executions and Hook deliveries) from the
immutable ``BundleManifest``.  The in-memory and JSON implementations provide
single-process claim/complete semantics; distributed coordination remains a
separate storage implementation concern.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.manifest import ManifestExecutionNode
from tributo.exporting.models import HookStatus
from tributo.util.annotations import DeveloperAPI, PublicAPI

# ── Data models ────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ExecutionRecord(BaseModel):
    """Immutable snapshot of a single export execution.

    Written once per execution attempt and never modified.  Separated from
    ``BundleManifest`` so the manifest stays purely about model content.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    bundle_id: str
    run_id: str | None = None
    request_id: str | None = None
    attempt_id: str | None = None
    bundle_digest: str | None = None
    status: str = Field(pattern=r"^(pending|running|succeeded|partial|failed)$")
    source_kind: str = ""
    source_fingerprint: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = 0
    nodes: tuple[ManifestExecutionNode, ...] = ()
    roles: dict[str, str] = Field(default_factory=dict)
    tributo_version: str = "0.0.0"


@PublicAPI(stability="beta")
class PublicationAttempt(BaseModel):
    """A single attempt to run a post-publish hook.  Append-only — each
    retry creates a new ``PublicationAttempt`` with a fresh ``attempt_id``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: str
    hook_id: str
    bundle_digest: str
    idempotency_key: str
    status: str = Field(pattern=r"^(success|failed|skipped)$")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    retryable: bool = False


@PublicAPI(stability="beta")
class DeliveryRecord(BaseModel):
    """State of one locally claimed hook delivery attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delivery_id: str
    event_id: str
    bundle_digest: str
    hook_id: str
    idempotency_key: str
    status: HookStatus
    started_at: datetime
    lease_expires_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None
    error_summary: str | None = Field(default=None, max_length=4096)
    external_references: dict[str, str] = Field(default_factory=dict)


@DeveloperAPI
class DeliveryClaim(BaseModel):
    """Result of atomically claiming a hook idempotency key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: Literal[
        "acquired", "already_succeeded", "in_progress", "terminal_failed"
    ]
    record: DeliveryRecord


# ── Protocol ────────────────────────────────────────────────────────────────────


@runtime_checkable
@DeveloperAPI
class OperationStore(Protocol):
    """Append-only store for execution and publication records.

    Separated from ``BundleRepository`` so the two ports can be migrated
    independently (e.g. BundleRepository on S3, OperationStore in a database).
    """

    def record_execution(self, record: ExecutionRecord) -> None: ...

    def record_publication_attempt(self, attempt: PublicationAttempt) -> None: ...

    def get_execution(self, execution_id: str) -> ExecutionRecord | None: ...

    def list_executions(self, execution_id: str) -> list[ExecutionRecord]: ...

    def list_attempts(
        self, bundle_digest: str, hook_id: str
    ) -> list[PublicationAttempt]: ...

    def claim_delivery(
        self,
        *,
        event_id: str,
        bundle_digest: str,
        hook_id: str,
        idempotency_key: str,
        lease_seconds: int = 300,
    ) -> DeliveryClaim: ...

    def complete_delivery(
        self,
        delivery_id: str,
        *,
        status: HookStatus,
        error_code: str | None = None,
        error_summary: str | None = None,
        external_references: dict[str, str] | None = None,
    ) -> DeliveryRecord: ...

    def list_deliveries(
        self, bundle_digest: str, hook_id: str
    ) -> list[DeliveryRecord]: ...


@DeveloperAPI
class InMemoryOperationStore:
    """In-memory ``OperationStore`` for testing and single-process use."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._executions: dict[str, list[ExecutionRecord]] = {}
        self._attempts: dict[str, list[PublicationAttempt]] = {}
        self._deliveries: dict[str, list[DeliveryRecord]] = {}
        self._delivery_ids: dict[str, str] = {}
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()

    def record_execution(self, record: ExecutionRecord) -> None:
        with self._lock:
            self._executions.setdefault(record.execution_id, []).append(record)

    def record_publication_attempt(self, attempt: PublicationAttempt) -> None:
        key = f"{attempt.bundle_digest}/{attempt.hook_id}"
        with self._lock:
            self._attempts.setdefault(key, []).append(attempt)

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        with self._lock:
            records = self._executions.get(execution_id, [])
            return records[-1] if records else None

    def list_executions(self, execution_id: str) -> list[ExecutionRecord]:
        with self._lock:
            return list(self._executions.get(execution_id, []))

    def list_attempts(
        self, bundle_digest: str, hook_id: str
    ) -> list[PublicationAttempt]:
        with self._lock:
            return list(self._attempts.get(f"{bundle_digest}/{hook_id}", []))

    def claim_delivery(
        self,
        *,
        event_id: str,
        bundle_digest: str,
        hook_id: str,
        idempotency_key: str,
        lease_seconds: int = 300,
    ) -> DeliveryClaim:
        """Atomically claim one local delivery lease."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._lock:
            matches = self._matching_deliveries(
                event_id=event_id,
                bundle_digest=bundle_digest,
                hook_id=hook_id,
                idempotency_key=idempotency_key,
            )
            now = self._clock()
            succeeded = next(
                (
                    record
                    for record in reversed(matches)
                    if record.status in (HookStatus.SUCCEEDED, HookStatus.SKIPPED)
                ),
                None,
            )
            if succeeded is not None:
                return DeliveryClaim(disposition="already_succeeded", record=succeeded)
            existing = matches[-1] if matches else None
            if existing is not None:
                if existing.status is HookStatus.TERMINAL_FAILED:
                    return DeliveryClaim(disposition="terminal_failed", record=existing)
                if (
                    existing.status is HookStatus.ACCEPTED
                    and existing.lease_expires_at > now
                ):
                    return DeliveryClaim(disposition="in_progress", record=existing)

            record = DeliveryRecord(
                delivery_id=f"delivery-{uuid.uuid4().hex}",
                event_id=event_id,
                bundle_digest=bundle_digest,
                hook_id=hook_id,
                idempotency_key=idempotency_key,
                status=HookStatus.ACCEPTED,
                started_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            key = f"{bundle_digest}/{hook_id}"
            self._deliveries.setdefault(key, []).append(record)
            self._delivery_ids[record.delivery_id] = key
            return DeliveryClaim(disposition="acquired", record=record)

    def complete_delivery(
        self,
        delivery_id: str,
        *,
        status: HookStatus,
        error_code: str | None = None,
        error_summary: str | None = None,
        external_references: dict[str, str] | None = None,
    ) -> DeliveryRecord:
        """Complete a claimed delivery exactly once within this process."""
        if status is HookStatus.ACCEPTED:
            raise ValueError("accepted is not a completion status")
        with self._lock:
            key = self._delivery_ids.get(delivery_id)
            if key is None:
                raise KeyError(f"Unknown delivery_id {delivery_id!r}")
            records = self._deliveries[key]
            index = next(
                i
                for i, record in enumerate(records)
                if record.delivery_id == delivery_id
            )
            current = records[index]
            if current.completed_at is not None:
                raise ValueError(f"Delivery {delivery_id!r} is already complete")
            completed = current.model_copy(
                update={
                    "status": status,
                    "completed_at": self._clock(),
                    "error_code": error_code,
                    "error_summary": error_summary,
                    "external_references": external_references or {},
                }
            )
            records[index] = completed
            return completed

    def list_deliveries(self, bundle_digest: str, hook_id: str) -> list[DeliveryRecord]:
        """Return all local delivery attempts in claim order."""
        with self._lock:
            return list(self._deliveries.get(f"{bundle_digest}/{hook_id}", []))

    def _matching_deliveries(
        self,
        *,
        event_id: str,
        bundle_digest: str,
        hook_id: str,
        idempotency_key: str,
    ) -> list[DeliveryRecord]:
        records = self._deliveries.get(f"{bundle_digest}/{hook_id}", ())
        matches = [
            record
            for record in records
            if record.event_id == event_id and record.idempotency_key == idempotency_key
        ]
        matches.sort(key=lambda record: record.started_at)
        return matches

    def get_delivery(self, delivery_id: str) -> DeliveryRecord:
        """Return one delivery record for transactional storage adapters."""
        with self._lock:
            key = self._delivery_ids.get(delivery_id)
            if key is None:
                raise KeyError(f"Unknown delivery_id {delivery_id!r}")
            return next(
                record
                for record in self._deliveries[key]
                if record.delivery_id == delivery_id
            )

    def replace_delivery(self, record: DeliveryRecord) -> None:
        """Replace one immutable snapshot while preserving attempt order."""
        with self._lock:
            key = self._delivery_ids.get(record.delivery_id)
            if key is None:
                raise KeyError(f"Unknown delivery_id {record.delivery_id!r}")
            records = self._deliveries[key]
            index = next(
                i
                for i, current in enumerate(records)
                if current.delivery_id == record.delivery_id
            )
            records[index] = record

    def remove_delivery(self, delivery_id: str) -> None:
        """Remove a newly claimed record when durable persistence fails."""
        with self._lock:
            key = self._delivery_ids.pop(delivery_id, None)
            if key is None:
                return
            records = self._deliveries[key]
            self._deliveries[key] = [
                record for record in records if record.delivery_id != delivery_id
            ]
            if not self._deliveries[key]:
                del self._deliveries[key]

    def restore_delivery(self, record: DeliveryRecord) -> None:
        """Restore a persisted record without applying claim transitions."""
        with self._lock:
            key = f"{record.bundle_digest}/{record.hook_id}"
            records = self._deliveries.setdefault(key, [])
            if any(item.delivery_id == record.delivery_id for item in records):
                return
            records.append(record)
            records.sort(key=lambda item: item.started_at)
            self._delivery_ids[record.delivery_id] = key
