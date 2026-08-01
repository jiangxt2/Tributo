"""Operation store — execution and publication record keeping.

Separates mutable operation state (executions, publication attempts)
from the immutable ``BundleManifest``.  Persistence is deferred to P1;
P0 defines the data models and the ``OperationStore`` protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.manifest import ManifestExecutionNode
from tributo.util.annotations import DeveloperAPI, PublicAPI

# ── Data models ────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ExecutionRecord(BaseModel):
    """Immutable snapshot of a single export execution.

    Written once after commit and never modified.  Separated from
    ``BundleManifest`` so the manifest stays purely about model content.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    bundle_id: str
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

    def list_attempts(
        self, bundle_digest: str, hook_id: str
    ) -> list[PublicationAttempt]: ...


@DeveloperAPI
class InMemoryOperationStore:
    """In-memory ``OperationStore`` for testing and single-process use."""

    def __init__(self) -> None:
        self._executions: dict[str, ExecutionRecord] = {}
        self._attempts: dict[str, list[PublicationAttempt]] = {}

    def record_execution(self, record: ExecutionRecord) -> None:
        self._executions[record.execution_id] = record

    def record_publication_attempt(self, attempt: PublicationAttempt) -> None:
        key = f"{attempt.bundle_digest}/{attempt.hook_id}"
        self._attempts.setdefault(key, []).append(attempt)

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        return self._executions.get(execution_id)

    def list_attempts(
        self, bundle_digest: str, hook_id: str
    ) -> list[PublicationAttempt]:
        return self._attempts.get(f"{bundle_digest}/{hook_id}", [])
