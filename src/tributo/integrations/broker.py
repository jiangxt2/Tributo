"""Broker extension contracts for Tributo.

The contracts in this module are deliberately transport-neutral.  Redis,
Kafka, RabbitMQ, and protocol-specific models belong to separately installed
provider packages and are discovered only when a broker operation is
explicitly requested.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

BROKER_API_VERSION = 1


class TaskDisposition(StrEnum):
    """Decision the generic runner applies after a provider handles a task."""

    ACK = "ack"
    RETRY = "retry"
    REJECT = "reject"


@dataclass
class Message:
    """A job request consumed from a message queue.

    Existing callers may continue constructing a message with only
    ``job_id`` and ``payload``.  Transport metadata is optional because the
    public Core contract must also support brokers without Redis-style
    delivery IDs.
    """

    job_id: str | None
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    delivery_id: str | None = None
    delivery_attempt: int = 1
    run_id: str | None = None
    attempt_id: str | None = None


@dataclass
class JobResult:
    """Outcome of a completed ML training job.

    The identity fields are optional for backwards-compatible construction;
    provider implementations should populate them whenever a broker task is
    used.
    """

    job_id: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    execution_id: str | None = None
    submission_id: str | None = None
    bundle_id: str | None = None
    bundle_uri: str | None = None
    manifest_uri: str | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CancellationSpec:
    """JSON-safe instructions for reconstructing a worker-side checker.

    This object intentionally contains no client, socket, pool, or secret.
    ``options`` may contain provider-specific non-sensitive values or secret
    references, but not credentials themselves.
    """

    broker_id: str
    job_id: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.broker_id.strip():
            raise ValueError("CancellationSpec.broker_id must not be empty")
        if not self.job_id.strip():
            raise ValueError("CancellationSpec.job_id must not be empty")
        try:
            json.dumps(self.as_dict(), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("CancellationSpec must be JSON serializable") from exc

    def as_dict(self) -> dict[str, Any]:
        """Return the wire-safe representation placed in Ray config."""
        return {
            "broker_id": self.broker_id,
            "job_id": self.job_id,
            "options": dict(self.options),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CancellationSpec":
        """Validate and reconstruct a spec received by a Ray worker."""
        broker_id = value.get("broker_id")
        job_id = value.get("job_id")
        options = value.get("options", {})
        if not isinstance(broker_id, str) or not isinstance(job_id, str):
            raise ValueError("CancellationSpec requires string broker_id and job_id")
        if not isinstance(options, dict):
            raise ValueError("CancellationSpec.options must be an object")
        return cls(broker_id=broker_id, job_id=job_id, options=dict(options))


@dataclass
class TaskOutcome:
    """Provider result consumed by :class:`BrokerRunner`."""

    disposition: TaskDisposition
    result: JobResult | None = None
    error: str | None = None


class TaskConsumer(ABC):
    """Consume ML job requests from a message queue."""

    @abstractmethod
    def poll(self, timeout_ms: int = 5000) -> Message | None:
        """Block until a job request arrives or *timeout_ms* expires."""
        ...

    @abstractmethod
    def ack(self, message: Message) -> None:
        """Acknowledge successful processing of *message*."""
        ...

    def retry(self, message: Message, error: str | None = None) -> None:
        """Leave a temporarily failed message eligible for redelivery.

        Brokers such as Redis Streams retain a pending message by doing
        nothing here.  Brokers with an explicit negative acknowledgement may
        override this method.
        """
        del message, error

    def reject(self, message: Message, error: str | None = None) -> None:
        """Reject a permanently invalid message without acknowledging it.

        A provider should normally return ``ACK`` after publishing a FAILED
        event for a poison message.  This hook exists for transports that have
        a dead-letter operation.
        """
        del message, error

    def recover_pending(self) -> int:
        """Reclaim pending messages when the transport supports it."""
        return 0

    def close(self) -> None:
        """Close transport resources; the default is intentionally a no-op."""
        return None


class EventReporter(ABC):
    """Publish ML job lifecycle events."""

    @abstractmethod
    def report_phase(self, job_id: str, phase: str) -> None:
        """Report a lifecycle phase transition."""
        ...

    @abstractmethod
    def report_metrics(
        self, job_id: str, metrics: dict[str, float], progress: float
    ) -> None:
        """Report metrics; v1 uses this for post-training history replay."""
        ...

    @abstractmethod
    def report_completed(self, job_id: str, result: JobResult) -> None:
        """Report job completion with final results."""
        ...

    @abstractmethod
    def report_failed(self, job_id: str, error: str) -> None:
        """Report job failure with error details."""
        ...

    def report_log(self, job_id: str, message: str, level: str = "INFO") -> None:
        """Optionally report a user-visible log line."""
        del job_id, message, level

    def report_cancelled(self, job_id: str, phase: str = "TRAINING") -> None:
        """Optionally report cooperative cancellation."""
        del job_id, phase


class CancellationChecker(ABC):
    """Check whether a running job has been requested to cancel."""

    @abstractmethod
    def is_cancelled(self, job_id: str) -> bool:
        """Return ``True`` if *job_id* should stop early."""
        ...


class BrokerRuntime(ABC):
    """Provider-owned runtime assembled by the generic Core runner."""

    @property
    @abstractmethod
    def consumer(self) -> TaskConsumer:
        """Return the transport consumer."""
        ...

    @abstractmethod
    def handle(self, message: Message) -> TaskOutcome:
        """Validate and process one message without transport ACK side effects."""
        ...

    def close(self) -> None:
        """Close provider resources."""
        self.consumer.close()


class BrokerPlugin(ABC):
    """Structural base for an independently installed broker plugin."""

    api_version: int = BROKER_API_VERSION
    broker_id: str
    capabilities: frozenset[str] = frozenset()

    @abstractmethod
    def validate_config(
        self, config: Mapping[str, Any], *, check_connectivity: bool = False
    ) -> None:
        """Validate provider-owned config and optionally probe connectivity."""
        ...

    @abstractmethod
    def create_runtime(self, config: Mapping[str, Any]) -> BrokerRuntime:
        """Create a provider runtime; no network I/O belongs in discovery."""
        ...

    @abstractmethod
    def create_cancellation_checker(
        self, spec: CancellationSpec
    ) -> CancellationChecker:
        """Rebuild a worker-side checker from a JSON-safe spec."""
        ...
