"""Message broker abstraction for ML job lifecycle integration.

Provides abstract interfaces for task consumption, event reporting,
and cancellation checking. Third-party implementations (Redis, Kafka,
Pulsar, etc.) will register via the ``tributo.brokers`` entry point group.

.. note::

    Plugin discovery and the ``tributo.brokers`` entry point group are
    planned for v1.1.  The ABCs in this module currently have zero
    concrete implementations — they define the contract that broker
    packages (e.g. ``tributo-broker-redis``) will fulfill.

See Also:
    :mod:`tributo.plugin` — plugin discovery infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Message:
    """A job request consumed from a message queue.

    Attributes:
        job_id: Unique identifier for the job.
        payload: The deserialized request body (e.g. training config dict).
        metadata: Optional routing headers or trace context.
    """

    job_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Outcome of a completed ML training job.

    Attributes:
        job_id: Unique identifier for the job.
        status: Terminal status — ``"success"``, ``"failed"``, or ``"cancelled"``.
        metrics: Final evaluation metrics (e.g. accuracy, loss).
        artifacts: Paths or URIs of produced artifacts (model files, reports).
        error: Human-readable error message when status is ``"failed"``.
    """

    job_id: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


# ── Abstract interfaces ───────────────────────────────────────────────────────


class TaskConsumer(ABC):
    """Consume ML job requests from a message queue.

    Implementations wrap a specific broker client (Redis Streams,
    Kafka consumer group, etc.) and yield :class:`Message` objects.
    """

    @abstractmethod
    def poll(self, timeout_ms: int = 5000) -> Message | None:
        """Block until a job request arrives or *timeout_ms* expires.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.

        Returns:
            A :class:`Message` if one is available, or ``None`` on timeout.
        """
        ...

    @abstractmethod
    def ack(self, message: Message) -> None:
        """Acknowledge successful processing of *message*.

        After ``ack`` the broker guarantees the message will not be
        redelivered.
        """
        ...


class EventReporter(ABC):
    """Publish ML job lifecycle events.

    Every event is keyed by *job_id* so downstream systems can
    reconstruct the full timeline of a job.
    """

    @abstractmethod
    def report_phase(self, job_id: str, phase: str) -> None:
        """Report a lifecycle phase transition.

        Typical phases: ``"initializing"``, ``"training"``,
        ``"exporting"``, ``"completed"``.
        """
        ...

    @abstractmethod
    def report_metrics(
        self, job_id: str, metrics: dict[str, float], progress: float
    ) -> None:
        """Report intermediate training metrics.

        Args:
            job_id: The job identifier.
            metrics: Current metric values (e.g. ``{"loss": 0.35}``).
            progress: Progress fraction in ``[0.0, 1.0]``.
        """
        ...

    @abstractmethod
    def report_completed(self, job_id: str, result: JobResult) -> None:
        """Report job completion with final results."""
        ...

    @abstractmethod
    def report_failed(self, job_id: str, error: str) -> None:
        """Report job failure with error details."""
        ...


class CancellationChecker(ABC):
    """Check whether a running job has been requested to cancel.

    The training loop calls :meth:`is_cancelled` periodically and
    stops early when it returns ``True``.
    """

    @abstractmethod
    def is_cancelled(self, job_id: str) -> bool:
        """Return ``True`` if *job_id* should stop early."""
        ...
