"""StreamSource protocol — unbounded data for online inference.

``StreamSource`` is intentionally separate from ``DataConnector``:
where ``DataConnector.read()`` returns a finite ``ray.data.Dataset``,
``StreamSource`` produces an unbounded stream of micro-batches suitable
for online inference (user churn prediction, real-time anomaly detection).

Design constraints:
- Kafka partition ownership is decoupled from model replica scaling —
  source actors manage offset/rebalance independently; inference replicas
  receive micro-batches through internal queues.
- Delivery semantics (fail-closed): at-least-once is claimed only for
  batches that are actually committed.  A failed commit retains the
  pending offsets and raises a ``StreamSourceError`` subtype so the
  caller can retry; poisoned records raise instead of being skipped.
  Duplicates are handled by downstream idempotency keys.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class StreamSource(ABC):
    """Abstract source for unbounded streaming data.

    Subclasses implement ``poll()`` which yields micro-batches as
    ``dict[str, Any]`` records.  The caller (typically a Ray Serve
    deployment) converts each batch to inference input.
    """

    @abstractmethod
    def open(self, config: dict[str, Any]) -> None:
        """Initialise the source (e.g. connect to Kafka, subscribe to topic).

        Args:
            config: Source-specific configuration (broker, topic, group id).
        """

    @abstractmethod
    def poll(self, timeout_ms: int = 1000) -> Iterator[dict[str, Any]]:
        """Poll for the next micro-batch.

        Args:
            timeout_ms: Max time to wait for a batch.

        Yields:
            Micro-batches as dicts (keys are column names, values are lists).
        """

    @abstractmethod
    def commit(self) -> None:
        """Commit the current offset after successful processing."""

    @abstractmethod
    def close(self) -> None:
        """Gracefully close the source and release resources.

        Pending offsets are NOT committed here — an uncommitted batch
        is replayed from the broker after a restart (fail-closed).
        """
