"""tributo.streaming — Streaming data sources for online inference.

Separate from ``tributo.data`` (bounded ingestion) and ``tributo.serving``
(HTTP/gRPC deployment).  ``StreamSource`` is an unbounded data protocol
— it never pretends to return a finite ``ray.data.Dataset``.
"""

from __future__ import annotations

from tributo.exceptions import (
    KafkaCommitError,
    KafkaPoisonMessageError,
    StreamSourceError,
)
from tributo.streaming.protocol import StreamSource

__all__ = [
    "StreamSource",
    "StreamSourceError",
    "KafkaCommitError",
    "KafkaPoisonMessageError",
]
