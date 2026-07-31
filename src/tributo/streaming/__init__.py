"""tributo.streaming — Streaming data sources for online inference.

Separate from ``tributo.data`` (batch connectors) and ``tributo.serving``
(HTTP/gRPC deployment).  ``StreamSource`` is an unbounded data protocol
— it never pretends to return a finite ``ray.data.Dataset``.
"""

from __future__ import annotations

from tributo.streaming.protocol import StreamSource

__all__ = [
    "StreamSource",
]
