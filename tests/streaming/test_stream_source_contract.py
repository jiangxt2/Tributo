"""Contract tests for the ``StreamSource`` protocol.

Lock the protocol semantics that S1/S2 (or third-party implementations)
must preserve: abstract-only protocol, the ``open -> poll -> commit ->
close`` lifecycle, idempotent close, and safe commit without pending
offsets.  Kafka-specific fail-closed behavior lives in
``test_kafka_source.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from tributo.streaming import StreamSource


def test_protocol_is_abstract() -> None:
    """StreamSource cannot be instantiated without all abstract methods."""
    # Runtime-behaviour check: route through Any so the static checker
    # does not flag the intentionally-invalid instantiation.
    stream_source_cls: Any = StreamSource
    with pytest.raises(TypeError):
        stream_source_cls()


class _RecordingSource(StreamSource):
    """Concrete source recording lifecycle events for contract checks."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self._opened = False
        self._closed = False

    def open(self, config: dict[str, Any]) -> None:
        self.events.append("open")
        self._opened = True

    def poll(self, timeout_ms: int = 1000) -> Iterator[dict[str, Any]]:
        if not self._opened:
            raise RuntimeError("source not open — call open() first")
        self.events.append("poll")
        yield {"col": [1, 2]}

    def commit(self) -> None:
        self.events.append("commit")

    def close(self) -> None:
        if self._closed:
            return
        self.events.append("close")
        self._closed = True


def test_concrete_source_follows_lifecycle_order() -> None:
    """open -> poll -> commit -> close is the documented lifecycle."""
    source = _RecordingSource()
    source.open({"topic": "t"})
    batches = list(source.poll())
    source.commit()
    source.close()

    assert batches == [{"col": [1, 2]}]
    assert source.events == ["open", "poll", "commit", "close"]


def test_close_is_idempotent() -> None:
    """Implementations must tolerate repeated close() calls."""
    source = _RecordingSource()
    source.open({})
    source.close()
    source.close()
    assert source.events.count("close") == 1


def test_commit_without_pending_batch_is_safe() -> None:
    """commit() with no yielded batch must not raise (no-op)."""
    source = _RecordingSource()
    source.open({})
    source.commit()
    source.close()


def test_poll_without_open_raises() -> None:
    """poll() before open() must fail loudly, never silently no-op."""
    source = _RecordingSource()
    with pytest.raises(RuntimeError):
        next(source.poll())
