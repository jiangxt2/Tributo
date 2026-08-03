"""Fail-closed safety tests for ``KafkaStreamSource``.

``confluent-kafka`` is an optional dependency, so the consumer is faked
by injecting a fake ``confluent_kafka`` module into ``sys.modules``.
``KafkaStreamSource.open()`` instantiates the consumer itself, so the
fake is programmed via class attributes (message queue + commit
failure) rather than instance injection.  Each test asserts the exit
gates: pending offsets are never overwritten, failed commits retain
offsets for retry, and poisoned records stop the source instead of
being skipped (offsets beyond a failed record are never committed).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from tributo.exceptions import (
    KafkaCommitError,
    KafkaPoisonMessageError,
    StreamSourceError,
)
from tributo.streaming.kafka_source import KafkaStreamSource


class _FakeKafkaError:
    """Mirror of ``confluent_kafka.KafkaError`` constants used here."""

    _PARTITION_EOF = -191


class _FakeError:
    """Message error object: ``code()`` mirrors confluent semantics."""

    def __init__(self, code: int, text: str = "") -> None:
        self._code = code
        self._text = text

    def code(self) -> int:
        return self._code

    def __str__(self) -> str:
        return self._text


class _FakeMessage:
    def __init__(
        self,
        topic: str = "t",
        partition: int = 0,
        offset: int = 0,
        value: bytes | None = None,
        error: _FakeError | None = None,
    ) -> None:
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._value = value
        self._error = error

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def value(self) -> bytes | None:
        return self._value

    def error(self) -> _FakeError | None:
        return self._error


class _FakeTopicPartition:
    def __init__(self, topic: str, partition: int, offset: int) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeTopicPartition) and (
            self.topic,
            self.partition,
            self.offset,
        ) == (other.topic, other.partition, other.offset)

    def __repr__(self) -> str:
        return f"_FakeTopicPartition({self.topic!r}, {self.partition}, {self.offset})"


class _FakeConsumer:
    """Programmable consumer.

    ``open()`` instantiates ``Consumer(config)`` internally, so the
    message queue and commit-failure flag are class attributes that
    ``_open_source`` programmes before opening.
    """

    messages: list[_FakeMessage] = []
    fail_commit: bool = False
    fail_subscribe: bool = False
    instances: list["_FakeConsumer"] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._messages = list(self.messages)
        self.fail_commit = type(self).fail_commit  # instance snapshot
        self.subscribed: list[str] = []
        self.closed = False
        self.commit_calls: list[list[_FakeTopicPartition]] = []
        type(self).instances.append(self)

    def subscribe(self, topics: list[str]) -> None:
        if type(self).fail_subscribe:
            raise RuntimeError("subscribe boom")
        self.subscribed = topics

    def poll(self, timeout: float) -> _FakeMessage | None:
        return self._messages.pop(0) if self._messages else None

    def commit(
        self,
        offsets: list[_FakeTopicPartition] | None = None,
        asynchronous: bool = False,
    ) -> None:
        if self.fail_commit:
            raise RuntimeError("commit boom")
        if offsets is not None:
            self.commit_calls.append(offsets)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_kafka(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``confluent_kafka`` module (the package is optional)."""

    module = types.ModuleType("confluent_kafka")
    # Route attribute injection through Any: types.ModuleType has no
    # declared attributes, and the fake must mirror the real package.
    fake_module: Any = module
    fake_module.Consumer = _FakeConsumer
    fake_module.TopicPartition = _FakeTopicPartition
    fake_module.KafkaError = _FakeKafkaError
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)


def _msg(
    offset: int = 0,
    value: bytes | None = b'{"a": 1}',
    error: _FakeError | None = None,
    partition: int = 0,
) -> _FakeMessage:
    return _FakeMessage(partition=partition, offset=offset, value=value, error=error)


def _open_source(
    messages: list[_FakeMessage],
    *,
    fail_commit: bool = False,
    batch_size: int = 64,
) -> KafkaStreamSource:
    _FakeConsumer.messages = messages
    _FakeConsumer.fail_commit = fail_commit
    _FakeConsumer.fail_subscribe = False
    source = KafkaStreamSource()
    source.open(
        {
            "bootstrap_servers": "localhost:9092",
            "topic": "t",
            "group_id": "g",
            "batch_size": batch_size,
        }
    )
    return source


def test_poll_yields_columnar_batch_and_tracks_offsets(fake_kafka: None) -> None:
    source = _open_source(
        [_msg(0, b'{"a": 1}'), _msg(1, b'{"b": 2}'), _msg(2, b'{"a": 3}')]
    )

    batch = next(source.poll())

    assert batch == {"a": [1, None, 3], "b": [None, 2, None]}
    assert source._batch_offsets == {("t", 0): 3}


def test_batch_size_limits_collection(fake_kafka: None) -> None:
    source = _open_source([_msg(0), _msg(1), _msg(2)], batch_size=2)

    it = source.poll()
    assert next(it) == {"a": [1, 1]}
    # A yielded batch must be committed before the next one is polled
    # Fail-closed barrier; commit then continue.
    source.commit()
    assert next(it) == {"a": [1]}
    # The iterator runs until close() (the fake poll returns immediately
    # instead of blocking on the broker, so close is what terminates it).
    source.close()
    with pytest.raises(StopIteration):
        next(it)


def test_commit_success_clears_pending_and_advances(fake_kafka: None) -> None:
    source = _open_source([_msg(0), _msg(1), _msg(2), _msg(3)], batch_size=2)
    consumer = source._consumer

    it = source.poll()
    next(it)
    source.commit()

    assert consumer.commit_calls == [[_FakeTopicPartition("t", 0, 2)]]
    assert source._batch_offsets == {}
    assert next(it) == {"a": [1, 1]}
    source.close()


def test_commit_failure_retains_offsets_and_raises(fake_kafka: None) -> None:
    source = _open_source([_msg(0)], fail_commit=True)

    next(source.poll())

    with pytest.raises(KafkaCommitError) as excinfo:
        source.commit()
    assert "commit" in str(excinfo.value)
    # Pending offsets are retained for retry (never cleared on failure).
    assert source._batch_offsets == {("t", 0): 1}


def test_commit_retry_after_failure_succeeds(fake_kafka: None) -> None:
    source = _open_source([_msg(0), _msg(1), _msg(2)], batch_size=2, fail_commit=True)
    consumer = source._consumer

    it = source.poll()
    next(it)
    with pytest.raises(KafkaCommitError):
        source.commit()
    # The failed batch still blocks further polling until committed.
    with pytest.raises(StreamSourceError):
        next(it)

    consumer.fail_commit = False
    source.commit()
    assert source._batch_offsets == {}
    # The barrier raises inside the generator, which terminates it — the
    # caller resumes by opening a fresh iterator after the commit.
    fresh = source.poll()
    assert next(fresh) == {"a": [1]}
    source.close()


def test_uncommitted_batch_blocks_next_poll(fake_kafka: None) -> None:
    source = _open_source([_msg(0), _msg(1), _msg(2)], batch_size=2)
    consumer = source._consumer

    it = source.poll()
    next(it)

    with pytest.raises(StreamSourceError):
        next(it)
    # Nothing was committed or overwritten while the batch was pending.
    assert source._batch_offsets == {("t", 0): 2}
    assert consumer.commit_calls == []


def test_msg_error_raises_poison(fake_kafka: None) -> None:
    source = _open_source([_msg(0, error=_FakeError(1, "kafka error")), _msg(1)])

    with pytest.raises(KafkaPoisonMessageError) as excinfo:
        next(source.poll())
    assert excinfo.value.reason == "message_error"
    assert (excinfo.value.topic, excinfo.value.partition, excinfo.value.offset) == (
        "t",
        0,
        0,
    )


def test_partition_eof_is_not_poison(fake_kafka: None) -> None:
    source = _open_source(
        [_msg(0, error=_FakeError(_FakeKafkaError._PARTITION_EOF)), _msg(1)]
    )

    assert next(source.poll()) == {"a": [1]}


def test_tombstone_raises_poison(fake_kafka: None) -> None:
    source = _open_source([_msg(0, value=None), _msg(1)])

    with pytest.raises(KafkaPoisonMessageError) as excinfo:
        next(source.poll())
    assert excinfo.value.reason == "tombstone"


def test_decode_failure_raises_poison(fake_kafka: None) -> None:
    source = _open_source([_msg(0, value=b"not-json"), _msg(1)])

    with pytest.raises(KafkaPoisonMessageError) as excinfo:
        next(source.poll())
    assert excinfo.value.reason == "decode"


def test_non_dict_raises_poison(fake_kafka: None) -> None:
    source = _open_source([_msg(0, value=b"[1, 2]"), _msg(1)])

    with pytest.raises(KafkaPoisonMessageError) as excinfo:
        next(source.poll())
    assert excinfo.value.reason == "non_dict"


def test_poison_stops_iterator_and_never_commits_past_bad_offset(
    fake_kafka: None,
) -> None:
    """Offset N fails -> offsets N+1 (and beyond) are never committed."""
    source = _open_source(
        [_msg(0, b'{"a": 1}'), _msg(1, b"bad-json"), _msg(2, b'{"a": 3}')]
    )
    consumer = source._consumer

    with pytest.raises(KafkaPoisonMessageError):
        next(source.poll())

    # The iterator is dead: no batch was yielded, so nothing to commit,
    # and the healthy record past the poison one was never consumed.
    assert source._batch_offsets == {}
    source.commit()
    assert consumer.commit_calls == []


def test_poison_terminates_source_for_fresh_poll(fake_kafka: None) -> None:
    """After a poison record the source terminates itself: a fresh
    ``poll()`` must not consume past the bad offset (exit gate: never
    commit N+1 through a restarted iterator)."""
    source = _open_source(
        [_msg(0, b'{"a": 1}'), _msg(1, b"bad-json"), _msg(2, b'{"a": 3}')]
    )
    consumer = source._consumer

    with pytest.raises(KafkaPoisonMessageError):
        next(source.poll())

    # The source terminated itself: consumer closed, and a fresh poll()
    # refuses to read (poison cause) instead of consuming past the bad
    # record.
    assert consumer.closed
    assert source._consumer is None
    with pytest.raises(KafkaPoisonMessageError) as excinfo:
        next(source.poll())
    assert excinfo.value.reason == "terminated"
    assert consumer.commit_calls == []


def test_close_then_fresh_poll_raises(fake_kafka: None) -> None:
    """After a plain close() a fresh poll() raises (no consumer open),
    distinct from the poison cause, with a close-specific message."""
    source = _open_source([_msg(0)])
    source.close()

    with pytest.raises(RuntimeError, match="create a new instance"):
        next(source.poll())


def test_active_reopen_rejected(fake_kafka: None) -> None:
    """open() on an already-open instance is rejected before a second
    consumer is created, so no Kafka connection leaks."""
    source = _open_source([_msg(0)])
    first_consumer = source._consumer

    with pytest.raises(RuntimeError):
        source.open(
            {
                "bootstrap_servers": "localhost:9092",
                "topic": "t",
                "group_id": "g",
            }
        )
    # The original consumer is untouched and still closable.
    assert source._consumer is first_consumer
    source.close()
    assert first_consumer.closed


def test_open_failure_closes_half_initialised_consumer(
    fake_kafka: None,
) -> None:
    """If subscribe() fails after Consumer() succeeded, the consumer is
    closed and the reference cleared (no leak on open failure)."""
    _FakeConsumer.messages = []
    _FakeConsumer.fail_commit = False
    _FakeConsumer.fail_subscribe = True
    source = KafkaStreamSource()

    with pytest.raises(ConnectionError):
        source.open(
            {
                "bootstrap_servers": "localhost:9092",
                "topic": "t",
                "group_id": "g",
            }
        )
    assert source._consumer is None
    # The half-initialised consumer was closed.
    assert _FakeConsumer.instances[-1].closed


def test_reopen_same_instance_rejected(fake_kafka: None) -> None:
    """close() terminates the source: open() on the same instance is
    rejected before any consumer is created, so no Kafka connection
    leaks.  Restarting requires a new instance (documented in
    open()/poll() — no in-place recovery is promised)."""
    source = _open_source([_msg(0)])
    source.close()

    with pytest.raises(RuntimeError):
        source.open(
            {
                "bootstrap_servers": "localhost:9092",
                "topic": "t",
                "group_id": "g",
            }
        )
    assert source._consumer is None


def test_poll_before_open_raises(fake_kafka: None) -> None:
    source = KafkaStreamSource()
    with pytest.raises(RuntimeError):
        next(source.poll())


def test_commit_before_open_is_noop(fake_kafka: None) -> None:
    source = KafkaStreamSource()
    source.commit()  # no consumer, no pending batch — safe no-op


def test_close_is_idempotent_and_closes_consumer(fake_kafka: None) -> None:
    source = _open_source([])
    consumer = source._consumer
    source.close()
    source.close()
    assert consumer.closed
    assert source._consumer is None


def test_poll_after_close_stops(fake_kafka: None) -> None:
    source = _open_source([_msg(0)])
    it = source.poll()
    next(it)
    source.close()
    with pytest.raises(StopIteration):
        next(it)


def test_poll_batch_across_partitions_tracks_per_partition_offsets(
    fake_kafka: None,
) -> None:
    source = _open_source(
        [_msg(0, partition=0), _msg(0, partition=1), _msg(5, partition=0)]
    )

    next(source.poll())

    assert source._batch_offsets == {("t", 0): 6, ("t", 1): 1}
