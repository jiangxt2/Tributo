"""KafkaStreamSource — Kafka consumer for online inference streaming.

Reads feature events from a Kafka topic as micro-batches.  Designed to
run as a Ray actor independent of inference replicas — partition
ownership and offset management are decoupled from model scaling.

Delivery semantics (fail-closed safety baseline): at-least-once is
claimed only for batches that are actually committed.  A failed commit
retains the pending offsets and raises ``KafkaCommitError`` so the
caller can retry; poisoned records (message errors, tombstones, decode
failures, non-dict values) raise ``KafkaPoisonMessageError`` and stop
the source instead of being skipped.  Offsets beyond a failed record
are never committed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from tributo.exceptions import (
    KafkaCommitError,
    KafkaPoisonMessageError,
    StreamSourceError,
)
from tributo.streaming.protocol import StreamSource
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
class KafkaStreamSource(StreamSource):
    """Kafka event stream source for online inference.

    Manages Kafka consumer group membership, partition assignment,
    and offset commits.  Each ``poll()`` call returns one micro-batch
    of records from the assigned partitions.

    Example::

        source = KafkaStreamSource()
        source.open({
            "bootstrap_servers": "localhost:9092",
            "topic": "user_features",
            "group_id": "churn_predictor",
            "batch_size": 64,
        })
        for batch in source.poll():
            predictions = model(batch)
            sink.write(predictions)
            source.commit()
    """

    def __init__(self) -> None:
        self._consumer: Any = None
        self._batch_size: int = 64
        self._config: dict[str, Any] = {}
        self._closed: bool = False
        # Set when a poison record terminated the source — distinct from
        # a plain close() so poll() can report the termination cause.
        self._poisoned: bool = False
        # Track max offset per partition for the current batch so
        # commit() issues a single batched RPC (at-least-once semantics).
        self._batch_offsets: dict[tuple[str, int], int] = {}  # (topic, part) → offset+1

    # -- StreamSource interface -----------------------------------------------

    def open(self, config: dict[str, Any]) -> None:
        """Connect to Kafka and subscribe to the configured topic.

        Required config keys:
        ``bootstrap_servers`` (str), ``topic`` (str), ``group_id`` (str).
        Optional: ``batch_size`` (int, default 64), ``auto_offset_reset``
        (str, default ``"latest"``).

        Raises:
            RuntimeError: If the source is already open, closed or
                poison-terminated.  A new consumer is never created on
                an active or terminated instance — create a new
                ``KafkaStreamSource`` to restart.
        """
        if self._closed or self._consumer is not None:
            raise RuntimeError(
                "KafkaStreamSource is closed or already open — "
                "create a new instance to restart"
            )
        self._config = config
        self._batch_size = config.get("batch_size", 64)

        try:
            from confluent_kafka import Consumer

            consumer_config: dict[str, Any] = {
                "bootstrap.servers": config["bootstrap_servers"],
                "group.id": config["group_id"],
                "auto.offset.reset": config.get("auto_offset_reset", "latest"),
                "enable.auto.commit": False,  # manual commit (at-least-once)
                "max.poll.interval.ms": config.get("max_poll_interval_ms", 300000),
                "session.timeout.ms": config.get("session_timeout_ms", 30000),
            }

            self._consumer = Consumer(consumer_config)
            self._consumer.subscribe([config["topic"]])
            logger.info(
                "KafkaStreamSource connected to %s, topic=%s, group=%s",
                config["bootstrap_servers"],
                config["topic"],
                config["group_id"],
            )
        except ImportError as err:
            raise ImportError(
                "confluent-kafka is required for KafkaStreamSource. "
                "Install with: pip install tributo[streaming-inference]"
            ) from err
        except Exception as exc:
            # Do not leak a half-initialised consumer: if subscribe()
            # failed after Consumer() succeeded, close it before raising.
            if self._consumer is not None:
                try:
                    self._consumer.close()
                except Exception:
                    pass
                self._consumer = None
            logger.exception("Failed to connect to Kafka")
            raise ConnectionError(f"Kafka connection failed: {exc}") from exc

    def poll(self, timeout_ms: int = 1000) -> Iterator[dict[str, Any]]:
        """Continuously poll for micro-batches from Kafka.

        Collects up to ``batch_size`` messages per batch, yielding
        each batch as a columnar dict.  The iterator runs until
        ``close()`` is called.

        Fail-closed: a batch that was yielded but not committed
        blocks the next ``poll()`` with ``StreamSourceError`` until the
        caller commits or closes; poisoned records raise
        ``KafkaPoisonMessageError`` and terminate the iterator instead
        of being skipped, so offsets beyond a failed record are never
        committed.

        Args:
            timeout_ms: Max time to wait for each message.

        Yields:
            Columnar batch dicts indefinitely.

        Raises:
            KafkaPoisonMessageError: If a previous poison record
                terminated the source (reason ``"terminated"``), or on
                a fresh poison record (message errors, tombstones,
                decode failures, non-dict values — reasons
                ``"message_error"``/``"tombstone"``/``"decode"``/
                ``"non_dict"``).  The source terminates itself first:
                the consumer is closed and a fresh ``poll()`` raises.
                Re-``open()`` on the same instance does not resume
                polling — create a new ``KafkaStreamSource`` to restart
                (the broker re-delivers from the last committed offset;
                an explicit poison handling strategy is S1 scope).
            RuntimeError: If ``open()`` was not called first (``call
                open() first``), or after a plain ``close()`` (``create
                a new KafkaStreamSource``).
            StreamSourceError: If a pending batch is not committed.
        """
        if self._poisoned:
            raise KafkaPoisonMessageError(
                "KafkaStreamSource was terminated by a poison record — "
                "create a new instance to restart",
                reason="terminated",
            )
        if self._consumer is None:
            if self._closed:
                raise RuntimeError(
                    "KafkaStreamSource is closed — create a new instance to restart"
                )
            raise RuntimeError("KafkaStreamSource not open — call open() first.")

        import json

        from confluent_kafka import KafkaError

        while not self._closed:
            # Fail-closed barrier: a yielded batch must be committed
            # before the next one is polled, otherwise the pending
            # offsets would be overwritten.  The caller either retries
            # commit() (retryable on failure) or closes the source.
            if self._batch_offsets:
                raise StreamSourceError(
                    "Pending Kafka offsets must be committed before polling "
                    "the next batch (call commit() or close())"
                )

            batch: list[dict[str, Any]] = []
            offsets: dict[tuple[str, int], int] = {}  # (topic, part) → max offset+1

            while len(batch) < self._batch_size:
                msg = self._consumer.poll(timeout=timeout_ms / 1000.0)
                if msg is None:
                    break  # no more messages within timeout
                if msg.error():
                    err = msg.error()
                    # Partition EOF is a normal end-of-data marker, not a
                    # poison message — keep polling for other partitions.
                    # _PARTITION_EOF (-191) is confluent-kafka's internal
                    # code; verify against the installed version in the
                    # real-broker integration run.
                    if err.code() == KafkaError._PARTITION_EOF:
                        continue
                    # Fail-closed: terminate the source before raising so
                    # a fresh poll() cannot consume past the bad record.
                    self._terminate(poisoned=True)
                    raise KafkaPoisonMessageError(
                        f"Kafka message error: {err}",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        reason="message_error",
                    )

                # Tombstones (null values) are delete markers; a poison
                # record is raised instead of skipping (fail-closed).
                raw_value = msg.value()
                if raw_value is None:
                    self._terminate(poisoned=True)
                    raise KafkaPoisonMessageError(
                        f"Kafka tombstone (null value) at "
                        f"topic={msg.topic()} partition={msg.partition()} "
                        f"offset={msg.offset()}",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        reason="tombstone",
                    )

                try:
                    value = json.loads(raw_value.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    self._terminate(poisoned=True)
                    raise KafkaPoisonMessageError(
                        f"Failed to decode Kafka message at "
                        f"topic={msg.topic()} partition={msg.partition()} "
                        f"offset={msg.offset()}: {exc}",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        reason="decode",
                    ) from exc

                # Non-dict records (scalars, lists) cannot form a columnar
                # batch — fail closed instead of silently dropping them.
                if not isinstance(value, dict):
                    self._terminate(poisoned=True)
                    raise KafkaPoisonMessageError(
                        f"Non-dict Kafka record (type={type(value).__name__}) "
                        f"at topic={msg.topic()} partition={msg.partition()} "
                        f"offset={msg.offset()}",
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                        reason="non_dict",
                    )

                batch.append(value)
                # Track the max offset per partition (offset+1 = next to read).
                tp = (msg.topic(), msg.partition())
                offsets[tp] = max(offsets.get(tp, 0), msg.offset() + 1)

            if not batch:
                continue

            # Transpose list-of-dicts → dict-of-lists (columnar batch).
            all_keys: set[str] = set()
            for row in batch:
                all_keys.update(row.keys())
            column_batch = {k: [row.get(k) for row in batch] for k in all_keys}
            self._batch_offsets = offsets
            yield column_batch

    def commit(self) -> None:
        """Commit offsets for the last yielded batch in a single RPC.

        Advances each partition's offset to the highest successfully
        yielded message + 1 (at-least-once semantics).  Skipped /
        errored / tombstone records are never committed.

        Fail-closed: on failure the pending offsets are retained
        and ``KafkaCommitError`` is raised, so the caller can retry
        ``commit()`` — offsets are cleared only after a successful
        commit.  The uncommitted batch also blocks further ``poll()``.
        """
        if self._consumer is None or not self._batch_offsets:
            return
        try:
            from confluent_kafka import TopicPartition

            tp_offsets = [
                TopicPartition(tp[0], tp[1], offset)
                for tp, offset in self._batch_offsets.items()
            ]
            self._consumer.commit(offsets=tp_offsets, asynchronous=False)
            logger.debug("Committed %d Kafka partition(s)", len(tp_offsets))
        except Exception as exc:
            raise KafkaCommitError(f"Kafka offset commit failed: {exc}") from exc
        self._batch_offsets.clear()

    def _terminate(self, *, poisoned: bool = False) -> None:
        """Close the consumer and stop the source.

        Shared by the graceful ``close()`` path (``poisoned=False``) and
        the poison fail-closed path (``poisoned=True``, which marks the
        source so a fresh ``poll()`` reports the termination cause).
        """
        self._closed = True
        if poisoned:
            self._poisoned = True
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception as exc:
                logger.warning("Error closing Kafka consumer: %s", exc)
            self._consumer = None

    def close(self) -> None:
        """Gracefully close the Kafka consumer.

        Pending offsets are left uncommitted (no automatic commit): a
        batch yielded but not committed is replayed from the broker
        after a restart (fail-closed, S0).
        """
        if self._closed:
            return
        self._terminate()
        logger.info("KafkaStreamSource closed.")

    def __del__(self) -> None:
        self.close()
