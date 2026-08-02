"""KafkaStreamSource — Kafka consumer for online inference streaming.

Reads feature events from a Kafka topic as micro-batches.  Designed to
run as a Ray actor independent of inference replicas — partition
ownership and offset management are decoupled from model scaling.

Delivery semantics: at-least-once.  Offsets are committed only after
the inference result has been accepted by the durable sink.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

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
        # Track max offset per partition for the current batch so
        # commit() issues a single batched RPC (at-least-once semantics).
        self._batch_offsets: dict[Any, int] = {}  # TopicPartition → offset+1

    # -- StreamSource interface -----------------------------------------------

    def open(self, config: dict[str, Any]) -> None:
        """Connect to Kafka and subscribe to the configured topic.

        Required config keys:
        ``bootstrap_servers`` (str), ``topic`` (str), ``group_id`` (str).
        Optional: ``batch_size`` (int, default 64), ``auto_offset_reset``
        (str, default ``"latest"``).
        """
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
            logger.exception("Failed to connect to Kafka")
            raise ConnectionError(f"Kafka connection failed: {exc}") from exc

    def poll(self, timeout_ms: int = 1000) -> Iterator[dict[str, Any]]:
        """Continuously poll for micro-batches from Kafka.

        Collects up to ``batch_size`` messages per batch, yielding
        each batch as a columnar dict.  The iterator runs until
        ``close()`` is called.

        Args:
            timeout_ms: Max time to wait for each message.

        Yields:
            Columnar batch dicts indefinitely.
        """
        if self._consumer is None:
            raise RuntimeError("KafkaStreamSource not open — call open() first.")

        import json

        while not self._closed:
            batch: list[dict[str, Any]] = []
            offsets: dict[Any, int] = {}  # TopicPartition → max offset+1

            while len(batch) < self._batch_size:
                msg = self._consumer.poll(timeout=timeout_ms / 1000.0)
                if msg is None:
                    break  # no more messages within timeout
                if msg.error():
                    logger.warning("Kafka message error: %s", msg.error())
                    continue

                # Guard against tombstone / null-value records.
                raw_value = msg.value()
                if raw_value is None:
                    logger.debug("Skipping tombstone (null-value) Kafka message")
                    continue

                try:
                    value = json.loads(raw_value.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Failed to decode Kafka message: %s", exc)
                    continue

                # Skip non-dict records (scalars, lists).
                if not isinstance(value, dict):
                    logger.warning(
                        "Skipping non-dict Kafka record (type=%s)",
                        type(value).__name__,
                    )
                    continue

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
            logger.warning("Kafka commit failed: %s", exc)
        finally:
            self._batch_offsets.clear()

    def close(self) -> None:
        """Gracefully close the Kafka consumer."""
        if self._closed:
            return
        self._closed = True
        if self._consumer is not None:
            try:
                self._consumer.close()
                logger.info("KafkaStreamSource closed.")
            except Exception as exc:
                logger.warning("Error closing Kafka consumer: %s", exc)
            self._consumer = None

    def __del__(self) -> None:
        self.close()
