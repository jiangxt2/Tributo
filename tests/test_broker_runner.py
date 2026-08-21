"""Core drives provider maintenance on every broker loop iteration."""

from __future__ import annotations

from unittest.mock import MagicMock

from tributo.integrations.broker import (
    BrokerRuntime,
    Message,
    TaskConsumer,
    TaskDisposition,
    TaskOutcome,
)
from tributo.integrations.broker_runner import BrokerRunner, BrokerRunnerState


def _runner(message: Message | None) -> tuple[BrokerRunner, MagicMock, MagicMock]:
    consumer = MagicMock()
    consumer.poll.return_value = message
    consumer.recover_pending.return_value = 0
    runtime = MagicMock()
    runtime.consumer = consumer
    runtime.handle.return_value = TaskOutcome(TaskDisposition.ACK)
    plugin = MagicMock()
    plugin.broker_id = "fake"
    plugin.create_runtime.return_value = runtime
    return BrokerRunner(plugin, {}, sleep=lambda _: None), runtime, consumer


def test_runner_ticks_maintenance_before_message_poll() -> None:
    message = Message({}, "7-0")
    runner, runtime, consumer = _runner(message)
    calls: list[str] = []
    runtime.maintain.side_effect = lambda: calls.append("maintain")
    consumer.poll.side_effect = lambda _timeout: calls.append("poll") or message

    assert runner.run_once() is True

    assert calls == ["maintain", "poll"]
    runtime.maintain.assert_called_once_with()
    consumer.poll.assert_called_once_with(5000)
    runtime.handle.assert_called_once_with(message)
    consumer.ack.assert_called_once_with(message)


def test_runner_ticks_maintenance_during_idle_poll() -> None:
    runner, runtime, consumer = _runner(None)

    assert runner.run_once() is False

    runtime.maintain.assert_called_once_with()
    consumer.poll.assert_called_once_with(5000)


def test_maintenance_failure_closes_runtime_and_enters_reconnect() -> None:
    runner, runtime, consumer = _runner(None)
    runtime.maintain.side_effect = ConnectionError("supervisor unavailable")

    assert runner.run_once() is False

    consumer.poll.assert_not_called()
    runtime.close.assert_called_once_with()
    assert runner.state is BrokerRunnerState.RECONNECTING


def test_default_runtime_maintenance_is_noop() -> None:
    class Consumer(TaskConsumer):
        def poll(self, timeout_ms: int = 5000) -> Message | None:
            return None

        def ack(self, message: Message) -> None:
            return None

    class Runtime(BrokerRuntime):
        consumer = Consumer()

        def handle(self, message: Message) -> TaskOutcome:
            return TaskOutcome(TaskDisposition.ACK)

    assert Runtime().maintain() is None
