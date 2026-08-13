"""Core Broker SPI, lazy discovery, runner, and worker-context tests."""

from __future__ import annotations

import json
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

import tributo.plugin as plugin
from tributo.exceptions import JobConfigurationError
from tributo.integrations.broker import (
    BROKER_API_VERSION,
    BrokerPlugin,
    BrokerRuntime,
    CancellationChecker,
    CancellationSpec,
    JobResult,
    Message,
    TaskConsumer,
    TaskDisposition,
    TaskOutcome,
)
from tributo.integrations.broker_registry import (
    BrokerRegistry,
    rebuild_cancellation_checker,
)
from tributo.integrations.broker_runner import BrokerRunner, BrokerRunnerState


class _EntryPoint:
    def __init__(self, name: str, loaded: Any) -> None:
        self.name = name
        self.value = "tests:_Plugin"
        self._loaded = loaded

    def load(self) -> Any:
        if isinstance(self._loaded, Exception):
            raise self._loaded
        return self._loaded


class _Consumer(TaskConsumer):
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages
        self.acked: list[Message] = []
        self.retried: list[Message] = []

    def poll(self, timeout_ms: int = 5000) -> Message | None:
        del timeout_ms
        return self.messages.pop(0) if self.messages else None

    def ack(self, message: Message) -> None:
        self.acked.append(message)

    def retry(self, message: Message, error: str | None = None) -> None:
        del error
        self.retried.append(message)


class _Runtime(BrokerRuntime):
    def __init__(self, consumer: _Consumer, outcome: TaskOutcome) -> None:
        self._consumer = consumer
        self.outcome = outcome

    @property
    def consumer(self) -> _Consumer:
        return self._consumer

    def handle(self, message: Message) -> TaskOutcome:
        del message
        return self.outcome


class _Plugin(BrokerPlugin):
    api_version: ClassVar[int] = BROKER_API_VERSION
    broker_id: ClassVar[str] = "fake"
    capabilities: ClassVar[frozenset[str]] = frozenset({"task-consumer"})
    runtime: _Runtime

    def validate_config(self, config, *, check_connectivity=False) -> None:
        del config, check_connectivity

    def create_runtime(self, config) -> _Runtime:
        del config
        return self.runtime

    def create_cancellation_checker(
        self, spec: CancellationSpec
    ) -> CancellationChecker:
        return _Checker(spec.job_id)


class _Checker(CancellationChecker):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def is_cancelled(self, job_id: str) -> bool:
        return job_id == self.job_id


def test_cancellation_spec_is_json_safe_and_rejects_client_objects() -> None:
    spec = CancellationSpec("fake", "job-1", {"secret_ref": "env:REDIS_PASSWORD"})
    assert json.loads(json.dumps(spec.as_dict()))["job_id"] == "job-1"
    with pytest.raises(ValueError, match="JSON serializable"):
        CancellationSpec("fake", "job-1", {"client": object()})


def test_discovery_is_lazy_and_records_import_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda group: iter(
            [
                _EntryPoint("broken", ImportError("redis secret")),
                _EntryPoint("fake", _Plugin),
            ]
            if group == "tributo.brokers"
            else []
        ),
    )
    diagnostics = []
    classes = plugin.discover_broker_plugins(diagnostics)
    assert classes == [_Plugin]
    assert diagnostics[0].entry_point_name == "broken"
    assert "redis secret" in diagnostics[0].reason


def test_discovery_rejects_non_frozen_capabilities(monkeypatch) -> None:
    class _TupleCapabilitiesPlugin(_Plugin):
        capabilities = ("task-consumer",)

    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda group: iter(
            [_EntryPoint("tuple", _TupleCapabilitiesPlugin)]
            if group == "tributo.brokers"
            else []
        ),
    )
    diagnostics = []
    assert plugin.discover_broker_plugins(diagnostics) == []
    assert "capabilities" in diagnostics[0].reason


def test_discovery_rejects_api_version_and_entrypoint_identity_mismatch(
    monkeypatch,
) -> None:
    class _WrongVersionPlugin(_Plugin):
        api_version = BROKER_API_VERSION + 1

    class _WrongIdentityPlugin(_Plugin):
        broker_id = "other"

    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda group: iter(
            [
                _EntryPoint("wrong-version", _WrongVersionPlugin),
                _EntryPoint("fake", _WrongIdentityPlugin),
            ]
            if group == "tributo.brokers"
            else []
        ),
    )
    diagnostics = []
    assert plugin.discover_broker_plugins(diagnostics) == []
    assert len(diagnostics) == 2
    assert "api_version" in diagnostics[0].reason
    assert "does not match broker_id" in diagnostics[1].reason


def test_explicit_broker_filtered_by_tributo_plugins_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("TRIBUTO_PLUGINS", "another")
    monkeypatch.setattr(
        plugin, "_iter_entry_points", lambda group: iter([_EntryPoint("fake", _Plugin)])
    )
    with pytest.raises(JobConfigurationError, match="disabled"):
        plugin.resolve_broker_plugin("fake")


def test_registry_reports_duplicate_broker_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.discover_broker_plugins",
        lambda _diagnostics: [_Plugin, _Plugin],
    )
    registry = BrokerRegistry()
    assert len(registry.list()) == 1
    assert registry.diagnostics()[0].reason == "Duplicate broker_id discovered"


def test_runner_acks_only_ack_outcome_and_has_lifecycle_state() -> None:
    message = Message("job-1", {})
    consumer = _Consumer([message])
    plugin_instance = _Plugin()
    plugin_instance.runtime = _Runtime(
        consumer,
        TaskOutcome(
            TaskDisposition.ACK,
            result=JobResult("job-1", "accepted", run_id="job-1"),
        ),
    )
    runner = BrokerRunner(plugin_instance, {})
    assert runner.state == BrokerRunnerState.STOPPED
    assert runner.run_once() is True
    assert runner.state == BrokerRunnerState.READY
    assert consumer.acked == [message]
    runner.close()
    assert runner.state == BrokerRunnerState.STOPPED


def test_runner_retains_retryable_message() -> None:
    message = Message("job-1", {})
    consumer = _Consumer([message])
    plugin_instance = _Plugin()
    plugin_instance.runtime = _Runtime(
        consumer,
        TaskOutcome(TaskDisposition.RETRY, error="ray unavailable"),
    )
    runner = BrokerRunner(plugin_instance, {})
    assert runner.run_once() is True
    assert consumer.acked == []
    assert consumer.retried == [message]


def test_runner_rejects_without_acknowledging() -> None:
    message = Message("job-1", {})
    consumer = _Consumer([message])
    plugin_instance = _Plugin()
    plugin_instance.runtime = _Runtime(
        consumer,
        TaskOutcome(TaskDisposition.REJECT, error="poison"),
    )
    runner = BrokerRunner(plugin_instance, {})
    assert runner.run_once() is True
    assert consumer.acked == []
    assert consumer.retried == []


def test_runner_contains_ack_failure_and_enters_reconnect() -> None:
    message = Message("job-1", {})
    consumer = _Consumer([message])
    consumer.ack = MagicMock(side_effect=ConnectionError("redis down"))
    plugin_instance = _Plugin()
    plugin_instance.runtime = _Runtime(
        consumer,
        TaskOutcome(TaskDisposition.ACK),
    )
    runner = BrokerRunner(
        plugin_instance,
        {},
        backoff_initial=0.001,
        backoff_max=0.001,
        sleep=lambda _delay: None,
    )
    assert runner.run_once() is True
    assert runner.state == BrokerRunnerState.RECONNECTING
    consumer.ack.assert_called_once_with(message)


def test_runner_graceful_stop_stops_next_poll() -> None:
    consumer = _Consumer([])
    plugin_instance = _Plugin()
    plugin_instance.runtime = _Runtime(
        consumer,
        TaskOutcome(TaskDisposition.ACK),
    )
    runner = BrokerRunner(plugin_instance, {})
    runner.start()
    runner.request_stop()
    assert runner.state == BrokerRunnerState.STOPPING
    assert runner.run_once() is False
    runner.close()
    assert runner.state == BrokerRunnerState.STOPPED


def test_worker_checker_rebuilt_from_spec(monkeypatch) -> None:
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.BrokerRegistry.resolve",
        lambda _self, broker_id: _Plugin(),
    )
    checker = rebuild_cancellation_checker(
        {"broker_id": "fake", "job_id": "job-1", "options": {}}
    )
    assert isinstance(checker, _Checker)
    assert checker.is_cancelled("job-1") is True


def test_missing_cancellation_context_keeps_training_context_empty() -> None:
    assert rebuild_cancellation_checker(None) is None
