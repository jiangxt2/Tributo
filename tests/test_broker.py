"""Core Broker API v1 and lazy provider-discovery tests."""

from __future__ import annotations

import operator
from typing import Any, ClassVar, cast

import pytest

import tributo.integrations.broker as broker_contract
import tributo.plugin as plugin
from tributo.exceptions import JobConfigurationError
from tributo.integrations.broker import (
    BROKER_API_VERSION,
    BrokerError,
    BrokerPlugin,
    BrokerRuntime,
    Message,
    TaskConsumer,
    TaskDisposition,
    TaskOutcome,
)
from tributo.integrations.broker_registry import BrokerRegistry


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
    def poll(self, timeout_ms: int = 5000) -> Message | None:
        del timeout_ms
        return None

    def ack(self, message: Message) -> None:
        del message


class _Runtime(BrokerRuntime):
    consumer = _Consumer()

    def handle(self, message: Message) -> TaskOutcome:
        del message
        return TaskOutcome(TaskDisposition.ACK)


class _Plugin(BrokerPlugin):
    api_version: ClassVar[int] = BROKER_API_VERSION
    broker_id: ClassVar[str] = "fake"
    capabilities: ClassVar[frozenset[str]] = frozenset({"task-consumer"})
    stability: ClassVar[str] = "alpha"

    def validate_config(self, config, *, check_connectivity=False) -> None:
        del config, check_connectivity

    def create_runtime(self, config) -> _Runtime:
        del config
        return _Runtime()


def test_message_keeps_payload_opaque_and_metadata_restricted() -> None:
    payload = object()
    message = Message(
        payload,
        "delivery-1",
        metadata={"attempt": "1"},
    )

    assert message.payload is payload
    assert message.metadata == {"attempt": "1"}
    with pytest.raises(TypeError):
        operator.setitem(message.metadata, "attempt", "2")
    with pytest.raises(ValueError, match="string keys and values"):
        Message(
            {},
            "delivery-2",
            metadata=cast(Any, {"attempt": 1}),
        )


def test_task_outcome_is_not_a_workload_result_contract() -> None:
    outcome = TaskOutcome(
        TaskDisposition.RETRY,
        BrokerError(code="RAY_UNAVAILABLE", sanitized_message="retry later"),
    )

    assert outcome.error is not None
    assert outcome.error.code == "RAY_UNAVAILABLE"
    assert not hasattr(outcome, "result")
    assert not hasattr(broker_contract, "JobResult")
    assert not hasattr(broker_contract, "EventReporter")


def test_discovery_is_lazy_and_records_import_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda group: iter(
            [
                _EntryPoint("broken", ImportError("optional dependency unavailable")),
                _EntryPoint("fake", _Plugin),
            ]
            if group == "tributo.brokers"
            else []
        ),
    )
    diagnostics = []

    assert plugin.discover_broker_plugins(diagnostics) == [_Plugin]
    assert diagnostics[0].entry_point_name == "broken"
    assert diagnostics[0].error_type == "ImportError"
    assert "optional dependency unavailable" not in diagnostics[0].reason


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("capabilities", ("task-consumer",)),
        ("stability", "prototype"),
    ],
)
def test_discovery_rejects_invalid_provider_metadata(
    monkeypatch, attribute: str, value: object
) -> None:
    invalid = type("InvalidPlugin", (_Plugin,), {attribute: value})
    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda group: (
            iter([_EntryPoint("fake", invalid)])
            if group == "tributo.brokers"
            else iter(())
        ),
    )
    diagnostics = []

    assert plugin.discover_broker_plugins(diagnostics) == []
    assert attribute in diagnostics[0].reason


def test_discovery_rejects_version_and_entrypoint_identity_mismatch(
    monkeypatch,
) -> None:
    class _WrongVersion(_Plugin):
        api_version = BROKER_API_VERSION + 1

    class _WrongIdentity(_Plugin):
        broker_id = "other"

    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda group: iter(
            [
                _EntryPoint("wrong-version", _WrongVersion),
                _EntryPoint("fake", _WrongIdentity),
            ]
            if group == "tributo.brokers"
            else []
        ),
    )
    diagnostics = []

    assert plugin.discover_broker_plugins(diagnostics) == []
    assert "api_version" in diagnostics[0].reason
    assert "does not match broker_id" in diagnostics[1].reason


def test_explicit_disabled_provider_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("TRIBUTO_PLUGINS", "another")
    monkeypatch.setattr(
        plugin,
        "_iter_entry_points",
        lambda _group: iter([_EntryPoint("fake", _Plugin)]),
    )

    with pytest.raises(JobConfigurationError, match="disabled"):
        plugin.resolve_broker_plugin("fake")


def test_registry_reports_metadata_and_duplicate_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "tributo.integrations.broker_registry.discover_broker_plugins",
        lambda _diagnostics: [_Plugin, _Plugin],
    )
    registry = BrokerRegistry()

    descriptors = registry.list()

    assert descriptors[0].broker_id == "fake"
    assert descriptors[0].stability == "alpha"
    assert descriptors[0].capabilities == ("task-consumer",)
    assert registry.diagnostics()[0].reason == "Duplicate broker_id discovered"
