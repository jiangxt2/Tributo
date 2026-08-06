"""Third-party ingestion Binding discovery tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from tributo.data.binding_plugins import (
    discover_binding_descriptors,
    register_discovered_bindings,
)
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingDescriptor,
    BindingKey,
    EngineBindings,
)
from tributo.data.ingestion import RayDataHandle
from tributo.data.scan_plan import ScanKind


class _PluginBinding:
    def compile(self, request: Any) -> BindingCompilation:
        return BindingCompilation(
            handle=RayDataHandle(object()),
            engine_version="2.55.1",
            reader_api="plugin.read",
            transport_id="plugin",
        )


def _descriptor(binding_id: str) -> BindingDescriptor:
    return BindingDescriptor(
        key=BindingKey("tributo.ray_data", ScanKind.FILE, "parquet", binding_id),
        factory=_PluginBinding,
        capabilities=frozenset(),
        distribution_name="test-plugin",
        distribution_version="1.0.0",
        engine_version_spec="==2.55.1",
    )


@dataclass
class _EntryPoint:
    name: str
    value: Any

    def load(self) -> Any:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_discovery_is_deterministic_and_accepts_descriptor_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _descriptor("plugin.first")
    second = _descriptor("plugin.second")
    entry_points = (
        _EntryPoint("z-last", lambda: (second,)),
        _EntryPoint("a-first", first),
    )
    monkeypatch.setattr(
        "tributo.data.binding_plugins.importlib.metadata.entry_points",
        lambda **kwargs: entry_points,
    )

    assert discover_binding_descriptors() == (first, second)


def test_discovery_failure_does_not_expose_native_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "tributo.data.binding_plugins.importlib.metadata.entry_points",
        lambda **kwargs: (_EntryPoint("broken", RuntimeError("password=top-secret")),),
    )

    with caplog.at_level(logging.WARNING):
        assert discover_binding_descriptors() == ()

    assert "RuntimeError" in caplog.text
    assert "top-secret" not in caplog.text


def test_plugin_registration_never_overrides_an_existing_binding(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptor = _descriptor("plugin.duplicate")
    bindings = EngineBindings()
    monkeypatch.setattr(
        "tributo.data.engine_binding.importlib.metadata.version",
        lambda name: {"ray": "2.55.1", "test-plugin": "1.0.0"}[name],
    )
    bindings.register(descriptor)
    monkeypatch.setattr(
        "tributo.data.binding_plugins.discover_binding_descriptors",
        lambda: (descriptor,),
    )

    with caplog.at_level(logging.WARNING):
        register_discovered_bindings(bindings)

    assert bindings.resolve(descriptor.key) is descriptor
    assert "already registered" not in caplog.text
    assert "JobConfigurationError" in caplog.text
