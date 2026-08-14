"""Independent ``tributo.write_bindings`` discovery tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tributo.data.contracts.modes import WriteMode
from tributo.data.writing import WriteCapability, WriteDescriptor
from tributo.data.writing.plugins import (
    discover_write_bindings,
    register_discovered_write_bindings,
)
from tributo.data.writing.registry import WriteBindingRegistry


def _descriptor(binding_id: str) -> WriteDescriptor:
    return WriteDescriptor(
        engine_id="ray",
        target_kind="parquet",
        binding_id=binding_id,
        engine_version_spec="==2.55.1",
        binding_distribution="test-plugin",
        binding_distribution_version="1.0.0",
        capabilities=WriteCapability(supported_modes=frozenset({WriteMode.OVERWRITE})),
    )


@dataclass
class _EntryPoint:
    name: str
    value: Any

    def load(self) -> Any:
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_discovery_uses_independent_group_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (_descriptor("plugin.first"), lambda: object())
    second = (_descriptor("plugin.second"), lambda: object())
    calls: list[str] = []

    def entry_points(**kwargs: Any) -> tuple[_EntryPoint, ...]:
        calls.append(kwargs["group"])
        return (
            _EntryPoint("z-last", second),
            _EntryPoint("a-first", first),
        )

    monkeypatch.setattr(
        "tributo.data.writing.plugins.importlib.metadata.entry_points", entry_points
    )

    discovered = discover_write_bindings()

    assert calls == ["tributo.write_bindings"]
    assert [item[0].binding_id for item in discovered] == [
        "plugin.first",
        "plugin.second",
    ]


def test_discovery_hides_plugin_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "tributo.data.writing.plugins.importlib.metadata.entry_points",
        lambda **kwargs: (
            _EntryPoint("broken", RuntimeError("password=plugin-secret")),
        ),
    )

    with caplog.at_level("WARNING"):
        assert discover_write_bindings() == ()

    assert "RuntimeError" in caplog.text
    assert "plugin-secret" not in caplog.text


def test_registration_skips_duplicate_plugins(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptor = _descriptor("plugin.one")
    monkeypatch.setattr(
        "tributo.data.writing.plugins.discover_write_bindings",
        lambda: ((descriptor, lambda: object()),),
    )
    monkeypatch.setattr(
        "tributo.data.writing.registry.importlib.metadata.version",
        lambda name: {"ray": "2.55.1", "test-plugin": "1.0.0"}[name],
    )
    registry = WriteBindingRegistry()
    registry.register(descriptor, lambda: object())

    with caplog.at_level("WARNING"):
        register_discovered_write_bindings(registry)

    assert "plugin.one" in caplog.text
    assert "WriteCapabilityError" in caplog.text
