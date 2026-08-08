"""Third-party ingestion Provider discovery and isolation tests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from tributo.data.provider import DataSourceProvider, ResolvedSource
from tributo.data.provider_plugins import (
    ProviderDescriptor,
    discover_provider_descriptors,
    register_discovered_providers,
)
from tributo.data.provider_registry import ProviderRegistry


def _provider(provider_id: str) -> type[DataSourceProvider]:
    class _PluginProvider(DataSourceProvider):
        def normalize(self, source: Any) -> ResolvedSource:
            return ResolvedSource(
                provider_id=self.provider_id,
                canonical_uri="plugin://source",
            )

    _PluginProvider.provider_id = provider_id
    return _PluginProvider


def _descriptor(provider_id: str) -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_class=_provider(provider_id),
        distribution_name="test-ingestion-provider",
        distribution_version="1.2.3",
        tributo_version_spec=">=0.1.0,<2",
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
    first = _descriptor("example.first")
    second = _descriptor("example.second")
    entry_points = (
        _EntryPoint("z-last", lambda: (second,)),
        _EntryPoint("a-first", first),
    )
    monkeypatch.setattr(
        "tributo.data.provider_plugins.importlib.metadata.entry_points",
        lambda **kwargs: entry_points,
    )

    assert discover_provider_descriptors() == (first, second)


def test_discovery_failure_does_not_expose_native_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "tributo.data.provider_plugins.importlib.metadata.entry_points",
        lambda **kwargs: (_EntryPoint("broken", RuntimeError("password=top-secret")),),
    )

    with caplog.at_level(logging.WARNING):
        assert discover_provider_descriptors() == ()

    assert "RuntimeError" in caplog.text
    assert "top-secret" not in caplog.text


def test_registration_validates_distribution_and_tributo_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor("example.hive")
    registry = ProviderRegistry()
    monkeypatch.setattr(
        "tributo.data.provider_plugins.discover_provider_descriptors",
        lambda: (descriptor,),
    )
    monkeypatch.setattr(
        "tributo.data.provider_plugins.importlib.metadata.version",
        lambda name: {"test-ingestion-provider": "1.2.3"}[name],
    )

    register_discovered_providers(registry)

    assert registry.list_providers() == ["example.hive"]


def test_plugin_registration_never_overrides_an_existing_provider(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptor = _descriptor("example.hive")
    registry = ProviderRegistry()
    registry.register(descriptor.provider_class)
    monkeypatch.setattr(
        "tributo.data.provider_plugins.discover_provider_descriptors",
        lambda: (descriptor,),
    )
    monkeypatch.setattr(
        "tributo.data.provider_plugins.importlib.metadata.version",
        lambda name: "1.2.3",
    )

    with caplog.at_level(logging.WARNING):
        register_discovered_providers(registry)

    assert registry.list_providers() == ["example.hive"]
    assert "JobConfigurationError" in caplog.text


def test_incompatible_plugin_is_isolated_from_other_providers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    compatible = _descriptor("example.parquet")
    incompatible = ProviderDescriptor(
        provider_class=_provider("example.orc"),
        distribution_name="test-ingestion-provider",
        distribution_version="1.2.3",
        tributo_version_spec="==999.0.0",
    )
    registry = ProviderRegistry()
    monkeypatch.setattr(
        "tributo.data.provider_plugins.discover_provider_descriptors",
        lambda: (incompatible, compatible),
    )
    monkeypatch.setattr(
        "tributo.data.provider_plugins.importlib.metadata.version",
        lambda name: "1.2.3",
    )

    with caplog.at_level(logging.WARNING):
        register_discovered_providers(registry)

    assert registry.list_providers() == ["example.parquet"]
    assert "EngineNotAvailableError" in caplog.text


@pytest.mark.parametrize("version_spec", ["", "   "])
def test_provider_descriptor_requires_explicit_tributo_version_range(
    version_spec: str,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ProviderDescriptor(
            provider_class=_provider("example.hive"),
            distribution_name="test-ingestion-provider",
            distribution_version="1.2.3",
            tributo_version_spec=version_spec,
        )


def test_registration_failure_does_not_log_untrusted_provider_id(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptor = ProviderDescriptor(
        provider_class=_provider("password=top-secret"),
        distribution_name="test-ingestion-provider",
        distribution_version="1.2.3",
        tributo_version_spec=">=0.1.0,<2",
    )
    registry = ProviderRegistry()
    monkeypatch.setattr(
        "tributo.data.provider_plugins.discover_provider_descriptors",
        lambda: (descriptor,),
    )
    monkeypatch.setattr(
        "tributo.data.provider_plugins.importlib.metadata.version",
        lambda name: "1.2.3",
    )

    with caplog.at_level(logging.WARNING):
        register_discovered_providers(registry)

    assert registry.list_providers() == []
    assert "TypeError" in caplog.text
    assert "top-secret" not in caplog.text
