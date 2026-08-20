"""Lazy discovery and explicit resolution for broker providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import PluginLoadDiagnostic
from tributo.integrations.broker import BrokerPlugin
from tributo.plugin import discover_broker_plugins, resolve_broker_plugin
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class BrokerDescriptor:
    """Side-effect-free metadata reported for one installed provider."""

    broker_id: str
    api_version: int
    capabilities: tuple[str, ...]
    stability: str


@PublicAPI(stability="alpha")
class BrokerRegistry:
    """Resolve only explicitly selected providers, failing closed on errors."""

    def __init__(self) -> None:
        self._diagnostics: list[PluginLoadDiagnostic] = []

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        """Return non-fatal diagnostics from the most recent listing."""
        return tuple(self._diagnostics)

    def list(self) -> tuple[BrokerDescriptor, ...]:
        """List providers without constructing them or connecting to a broker."""
        self._diagnostics.clear()
        descriptors: list[BrokerDescriptor] = []
        seen: set[str] = set()
        for cls in discover_broker_plugins(self._diagnostics):
            broker_id = cls.broker_id
            if broker_id in seen:
                self._diagnostics.append(
                    PluginLoadDiagnostic(
                        group="tributo.brokers",
                        entry_point_name=broker_id,
                        reason="Duplicate broker_id discovered",
                    )
                )
                continue
            seen.add(broker_id)
            descriptors.append(
                BrokerDescriptor(
                    broker_id=broker_id,
                    api_version=cls.api_version,
                    capabilities=tuple(sorted(cls.capabilities)),
                    stability=cls.stability,
                )
            )
        return tuple(descriptors)

    def resolve(self, broker_id: str) -> BrokerPlugin:
        """Load and instantiate one explicitly selected provider."""
        cls = resolve_broker_plugin(broker_id)
        try:
            plugin = cls()
        except Exception as exc:
            raise JobConfigurationError(
                f"Failed to initialize broker {broker_id!r} ({type(exc).__name__})"
            ) from exc
        return cast(BrokerPlugin, plugin)

    def validate(
        self,
        broker_id: str,
        config: Mapping[str, Any],
        *,
        check_connectivity: bool = False,
    ) -> BrokerPlugin:
        """Resolve a provider and delegate its config validation."""
        plugin = self.resolve(broker_id)
        try:
            plugin.validate_config(
                config,
                check_connectivity=check_connectivity,
            )
        except JobConfigurationError:
            raise
        except Exception as exc:
            raise JobConfigurationError(
                f"Broker {broker_id!r} rejected configuration ({type(exc).__name__})"
            ) from exc
        return plugin


__all__ = ["BrokerDescriptor", "BrokerRegistry"]
