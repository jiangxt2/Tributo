"""Lazy registry and worker-side reconstruction helpers for brokers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import PluginLoadDiagnostic
from tributo.integrations.broker import (
    BrokerPlugin,
    CancellationChecker,
    CancellationSpec,
)
from tributo.plugin import discover_broker_plugins, resolve_broker_plugin
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokerDescriptor:
    """Metadata exposed by ``tributo broker list`` without instantiation."""

    broker_id: str
    api_version: int
    capabilities: tuple[str, ...]


@PublicAPI(stability="beta")
class BrokerRegistry:
    """Resolve explicitly selected provider plugins on demand."""

    def __init__(self) -> None:
        self._diagnostics: list[PluginLoadDiagnostic] = []

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        """Return non-fatal discovery diagnostics from the last listing."""
        return tuple(self._diagnostics)

    def list(self) -> tuple[BrokerDescriptor, ...]:
        """List discoverable brokers without constructing or connecting them."""
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
            capabilities = getattr(cls, "capabilities", frozenset())
            descriptors.append(
                BrokerDescriptor(
                    broker_id=broker_id,
                    api_version=cls.api_version,
                    capabilities=tuple(sorted(str(value) for value in capabilities)),
                )
            )
        return tuple(descriptors)

    def resolve(self, broker_id: str) -> BrokerPlugin:
        """Load one explicitly selected provider, failing closed on errors."""
        cls = resolve_broker_plugin(broker_id)
        try:
            plugin = cls()
        except Exception as exc:
            raise JobConfigurationError(
                f"Failed to initialize broker {broker_id!r} ({type(exc).__name__})"
            ) from exc
        return plugin

    def validate(
        self,
        broker_id: str,
        config: Mapping[str, Any],
        *,
        check_connectivity: bool = False,
    ) -> BrokerPlugin:
        """Resolve and delegate provider-owned config validation."""
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


def rebuild_cancellation_checker(
    value: Mapping[str, Any] | CancellationSpec | None,
) -> CancellationChecker | None:
    """Rebuild a checker in a Ray worker from JSON-safe config.

    This helper is intentionally fail-open: a missing or unavailable broker
    must not change ordinary training into a failed training run.  The error
    is logged with the broker and job identity, while secrets remain in the
    provider-owned config boundary.
    """
    if value is None:
        return None
    try:
        spec = (
            value
            if isinstance(value, CancellationSpec)
            else CancellationSpec.from_mapping(value)
        )
        plugin = BrokerRegistry().resolve(spec.broker_id)
        return plugin.create_cancellation_checker(spec)
    except Exception:
        broker_id = getattr(value, "broker_id", None)
        if isinstance(value, Mapping):
            broker_id = value.get("broker_id", broker_id)
        job_id = getattr(value, "job_id", None)
        if isinstance(value, Mapping):
            job_id = value.get("job_id", job_id)
        logger.warning(
            "Unable to rebuild cancellation checker: broker=%s job_id=%s",
            broker_id,
            job_id,
            exc_info=True,
        )
        return None
