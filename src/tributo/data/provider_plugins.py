"""Versioned discovery for third-party bounded-ingestion Providers."""

from __future__ import annotations

import importlib.metadata
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from tributo import __version__ as tributo_version
from tributo.data.provider import DataSourceProvider
from tributo.data.provider_registry import ProviderRegistry
from tributo.exceptions import EngineNotAvailableError
from tributo.util.annotations import DeveloperAPI

_ENTRY_POINT_GROUP = "tributo.ingestion_providers"
_DISTRIBUTION_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

logger = logging.getLogger(__name__)


@DeveloperAPI
@dataclass(frozen=True)
class ProviderDescriptor:
    """Metadata exported by an installed logical Provider plugin."""

    provider_class: type[DataSourceProvider]
    distribution_name: str
    distribution_version: str
    tributo_version_spec: str
    api_version: Literal[1] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.provider_class, type) or not issubclass(
            self.provider_class, DataSourceProvider
        ):
            raise TypeError(
                "ProviderDescriptor.provider_class must be a "
                "DataSourceProvider subclass"
            )
        if (
            not isinstance(self.distribution_name, str)
            or _DISTRIBUTION_NAME_RE.fullmatch(self.distribution_name) is None
        ):
            raise ValueError("Provider distribution_name must be an identifier")
        try:
            Version(self.distribution_version)
        except (InvalidVersion, TypeError) as exc:
            raise ValueError("Provider distribution_version must be valid") from exc
        try:
            SpecifierSet(self.tributo_version_spec)
        except (InvalidSpecifier, TypeError) as exc:
            raise ValueError("Provider tributo_version_spec must be valid") from exc
        if not self.tributo_version_spec.strip():
            raise ValueError("Provider tributo_version_spec must not be empty")
        if self.api_version != 1:
            raise ValueError("Provider api_version must be 1")


def _as_descriptors(value: Any) -> tuple[ProviderDescriptor, ...]:
    loaded = (
        value()
        if callable(value) and not isinstance(value, ProviderDescriptor)
        else value
    )
    if isinstance(loaded, ProviderDescriptor):
        return (loaded,)
    if isinstance(loaded, (str, bytes)) or not isinstance(loaded, Iterable):
        raise TypeError("Provider entry point must export descriptors")
    descriptors = tuple(loaded)
    if not descriptors or any(
        not isinstance(descriptor, ProviderDescriptor) for descriptor in descriptors
    ):
        raise TypeError("Provider entry point must export ProviderDescriptor values")
    return descriptors


def _validate_installed_versions(descriptor: ProviderDescriptor) -> None:
    try:
        installed_distribution = importlib.metadata.version(
            descriptor.distribution_name
        )
    except importlib.metadata.PackageNotFoundError as exc:
        raise EngineNotAvailableError(
            f"Provider distribution {descriptor.distribution_name!r} is not installed"
        ) from exc
    try:
        distribution_matches = Version(installed_distribution) == Version(
            descriptor.distribution_version
        )
        tributo_matches = Version(tributo_version) in SpecifierSet(
            descriptor.tributo_version_spec
        )
    except InvalidVersion as exc:
        raise EngineNotAvailableError(
            "Provider or Tributo distribution version is invalid"
        ) from exc
    if not distribution_matches:
        raise EngineNotAvailableError(
            f"Provider distribution {descriptor.distribution_name!r} declares "
            f"version {descriptor.distribution_version}, "
            f"installed {installed_distribution}"
        )
    if not tributo_matches:
        raise EngineNotAvailableError(
            f"Provider distribution {descriptor.distribution_name!r} requires "
            f"Tributo {descriptor.tributo_version_spec}, installed {tributo_version}"
        )


@DeveloperAPI
def discover_provider_descriptors() -> tuple[ProviderDescriptor, ...]:
    """Load Provider descriptors deterministically and isolate bad plugins."""
    discovered: list[ProviderDescriptor] = []
    entry_points = sorted(
        importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP),
        key=lambda entry_point: entry_point.name,
    )
    for entry_point in entry_points:
        failure_type: str | None = None
        try:
            discovered.extend(_as_descriptors(entry_point.load()))
        except Exception as exc:
            failure_type = type(exc).__name__
        if failure_type is not None:
            logger.warning(
                "Skipping ingestion Provider entry point %r after %s",
                entry_point.name,
                failure_type,
            )
    return tuple(discovered)


@DeveloperAPI
def register_discovered_providers(registry: ProviderRegistry) -> None:
    """Register installed Providers without replacing an existing route."""
    for descriptor in discover_provider_descriptors():
        failure_type: str | None = None
        try:
            _validate_installed_versions(descriptor)
            registry.register(descriptor.provider_class)
        except Exception as exc:
            failure_type = type(exc).__name__
        if failure_type is not None:
            logger.warning(
                "Skipping ingestion Provider from distribution %r after %s",
                descriptor.distribution_name,
                failure_type,
            )


__all__ = [
    "ProviderDescriptor",
    "discover_provider_descriptors",
    "register_discovered_providers",
]
