"""Lazy discovery for third-party ingestion Binding descriptors."""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Iterable
from typing import Any

from tributo.data.engine_binding import BindingDescriptor, EngineBindings
from tributo.util.annotations import DeveloperAPI

_ENTRY_POINT_GROUP = "tributo.ingestion_bindings"

logger = logging.getLogger(__name__)


def _as_descriptors(value: Any) -> tuple[BindingDescriptor, ...]:
    loaded = value() if callable(value) else value
    if isinstance(loaded, BindingDescriptor):
        return (loaded,)
    if isinstance(loaded, (str, bytes)) or not isinstance(loaded, Iterable):
        raise TypeError("Binding entry point must export descriptors")
    descriptors = tuple(loaded)
    if not descriptors or any(
        not isinstance(descriptor, BindingDescriptor) for descriptor in descriptors
    ):
        raise TypeError("Binding entry point must export BindingDescriptor values")
    return descriptors


@DeveloperAPI
def discover_binding_descriptors() -> tuple[BindingDescriptor, ...]:
    """Load valid descriptors deterministically without exposing native errors."""
    discovered: list[BindingDescriptor] = []
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
                "Skipping ingestion Binding entry point %r after %s",
                entry_point.name,
                failure_type,
            )
    return tuple(discovered)


@DeveloperAPI
def register_discovered_bindings(bindings: EngineBindings) -> None:
    """Register discovered descriptors without replacing an existing Binding."""
    for descriptor in discover_binding_descriptors():
        failure_type: str | None = None
        try:
            bindings.register(descriptor)
        except Exception as exc:
            failure_type = type(exc).__name__
        if failure_type is not None:
            logger.warning(
                "Skipping ingestion Binding %r after %s",
                descriptor.key.binding_id,
                failure_type,
            )


__all__ = ["discover_binding_descriptors", "register_discovered_bindings"]
