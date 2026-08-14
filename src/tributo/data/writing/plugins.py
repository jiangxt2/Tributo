"""Independent discovery for installed ``tributo.write_bindings`` plugins."""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Iterable
from typing import Any

from tributo.data.writing.contracts import WriteDescriptor
from tributo.data.writing.registry import WriteBindingRegistry
from tributo.util.annotations import DeveloperAPI

_ENTRY_POINT_GROUP = "tributo.write_bindings"
logger = logging.getLogger(__name__)


def _as_descriptors(value: Any) -> tuple[tuple[WriteDescriptor, Any], ...]:
    loaded = value() if callable(value) else value
    if isinstance(loaded, tuple) and len(loaded) == 2:
        loaded = (loaded,)
    if isinstance(loaded, (str, bytes)) or not isinstance(loaded, Iterable):
        raise TypeError(
            "write binding entry point must export descriptor/factory pairs"
        )
    pairs = tuple(loaded)
    if not pairs or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not isinstance(pair[0], WriteDescriptor)
        or not callable(pair[1])
        for pair in pairs
    ):
        raise TypeError(
            "write binding entry point has invalid descriptor/factory pairs"
        )
    return pairs


@DeveloperAPI
def discover_write_bindings() -> tuple[tuple[WriteDescriptor, Any], ...]:
    """Discover valid writing plugins deterministically and fail closed."""
    discovered: list[tuple[WriteDescriptor, Any]] = []
    for entry_point in sorted(
        importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP),
        key=lambda item: item.name,
    ):
        try:
            discovered.extend(_as_descriptors(entry_point.load()))
        except Exception as exc:
            logger.warning(
                "Skipping write binding entry point %r after %s",
                entry_point.name,
                type(exc).__name__,
            )
    return tuple(discovered)


@DeveloperAPI
def register_discovered_write_bindings(registry: WriteBindingRegistry) -> None:
    """Register valid writing plugins without replacing existing bindings."""
    for descriptor, factory in discover_write_bindings():
        try:
            registry.register(descriptor, factory)
        except Exception as exc:
            logger.warning(
                "Skipping write binding %r after %s",
                descriptor.binding_id,
                type(exc).__name__,
            )


__all__ = ["discover_write_bindings", "register_discovered_write_bindings"]
