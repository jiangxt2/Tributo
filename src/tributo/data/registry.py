"""DataConnector registry.

Follows the same pattern as ``embeddings/registry.py``: module-level
dict with write-path locking.
"""

from __future__ import annotations

import threading

from tributo.data.base import DataConnector
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

_CONNECTOR_REGISTRY: dict[str, type[DataConnector]] = {}
_REGISTRY_LOCK = threading.Lock()

# Known optional connectors with install hints.
_OPTIONAL_CONNECTORS: dict[str, str] = {
    "iceberg": "uv sync --extra data",
    "lance": "uv sync --extra data",
}


@PublicAPI(stability="alpha")
def register_connector(name: str, cls: type[DataConnector]) -> None:
    """Register a data connector.

    Args:
        name: Short connector name (e.g. ``"parquet"``, ``"lance"``).
        cls: ``DataConnector`` subclass.

    Raises:
        JobConfigurationError: If a connector with *name* is already
            registered.
    """
    with _REGISTRY_LOCK:
        if name in _CONNECTOR_REGISTRY:
            raise JobConfigurationError(
                f"Connector '{name}' already registered. "
                f"Available: {sorted(_CONNECTOR_REGISTRY)}"
            )
        _CONNECTOR_REGISTRY[name] = cls


@PublicAPI(stability="alpha")
def get_connector(name: str) -> DataConnector:
    """Return an instance of a registered data connector.

    Args:
        name: Short connector name.

    Returns:
        A ``DataConnector`` instance.

    Raises:
        JobConfigurationError: If the connector is not registered or
            requires optional dependencies.
    """
    with _REGISTRY_LOCK:
        if name not in _CONNECTOR_REGISTRY:
            if name in _OPTIONAL_CONNECTORS:
                raise JobConfigurationError(
                    f"Connector '{name}' requires optional dependencies. "
                    f"Install with: {_OPTIONAL_CONNECTORS[name]}"
                )
            raise JobConfigurationError(
                f"Unknown connector: '{name}'. Available: {sorted(_CONNECTOR_REGISTRY)}"
            )
        return _CONNECTOR_REGISTRY[name]()


@PublicAPI(stability="alpha")
def list_connectors() -> list[str]:
    """Return the names of all registered connectors."""
    with _REGISTRY_LOCK:
        return sorted(_CONNECTOR_REGISTRY)
