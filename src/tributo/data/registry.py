"""DataConnector registry.

Delegates to the generic ``Registry`` base class in ``_common/registry.py``.
"""

from __future__ import annotations

from tributo._common.registry import Registry
from tributo.data.base import DataConnector
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

_registry: Registry[str, type[DataConnector]] = Registry(name="connector")

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
    _registry.register(name, cls)


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
    try:
        cls = _registry.get(name)
    except JobConfigurationError:
        if name in _OPTIONAL_CONNECTORS:
            raise JobConfigurationError(
                f"Connector '{name}' requires optional dependencies. "
                f"Install with: {_OPTIONAL_CONNECTORS[name]}"
            ) from None
        raise
    return cls()


@PublicAPI(stability="alpha")
def list_connectors() -> list[str]:
    """Return the names of all registered connectors."""
    return _registry.list()
