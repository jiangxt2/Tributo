"""Default registry and Gateway for Tributo's native writer bindings."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import threading
from typing import Any

from tributo.data.writing.gateway import WriteGateway
from tributo.data.writing.native_bindings import (
    DaftCsvWriteBinding,
    DaftIcebergWriteBinding,
    DaftLanceWriteBinding,
    DaftParquetWriteBinding,
    RayCsvWriteBinding,
    RayIcebergWriteBinding,
    RayLanceWriteBinding,
    RayParquetWriteBinding,
)
from tributo.data.writing.plugins import register_discovered_write_bindings
from tributo.data.writing.registry import WriteBindingRegistry
from tributo.data.writing.target_registry import WriteTargetRegistry
from tributo.util.annotations import PublicAPI

_lock = threading.RLock()
_default_gateway: WriteGateway | None = None
logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
def default_write_gateway() -> WriteGateway:
    """Return the process-local Gateway with available native bindings."""
    global _default_gateway
    with _lock:
        if _default_gateway is None:
            registry = WriteBindingRegistry()
            _register_if_available(registry, RayParquetWriteBinding)
            _register_if_available(registry, RayCsvWriteBinding)
            _register_if_available(registry, RayIcebergWriteBinding)
            _register_if_available(registry, RayLanceWriteBinding)
            _register_if_available(registry, DaftParquetWriteBinding)
            _register_if_available(registry, DaftCsvWriteBinding)
            _register_if_available(registry, DaftIcebergWriteBinding)
            _register_if_available(registry, DaftLanceWriteBinding)
            register_discovered_write_bindings(registry)
            _default_gateway = WriteGateway(
                registry, target_registry=WriteTargetRegistry()
            )
        return _default_gateway


def _register_if_available(registry: WriteBindingRegistry, binding_type: Any) -> None:
    """Register an optional-format binding only when its dependency is present."""
    descriptor = binding_type._descriptor
    try:
        availability_check = getattr(binding_type, "is_available", None)
        if callable(availability_check) and not availability_check():
            return
        engine_distribution = (
            "ray" if descriptor.engine_id == "tributo.ray_data" else "daft"
        )
        if not _distribution_available(engine_distribution):
            return
        if any(
            not _distribution_available(name)
            for name in descriptor.dependency_distributions
        ):
            return
        registry.register(descriptor, binding_type)
    except Exception as exc:
        # Built-ins are optional capabilities.  One stale engine or binding
        # version must not prevent unrelated native writers or plugins from
        # being registered; resolution will report the missing capability.
        logger.warning(
            "Skipping built-in write binding %r after %s",
            descriptor.binding_id,
            type(exc).__name__,
        )


def _distribution_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


__all__ = ["default_write_gateway"]
