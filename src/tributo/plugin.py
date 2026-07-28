"""Third-party plugin discovery via ``importlib.metadata`` entry points.

Tributo discovers external plugins through standard Python entry points
declared in ``pyproject.toml`` under the ``[project.entry-points]`` table.

Discovery groups:
    ``tributo.trainers``
        Each entry point must point to a module with a ``trainer_spec``
        attribute of type ``TrainerSpec``.  Example::

            [project.entry-points."tributo.trainers"]
            my_algo = "my_package.trainer"

    ``tributo.connectors``
        Each entry point must point to a ``DataConnector`` subclass.
        Example::

            [project.entry-points."tributo.connectors"]
            my_db = "my_package.connector:MyDBConnector"

    ``tributo.models``
        Each entry point must point to a module with a ``model_specs``
        attribute (a list of ``ModelSpec``).  Example::

            [project.entry-points."tributo.models"]
            my_models = "my_package.embedding_models"

Filtering:
    Set the ``TRIBUTO_PLUGINS`` environment variable to a comma-separated
    list of entry point names to load only those plugins.  If unset, all
    discovered plugins are loaded.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)


def _get_enabled_plugins() -> set[str] | None:
    """Parse ``TRIBUTO_PLUGINS`` env var.  Returns ``None`` if unset (load all)."""
    raw = os.environ.get("TRIBUTO_PLUGINS", "").strip()
    if not raw:
        return None
    return {name.strip() for name in raw.split(",") if name.strip()}


def discover_trainer_plugins() -> list[Any]:
    """Discover third-party trainers registered via entry_points.

    Each entry point is expected to point to a module whose top-level
    ``trainer_spec`` attribute is a :class:`TrainerSpec` instance.
    """
    from tributo.training.base import TrainerSpec

    enabled = _get_enabled_plugins()
    specs: list[Any] = []

    for ep in _iter_entry_points("tributo.trainers"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping trainer plugin %r (not in TRIBUTO_PLUGINS)", ep.name)
            continue
        try:
            mod = ep.load()
        except Exception:
            logger.warning(
                "Failed to load trainer plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue

        spec = getattr(mod, "trainer_spec", None)
        if not isinstance(spec, TrainerSpec):
            logger.warning(
                "Trainer plugin %r does not export a TrainerSpec instance as "
                "%r.trainer_spec (got %r); skipping.",
                ep.name,
                ep.value,
                type(spec).__name__,
            )
            continue
        specs.append(spec)
        logger.info("Discovered trainer plugin %r (%s)", ep.name, ep.value)

    return specs


def discover_connector_plugins() -> list[type[Any]]:
    """Discover third-party data connectors registered via entry_points.

    Each entry point is expected to point to a ``DataConnector`` subclass.
    """
    from tributo.data.base import DataConnector

    enabled = _get_enabled_plugins()
    classes: list[type[Any]] = []

    for ep in _iter_entry_points("tributo.connectors"):
        if enabled is not None and ep.name not in enabled:
            logger.debug(
                "Skipping connector plugin %r (not in TRIBUTO_PLUGINS)", ep.name
            )
            continue
        try:
            cls = ep.load()
        except Exception:
            logger.warning(
                "Failed to load connector plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue

        if not (isinstance(cls, type) and issubclass(cls, DataConnector)):
            logger.warning(
                "Connector plugin %r is not a DataConnector subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            continue
        classes.append(cls)
        logger.info("Discovered connector plugin %r (%s)", ep.name, ep.value)

    return classes


def discover_model_plugins() -> list[Any]:
    """Discover third-party embedding models registered via entry_points.

    Each entry point is expected to point to a module whose top-level
    ``model_specs`` attribute is a list of :class:`ModelSpec` instances.
    """
    from tributo.embeddings.registry import ModelSpec

    enabled = _get_enabled_plugins()
    specs: list[Any] = []

    for ep in _iter_entry_points("tributo.models"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping model plugin %r (not in TRIBUTO_PLUGINS)", ep.name)
            continue
        try:
            mod = ep.load()
        except Exception:
            logger.warning(
                "Failed to load model plugin %r (%s)", ep.name, ep.value, exc_info=True
            )
            continue

        model_specs = getattr(mod, "model_specs", None)
        if not isinstance(model_specs, list):
            logger.warning(
                "Model plugin %r does not export a list of ModelSpec as "
                "%r.model_specs (got %r); skipping.",
                ep.name,
                ep.value,
                type(model_specs).__name__,
            )
            continue

        for spec in model_specs:
            if not isinstance(spec, ModelSpec):
                logger.warning(
                    "Model plugin %r contained non-ModelSpec item %r; skipping item.",
                    ep.name,
                    spec,
                )
                continue
            specs.append(spec)

        logger.info(
            "Discovered model plugin %r (%s) — %d model(s)",
            ep.name,
            ep.value,
            len(model_specs),
        )

    return specs


def _iter_entry_points(group: str) -> Any:
    """Iterate over entry points for *group*."""
    eps = entry_points(group=group)
    yield from eps
