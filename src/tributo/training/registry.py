"""Trainer registry.

Delegates to the generic ``Registry`` base class in ``_common/registry.py``.
"""

from __future__ import annotations

from tributo._common.registry import Registry
from tributo.training.base import TrainerSpec
from tributo.util.annotations import PublicAPI

_registry: Registry[str, TrainerSpec] = Registry(name="trainer")


@PublicAPI(stability="beta")
def register(spec: TrainerSpec) -> None:
    """Register a trainer spec.

    Args:
        spec: The trainer spec to register.

    Raises:
        JobConfigurationError: If a trainer with the same name is already
            registered.
    """
    _registry.register(spec.name, spec)


@PublicAPI(stability="beta")
def get_trainer(name: str) -> TrainerSpec:
    """Return a registered trainer spec by name.

    Args:
        name: Short trainer name (e.g. ``"xgboost"``).

    Returns:
        The corresponding ``TrainerSpec``.

    Raises:
        JobConfigurationError: If the trainer is not registered.
    """
    return _registry.get(name)


@PublicAPI(stability="beta")
def list_trainers() -> list[str]:
    """Return the names of all registered trainers."""
    return _registry.list()
