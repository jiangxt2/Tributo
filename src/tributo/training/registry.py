"""Trainer registry.

Follows the same pattern as ``embeddings/registry.py``: frozen dataclass
+ module-level dict + thread-safe lock.
"""

from __future__ import annotations

import threading

from tributo.exceptions import JobConfigurationError
from tributo.training.base import TrainerSpec
from tributo.util.annotations import PublicAPI

_TRAINER_REGISTRY: dict[str, TrainerSpec] = {}
_REGISTRY_LOCK = threading.Lock()


@PublicAPI(stability="beta")
def register(spec: TrainerSpec) -> None:
    """Register a trainer spec.

    Args:
        spec: The trainer spec to register.

    Raises:
        JobConfigurationError: If a trainer with the same name is already
            registered.
    """
    with _REGISTRY_LOCK:
        if spec.name in _TRAINER_REGISTRY:
            raise JobConfigurationError(
                f"Trainer '{spec.name}' already registered. "
                f"Available: {sorted(_TRAINER_REGISTRY)}"
            )
        _TRAINER_REGISTRY[spec.name] = spec


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
    with _REGISTRY_LOCK:
        if name not in _TRAINER_REGISTRY:
            raise JobConfigurationError(
                f"Unknown trainer: '{name}'. Available: {sorted(_TRAINER_REGISTRY)}"
            )
        return _TRAINER_REGISTRY[name]


@PublicAPI(stability="beta")
def list_trainers() -> list[str]:
    """Return the names of all registered trainers."""
    with _REGISTRY_LOCK:
        return sorted(_TRAINER_REGISTRY)
