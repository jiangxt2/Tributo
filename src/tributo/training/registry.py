"""Trainer registry.

Delegates to the generic ``Registry`` base class in ``_common/registry.py``.
"""

from __future__ import annotations

from tributo._common.registry import Registry
from tributo.training.algorithm_spec import ExecutionKind
from tributo.training.base import BaseTrainer, TrainerSpec
from tributo.util.annotations import PublicAPI

_registry: Registry[str, TrainerSpec] = Registry(name="trainer")


def _validate_spec(spec: TrainerSpec) -> None:
    """Validate that *spec*'s capability declarations match its trainer class.

    Runs before registration so an inconsistent spec never enters the
    registry.  ``TypeError`` (rather than ``JobConfigurationError``)
    keeps the failure distinguishable from duplicate registration, which
    plugin discovery intentionally swallows.

    Raises:
        TypeError: *spec* violates a capability invariant:

            - ``trainer_cls`` is not a class or not a ``BaseTrainer``
              subclass.
            - ``execution_kind == ESTIMATE`` but ``trainer_cls`` is not a
              ``BaseCausalEstimator`` subclass.
            - ``execution_kind != ESTIMATE`` but ``trainer_cls`` is a
              ``BaseCausalEstimator`` subclass — causal estimators only
              run the ESTIMATE lifecycle.
    """
    trainer_cls = spec.trainer_cls
    if not isinstance(trainer_cls, type):
        raise TypeError(
            f"Algorithm {spec.name!r}: trainer_cls must be a class, "
            f"got {type(trainer_cls).__name__!r}"
        )
    if not issubclass(trainer_cls, BaseTrainer):
        raise TypeError(
            f"Algorithm {spec.name!r}: trainer_cls {trainer_cls.__name__} "
            f"must be a BaseTrainer subclass."
        )
    # Lazy import: causal_estimator is an optional module (requires
    # dowhy/econml); the registry must not force-load it.
    from tributo.training.causal_estimator import BaseCausalEstimator

    is_causal = issubclass(trainer_cls, BaseCausalEstimator)
    if spec.execution_kind == ExecutionKind.ESTIMATE and not is_causal:
        raise TypeError(
            f"Algorithm {spec.name!r}: execution_kind=ESTIMATE requires "
            f"a BaseCausalEstimator subclass, got {trainer_cls.__name__}."
        )
    if spec.execution_kind != ExecutionKind.ESTIMATE and is_causal:
        raise TypeError(
            f"Algorithm {spec.name!r}: execution_kind={spec.execution_kind} "
            f"cannot use BaseCausalEstimator subclass "
            f"{trainer_cls.__name__}; causal estimators run the ESTIMATE "
            f"lifecycle."
        )


@PublicAPI(stability="beta")
def register(spec: TrainerSpec) -> None:
    """Register a trainer spec.

    Args:
        spec: The trainer spec to register.

    Raises:
        TypeError: *spec*'s capability declarations are inconsistent with
            its ``trainer_cls``.
        JobConfigurationError: If a trainer with the same name is already
            registered.
    """
    _validate_spec(spec)
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
