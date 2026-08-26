"""Generic compatibility descriptor for third-party legacy Trainers.

Tributo Core no longer registers first-party Trainer implementations. This
type remains only so an independently published Wheel can expose a bounded
compatibility adapter during its own migration window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from tributo.algorithms.api import (
    AlgorithmRegistration,
    ExecutionMode,
    QualifiedReference,
)
from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class LegacyTrainerDescriptor:
    """Bind one portable contract to externally owned legacy Trainer refs."""

    registration: AlgorithmRegistration
    trainer_ref: QualifiedReference
    config_model_ref: QualifiedReference
    limitations: tuple[str, ...]
    stability: Literal["beta"] = "beta"
    tested: bool = False
    supported: bool = False
    native_migration_complete: bool = False

    def __post_init__(self) -> None:
        if self.stability != "beta":
            raise ValueError("legacy Trainer descriptors must remain Beta")
        if self.registration.implementation.implementation_ref != self.trainer_ref:
            raise ValueError("legacy descriptor Trainer references must agree")
        if (
            self.registration.implementation.execution_mode
            is not ExecutionMode.LEGACY_TRAINER
        ):
            raise ValueError("legacy descriptor requires execution_mode=legacy_trainer")
        if self.supported and not self.tested:
            raise ValueError("supported legacy descriptors must also be tested")
        if self.native_migration_complete:
            raise ValueError(
                "LegacyTrainerDescriptor cannot represent a completed migration"
            )
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def name(self) -> str:
        """Return the canonical algorithm identity."""
        return self.registration.spec.name


def build_legacy_spec(
    descriptor: LegacyTrainerDescriptor,
    *,
    trainer_cls: type,
    config_model: type,
) -> AlgorithmSpec:
    """Hydrate the Beta TrainerSpec view without duplicating algorithm facts."""
    return replace(
        descriptor.registration.spec,
        trainer_cls=trainer_cls,
        config_model=config_model,
    )


BUILTIN_LEGACY_DESCRIPTORS: tuple[LegacyTrainerDescriptor, ...] = ()

__all__ = [
    "LegacyTrainerDescriptor",
]
