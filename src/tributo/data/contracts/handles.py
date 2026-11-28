"""Typed native data handles shared by bounded reads and writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class RayDataHandle:
    """A caller-owned Ray Dataset handle."""

    dataset: ray.data.Dataset = field(repr=False)


DataHandle: TypeAlias = RayDataHandle

__all__ = ["DataHandle", "RayDataHandle"]
