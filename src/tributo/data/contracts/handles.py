"""Typed native data handles shared by bounded reads and writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import daft
    import ray.data


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class RayDataHandle:
    """A caller-owned Ray Dataset handle."""

    dataset: ray.data.Dataset = field(repr=False)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class DaftDataFrameHandle:
    """A caller-owned Daft DataFrame handle."""

    dataframe: daft.DataFrame = field(repr=False)


DataHandle: TypeAlias = RayDataHandle | DaftDataFrameHandle

__all__ = ["DaftDataFrameHandle", "DataHandle", "RayDataHandle"]
