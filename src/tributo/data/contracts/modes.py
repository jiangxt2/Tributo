"""Engine-neutral write modes."""

from enum import Enum

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class WriteMode(Enum):
    """Compatibility-stable bounded write modes."""

    OVERWRITE = "overwrite"
    APPEND = "append"


__all__ = ["WriteMode"]
