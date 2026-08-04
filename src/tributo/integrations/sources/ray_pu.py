"""Ray PU Learning checkpoint → ExportSource provider."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from tributo.integrations.sources.ray_dnn import _open_torch_source
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class RayPUSourceProvider:
    """Resolve a PU Trainer checkpoint using the shared Torch adapter."""

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "ray-pu-v1"
    trainer_type: ClassVar[str] = "pu"
    priority: ClassVar[int] = 100

    def open_source(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Any:
        """Open a PU checkpoint as an ``ExportSource``."""
        return _open_torch_source(
            result,
            config,
            source_kind="pu_result",
            trainer_type="pu",
        )


__all__ = [
    "RayPUSourceProvider",
]
