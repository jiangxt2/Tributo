"""Single Ray-only compatibility adapter over the ingestion Gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tributo.data.ingestion import IngestionRequest, RayDataHandle, open_ingestion
from tributo.data.source_config import CanonicalSourceInput
from tributo.exceptions import JobConfigurationError

if TYPE_CHECKING:
    import ray.data


def open_ray_compat(source: CanonicalSourceInput) -> "ray.data.Dataset":
    """Return a Ray Dataset without maintaining a second Reader path."""
    result = open_ingestion(IngestionRequest(source=source, engine="ray"))
    try:
        if not isinstance(result.handle, RayDataHandle):
            raise JobConfigurationError(
                "Ray compatibility adapter received a non-Ray ingestion handle"
            )
        return result.handle.dataset
    finally:
        result.close()
