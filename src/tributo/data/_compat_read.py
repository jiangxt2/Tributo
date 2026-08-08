"""Single Ray-only compatibility adapter over the ingestion Gateway."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from tributo.data._source_paths import resolve_file_source_path
from tributo.data.ingestion import IngestionRequest, RayDataHandle, open_ingestion
from tributo.data.provider import DataSourceProvider
from tributo.data.provider_registry import resolve_provider
from tributo.data.source_config import CanonicalSourceInput
from tributo.exceptions import JobConfigurationError

if TYPE_CHECKING:
    import ray.data


def open_ray_compat(
    source: CanonicalSourceInput,
    *,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Return a Ray Dataset through the Gateway or deprecated Provider SPI."""
    resolved_input = resolve_file_source_path(source, project_root_path)
    provider = resolve_provider(resolved_input)
    if type(provider).plan is DataSourceProvider.plan:
        warnings.warn(
            f"Provider {provider.provider_id!r} implements the deprecated "
            "normalize()+open() Ray compatibility SPI; implement plan() and "
            "an EngineBinding before the next major release",
            FutureWarning,
            stacklevel=2,
        )
        resolved = provider.normalize(resolved_input)
        handle = provider.open(resolved)
        try:
            return handle.to_ray_dataset()
        finally:
            handle.close()

    result = open_ingestion(
        IngestionRequest(source=resolved_input, engine="ray"),
        project_root_path=project_root_path,
    )
    try:
        if not isinstance(result.handle, RayDataHandle):
            raise JobConfigurationError(
                "Ray compatibility adapter received a non-Ray ingestion handle"
            )
        return result.handle.dataset
    finally:
        result.close()
