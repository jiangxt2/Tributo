"""Ray Dataset consumer adapters over the canonical Ingestion Gateway.

These training helpers preserve existing input shapes and Ray Dataset return
values while constructing an explicit ``IngestionRequest`` and consuming a
typed ``RayDataHandle``. They are not an independent Reader: canonical inputs
and legacy flat dictionaries both enter ``open_ingestion()`` after
normalization. The deprecated Provider ``open()`` branch remains reachable
only from public compatibility entry points such as ``DataConnector.read()``.

``TRIBUTO_DATA_BACKEND=legacy`` remains a deprecated compatibility selector
during the migration window. It emits a warning and uses the same conversion
and Gateway path; it cannot reactivate the removed reader backend.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter

from tributo.data._source_paths import (
    require_local_file_source_exists,
    resolve_file_source_path,
)
from tributo.data.ingestion import IngestionRequest, RayDataHandle, open_ingestion
from tributo.data.source_config import (
    CanonicalSourceInput,
    LegacyConfigNormalizer,
    LegacySourceInput,
    RawSourceConfig,
)
from tributo.exceptions import JobConfigurationError

if TYPE_CHECKING:
    import pandas as pd
    import ray.data


DATA_BACKEND = os.getenv("TRIBUTO_DATA_BACKEND", "provider")


def _check_data_backend() -> None:
    """Validate the deprecated selector without restoring duplicate execution."""
    if DATA_BACKEND == "provider":
        return
    if DATA_BACKEND == "legacy":
        warnings.warn(
            "TRIBUTO_DATA_BACKEND=legacy is deprecated and now uses the "
            "canonical Provider/Gateway execution path; remove the selector",
            FutureWarning,
            stacklevel=3,
        )
        return
    raise JobConfigurationError(
        "TRIBUTO_DATA_BACKEND must be 'provider' or deprecated 'legacy'"
    )


def load_ray_dataset_from_source(
    source: dict[str, Any],
    *,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a Ray Dataset through the canonical Provider compatibility path.

    ``source`` accepts the typed ``type/path/dialect`` and ``provider/uri``
    shapes. Unknown fields fail Pydantic validation. This adapter always
    returns Ray Data for existing training consumers; new code that needs an
    explicit Ray/Daft choice should use ``IngestionGateway``.
    """
    _check_data_backend()
    adapter: TypeAdapter[Any] = TypeAdapter(CanonicalSourceInput)
    config = adapter.validate_python(source)
    config = resolve_file_source_path(config, project_root_path)
    return _load_via_ingestion(config)


def load_ray_dataset_from_config(
    data_config: dict[str, Any],
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Convert a legacy flat config and delegate to the canonical Provider.

    .. deprecated::
        Use ``load_ray_dataset_from_source()`` for a canonical Ray adapter or
        ``IngestionGateway`` for an explicit dual-engine request.
    """
    _check_data_backend()
    warnings.warn(
        "load_ray_dataset_from_config() is deprecated. "
        "Use load_ray_dataset_from_source() with a canonical source dict. "
        "See https://github.com/jiangxt2/Tributo/blob/master/docs/architecture/"
        "migration-safety.md.",
        FutureWarning,
        stacklevel=2,
    )
    if "provider" in data_config:
        raise JobConfigurationError(
            "provider/uri sources require load_ray_dataset_from_source(); "
            "the legacy config entrypoint only performs flat-config conversion"
        )
    return _load_via_ingestion(
        LegacySourceInput(raw=dict(data_config)),
        project_root_path=project_root_path,
    )


def _load_via_ingestion(
    source: CanonicalSourceInput | LegacySourceInput,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Normalize compatibility input and invoke the explicit Ray Binding path."""
    if isinstance(source, LegacySourceInput):
        normalized = LegacyConfigNormalizer.normalize(source.raw)
        if isinstance(normalized, RawSourceConfig):
            raise JobConfigurationError(
                f"Unknown legacy source type: {normalized.type!r}"
            )
        resolved_input = resolve_file_source_path(normalized, project_root_path)
    else:
        resolved_input = resolve_file_source_path(source, project_root_path)
    require_local_file_source_exists(resolved_input)
    result = open_ingestion(
        IngestionRequest(source=resolved_input, engine="ray"),
        project_root_path=project_root_path,
    )
    try:
        if not isinstance(result.handle, RayDataHandle):
            raise JobConfigurationError(
                "Training data loading requires a RayDataHandle; "
                "implicit Daft-to-Ray conversion is disabled"
            )
        return result.handle.dataset
    finally:
        result.close()


def load_dataframe_from_config(
    data_config: dict[str, Any],
    project_root_path: Path | None = None,
) -> "pd.DataFrame":
    """Load a legacy config into driver memory for small compatibility jobs."""
    dataset = load_ray_dataset_from_config(data_config, project_root_path)
    return cast("pd.DataFrame", dataset.to_pandas())
