"""Algorithm-neutral export boundary for explainability configuration."""

from __future__ import annotations

from tributo.exporting.models import BundleOutputConfig, ExportSource
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
def prepare_bundle_output_config(
    config: BundleOutputConfig,
    source: ExportSource,
) -> BundleOutputConfig:
    """Preserve explicit targets; algorithm Wheels add required companions.

    Core never infers an algorithm-specific model format from ``source_kind``.
    A Wheel that offers exact native attribution must publish its companion
    artifact and bind ``explainability_model`` explicitly.
    """
    del source
    return config


__all__ = ["prepare_bundle_output_config"]
