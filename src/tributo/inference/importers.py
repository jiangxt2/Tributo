"""Compatibility imports for the Integrations-owned importer registry.

New code imports from :mod:`tributo.integrations.model_importers`. This module
remains for the documented alpha compatibility window.
"""

from tributo.integrations.model_importers.registry import (
    ModelImporter,
    ModelImporterRegistry,
    build_default_model_importer_registry,
)

__all__ = [
    "ModelImporter",
    "ModelImporterRegistry",
    "build_default_model_importer_registry",
]
