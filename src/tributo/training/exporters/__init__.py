"""Deprecated re-exports from ``tributo.exporting``.

All bundle export functionality has moved to ``tributo.exporting``.
These re-exports exist for backward compatibility and will be removed
in a future release.

Also re-exports the generalised ``ArtifactExporter`` protocol for
non-model artifacts (reports, diagnostics, graph snapshots).
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from tributo.training.exporters.artifact_protocol import (
    ARTIFACT_KIND_DIAGNOSTICS,
    ARTIFACT_KIND_GRAPH_SNAPSHOT,
    ARTIFACT_KIND_MODEL,
    ARTIFACT_KIND_REPORT,
    ArtifactExporter,
    is_known_artifact_kind,
)

if TYPE_CHECKING:
    from tributo.training.exporters.causal_report import CausalReportExporter
    from tributo.training.exporters.safetensors import SafetensorsExporter
    from tributo.training.exporters.torch_onnx_exporter import export_pytorch_to_onnx
    from tributo.training.exporters.torchscript import TorchScriptExporter

_LAZY_EXPORTS = {
    "CausalReportExporter": (
        "tributo.training.exporters.causal_report",
        "CausalReportExporter",
    ),
    "SafetensorsExporter": (
        "tributo.training.exporters.safetensors",
        "SafetensorsExporter",
    ),
    "TorchScriptExporter": (
        "tributo.training.exporters.torchscript",
        "TorchScriptExporter",
    ),
    "export_pytorch_to_onnx": (
        "tributo.training.exporters.torch_onnx_exporter",
        "export_pytorch_to_onnx",
    ),
}

__all__ = [
    "ArtifactExporter",
    "ARTIFACT_KIND_MODEL",
    "ARTIFACT_KIND_REPORT",
    "ARTIFACT_KIND_DIAGNOSTICS",
    "ARTIFACT_KIND_GRAPH_SNAPSHOT",
    "CausalReportExporter",
    "export_pytorch_to_onnx",
    "is_known_artifact_kind",
    "SafetensorsExporter",
    "TorchScriptExporter",
]


def __getattr__(name: str) -> Any:
    lazy = _LAZY_EXPORTS.get(name)
    if lazy is not None:
        module_name, attribute_name = lazy
        value = getattr(importlib.import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    if name in globals():
        return globals()[name]
    warnings.warn(
        f"tributo.training.exporters.{name} is deprecated; "
        f"use tributo.exporting.{name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        return getattr(importlib.import_module(f"tributo.exporting.{name}"), name)
    except ModuleNotFoundError:
        raise AttributeError(
            f"module 'tributo.training.exporters' has no attribute {name!r}"
        ) from None
