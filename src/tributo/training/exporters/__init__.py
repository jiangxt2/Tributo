"""Model export module.

Provides PyTorch -> ONNX and other model export functionalities,
plus the generalised ``ArtifactExporter`` protocol for non-model
artifacts (reports, diagnostics, graph snapshots).
"""

from __future__ import annotations

from tributo.training.exporters.artifact_protocol import (
    ARTIFACT_KIND_DIAGNOSTICS,
    ARTIFACT_KIND_GRAPH_SNAPSHOT,
    ARTIFACT_KIND_MODEL,
    ARTIFACT_KIND_REPORT,
    ArtifactExporter,
    is_known_artifact_kind,
)
from tributo.training.exporters.causal_report import CausalReportExporter
from tributo.training.exporters.safetensors import SafetensorsExporter
from tributo.training.exporters.torch_onnx_exporter import export_pytorch_to_onnx
from tributo.training.exporters.torchscript import TorchScriptExporter

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
