"""ArtifactExporter protocol — generalised artifact export contract.

``ArtifactExporter`` is the super-protocol to ``ModelExporter``.  Where
``ModelExporter`` assumes the output is always a model file, ``ArtifactExporter``
accepts any serialisable output via the ``artifact_kind`` class variable:

* ``"model"`` — trained model (ONNX, TorchScript, safetensors, XGBoost).
* ``"report"`` — statistical report (causal effect JSON, evaluation summary).
* ``"diagnostics"`` — debugging / profiling data.
* ``"graph_snapshot"`` — GNN graph snapshot.

Existing ``ModelExporter`` implementors are treated as ``artifact_kind="model"``
by the compatibility adapter in the ``BundleExportService``.
"""

from __future__ import annotations

from typing import ClassVar, Mapping, Protocol, runtime_checkable

from tributo.exporting.models import (
    ArtifactDraft,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ResolvedArtifact,
)
from tributo.util.annotations import PublicAPI


@runtime_checkable
@PublicAPI(stability="beta")
class ArtifactExporter(Protocol):
    """Generalised artifact export protocol.

    Every exporter declares its ``artifact_kind`` so that the manifest
    can record what kind of artifact was produced — model, report,
    diagnostics, or graph snapshot.

    Class variables:
        api_version: Set to ``1`` for the first-generation protocol.
        exporter_id: Unique string (e.g. ``"causal-report-v1"``).
        artifact_kind: One of ``"model"``, ``"report"``, ``"diagnostics"``,
            ``"graph_snapshot"``.
    """

    api_version: ClassVar[int]
    exporter_id: ClassVar[str]
    artifact_kind: ClassVar[str]

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft: ...


# ── Artifact kind constants ──────────────────────────────────────────────────

ARTIFACT_KIND_MODEL = "model"
ARTIFACT_KIND_REPORT = "report"
ARTIFACT_KIND_DIAGNOSTICS = "diagnostics"
ARTIFACT_KIND_GRAPH_SNAPSHOT = "graph_snapshot"

_KNOWN_ARTIFACT_KINDS: frozenset[str] = frozenset(
    {
        ARTIFACT_KIND_MODEL,
        ARTIFACT_KIND_REPORT,
        ARTIFACT_KIND_DIAGNOSTICS,
        ARTIFACT_KIND_GRAPH_SNAPSHOT,
    }
)


@PublicAPI(stability="beta")
def is_known_artifact_kind(kind: str) -> bool:
    """Return ``True`` if *kind* is a recognised artifact kind constant."""
    return kind in _KNOWN_ARTIFACT_KINDS
