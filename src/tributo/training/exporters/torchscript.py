"""TorchScriptExporter — export PyTorch models to TorchScript (.pt).

Primary use case: GNN models whose dynamic scatter / message-passing
operations are not supported by ONNX export.  TorchScript tracing
captures the full forward pass including control flow.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Mapping

from tributo.training.exporters.artifact_protocol import ARTIFACT_KIND_MODEL
from tributo.training.exporters.models import (
    ArtifactDraft,
    DraftFile,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class TorchScriptExporter:
    """Export a ``torch.nn.Module`` to TorchScript (.pt).

    Uses ``torch.jit.script`` (preferred) with fallback to ``torch.jit.trace``
    when scripting is not supported (e.g. dynamic control flow).

    Class variables:
        artifact_kind: ``"model"`` — TorchScript is a model artifact.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "torchscript-v1"
    artifact_kind: ClassVar[str] = ARTIFACT_KIND_MODEL

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Export the model as a TorchScript file.

        Args:
            context: Per-node export context (artifact_dir, execution_id).
            source: Read-only snapshot of the training result.
            upstream: Resolved upstream artifacts (unused for single-model export).
            target: Matched export target with typed options.

        Returns:
            An ``ArtifactDraft`` referencing the .pt file.
        """
        import torch

        model = source.model_object
        if model is None:
            raise ValueError(
                "TorchScriptExporter requires source.model_object to be set."
            )
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}")

        model.eval()
        output_path = context.artifact_dir / "model.pt"

        # Prefer scripting; fall back to tracing with sample inputs.
        try:
            scripted = torch.jit.script(model)
        except Exception:
            logger.info(
                "torch.jit.script failed for %s, falling back to trace.",
                type(model).__name__,
            )
            sample_inputs = source.sample_inputs.get("inputs")
            if sample_inputs is not None:
                scripted = torch.jit.trace(model, sample_inputs)
            else:
                raise RuntimeError(
                    "torch.jit.trace requires sample_inputs, but "
                    "source.sample_inputs is empty."
                ) from None

        torch.jit.save(scripted, str(output_path))

        return ArtifactDraft(
            name=target.target.name,
            format="torchscript",
            flavor_id="torch",
            files=(DraftFile(relative_path="model.pt", role="model"),),
            entrypoint="model.pt",
            producer=ProducerInfo(exporter_id=self.exporter_id),
            artifact_kind=self.artifact_kind,
        )
