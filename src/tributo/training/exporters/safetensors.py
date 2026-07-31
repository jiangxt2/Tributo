"""SafetensorsExporter — export PyTorch state_dict to safetensors.

Safetensors is a safe, zero-copy serialisation format for tensor data.
This exporter saves the model's ``state_dict`` for fast inference loading,
and is suitable for both GNN and DNN models.
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
class SafetensorsExporter:
    """Export ``torch.nn.Module.state_dict`` to safetensors format.

    Suitable for both GNN and DNN models.  The saved file can be loaded
    with ``safetensors.torch.load_file`` for fast inference.

    Class variables:
        artifact_kind: ``"model"`` — safetensors is a model artifact.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "safetensors-v1"
    artifact_kind: ClassVar[str] = ARTIFACT_KIND_MODEL

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Export the model state_dict as a safetensors file.

        Args:
            context: Per-node export context (artifact_dir, execution_id).
            source: Read-only snapshot of the training result.
            upstream: Resolved upstream artifacts (unused).
            target: Matched export target with typed options.

        Returns:
            An ``ArtifactDraft`` referencing the .safetensors file.
        """
        import torch

        try:
            from safetensors.torch import save_file
        except ImportError as err:
            raise ImportError(
                "safetensors is required for SafetensorsExporter. "
                "Install with: pip install safetensors"
            ) from err

        model = source.model_object
        if model is None:
            raise ValueError(
                "SafetensorsExporter requires source.model_object to be set."
            )
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}")

        state_dict = model.state_dict()
        # Convert to CPU and contiguous for safetensors compatibility.
        cpu_state_dict = {
            k: v.detach().cpu().contiguous() for k, v in state_dict.items()
        }

        output_path = context.artifact_dir / "model.safetensors"
        save_file(cpu_state_dict, str(output_path))

        return ArtifactDraft(
            name=target.target.name,
            format="safetensors",
            flavor_id="torch",
            files=(DraftFile(relative_path="model.safetensors", role="model"),),
            entrypoint="model.safetensors",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
            ),
            artifact_kind=self.artifact_kind,
        )
