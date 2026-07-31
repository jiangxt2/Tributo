"""PyTorch 2 Export (PT2) — ``ModelExporter`` for ``torch.export``.

Produces a PT2 ``ExportedProgram`` via ``torch.export.export()`` and
serialises it to ``.pt2`` archive format.  PT2 archives are single-file,
self-contained, and include the graph IR plus parameters.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.models import (
    ArtifactDraft,
    DraftFile,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class TorchExportOptions(BaseModel):
    """Options for ``TorchExportExporter`` (PT2)."""

    model_config = ConfigDict(extra="forbid")

    dynamic_shapes: bool = False
    strict: bool = True
    preserve_module_call_signature: bool = True
    decompile_errors: bool = False


@PublicAPI(stability="beta")
class TorchExportExporter:
    """Export a PyTorch module via ``torch.export.export()`` → .pt2 archive.

    The PT2 format is the recommended export path in PyTorch 2.x,
    superseding ``torch.jit.script`` and ``torch.jit.trace``.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "torch-export-v1"
    priority: ClassVar[int] = 75
    output_format: ClassVar[str] = "pt2"
    options_model: ClassVar[type[BaseModel]] = TorchExportOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check PT2 support."""
        if request.source_kind not in ("dnn_result", "torch_module"):
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=f"Expected dnn_result/torch_module, got {request.source_kind!r}",
            )
        try:
            import torch  # noqa: F401
            if not hasattr(torch, "export"):
                return SupportResult(
                    supported=False,
                    code="TORCH_EXPORT_UNAVAILABLE",
                    reason="torch.export requires PyTorch >= 2.1",
                )
        except ImportError as exc:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason=f"torch not available: {exc}",
                missing_dependencies=("torch>=2.1",),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Run torch.export and save to .pt2."""
        import torch

        model = source.model_object
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Expected torch.nn.Module, got {type(model).__name__}"
            )

        # Prepare example inputs.
        example_inputs = _build_example_inputs(source, model)

        opts: Any = target.typed_options
        dynamic_shapes: bool = opts.get("dynamic_shapes", False)
        strict: bool = opts.get("strict", True)

        # Run torch.export.
        try:
            ep = torch.export.export(
                model,
                example_inputs,
                dynamic_shapes=_build_dynamic_shapes(example_inputs) if dynamic_shapes else None,
                strict=strict,
            )
        except Exception as exc:
            # Fallback to non-strict.
            if strict and "constraint" in str(exc).lower():
                logger.info(
                    "torch.export strict mode failed, retrying with strict=False: %s", exc
                )
                ep = torch.export.export(
                    model,
                    example_inputs,
                    dynamic_shapes=_build_dynamic_shapes(example_inputs) if dynamic_shapes else None,
                    strict=False,
                )
            else:
                raise

        # Save .pt2 archive.
        output_path = context.artifact_dir / "model.pt2"
        torch.export.save(ep, str(output_path))

        # Save example inputs metadata for runtime.
        meta_path = context.artifact_dir / "export_metadata.json"
        meta_data = {
            "export_method": "torch.export",
            "input_structure": _describe_inputs(example_inputs),
            "graph_node_count": len(ep.graph.nodes),
            "graph_module_call_signature": str(ep.graph.signature),
        }
        meta_path.write_text(json.dumps(meta_data, indent=2))

        files: list[DraftFile] = [
            DraftFile(relative_path="model.pt2", role="model"),
            DraftFile(relative_path="export_metadata.json", role="config"),
        ]

        logger.info(
            "PT2 archive exported to %s (%d graph nodes)",
            output_path,
            len(ep.graph.nodes),
        )

        return ArtifactDraft(
            name=target.target.name,
            format="pt2",
            flavor_id="torch-export-v1",
            variant=None,
            files=tuple(files),
            entrypoint="model.pt2",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "torch": torch.__version__,
                },
                effective_options={
                    k: v for k, v in target.typed_options.items()
                    if k not in ("dynamic_shapes", "strict")
                },
            ),
            derived_from=(),
        )


def _build_example_inputs(source: ExportSource, model: Any) -> tuple[Any, ...]:
    """Build example inputs for torch.export.

    Uses sample_inputs if available, otherwise generates a dummy tensor
    from the model's first parameter.
    """
    import torch

    sample = source.sample_inputs
    if sample:
        # If sample is a dict of tensors, convert to tuple.
        if isinstance(sample, dict):
            return tuple(sample.values())
        if isinstance(sample, (list, tuple)):
            return tuple(sample)
        return (torch.tensor(sample),)

    # Generate dummy input from model weight shape.
    try:
        first_param = next(model.parameters())
        batch_size = 1
        input_shape = (batch_size, first_param.shape[1] if len(first_param.shape) > 1 else first_param.shape[0])
        return (torch.randn(input_shape),)
    except (StopIteration, AttributeError):
        return (torch.randn(1, 64),)


def _build_dynamic_shapes(inputs: tuple[Any, ...]) -> dict[str, Any] | None:
    """Build dynamic shapes dict from example inputs."""
    if not inputs:
        return None
    result: dict[str, Any] = {}
    for i, inp in enumerate(inputs):
        if hasattr(inp, "shape"):
            shape_desc = {}
            for j, dim in enumerate(inp.shape):
                if j == 0:
                    shape_desc[j] = None  # Batch dim is dynamic.
                else:
                    shape_desc[j] = dim
            if shape_desc:
                result[str(i)] = shape_desc
    return result if result else None


def _describe_inputs(inputs: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Describe example inputs for metadata."""
    desc = []
    for i, inp in enumerate(inputs):
        if hasattr(inp, "shape"):
            desc.append({
                "index": i,
                "shape": list(inp.shape),
                "dtype": str(inp.dtype),
            })
        else:
            desc.append({"index": i, "type": type(inp).__name__})
    return desc
