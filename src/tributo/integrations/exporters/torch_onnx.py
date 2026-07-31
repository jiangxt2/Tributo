"""PyTorch → ONNX exporter — ``ModelExporter`` protocol for torch.onnx.

Supports both legacy ``torch.onnx.export(dynamo=False)`` and the new
``torch.onnx.dynamo_export()`` path (PyTorch >= 2.1).  The dynamo path
uses the TorchDynamo ONNX exporter which produces a more optimised graph.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel

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
from tributo.exporting.options import TorchONNXOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class TorchONNXExporter:
    """Export a PyTorch module to ONNX.

    Uses ``torch.onnx.dynamo_export()`` when ``dynamo=True`` and PyTorch
    >= 2.1, falling back to ``torch.onnx.export(dynamo=False)``.

    The dynamo path:
    - Uses FX graph capture instead of tracing.
    - Supports Python control flow and data-dependent shapes.
    - Produces a fully-traced ONNX model with dynamic shapes support.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "torch-onnx-v1"
    priority: ClassVar[int] = 95
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = TorchONNXOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=False),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is a PyTorch module."""
        if request.source_kind not in ("dnn_result", "torch_module"):
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=f"Expected dnn_result/torch_module, got {request.source_kind!r}",
            )
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason=f"torch not available: {exc}",
                missing_dependencies=("torch",),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Export the PyTorch model to ONNX."""
        import torch

        model = source.model_object
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"Expected torch.nn.Module, got {type(model).__name__}"
            )

        # Save training state for restoration (mutates_source=False guarantee).
        was_training = model.training
        model.eval()

        opts: dict[str, Any] = target.typed_options
        opset: int = opts.get("opset", 18)
        dynamo: bool = opts.get("dynamo", False)
        external_data: bool = opts.get("external_data", False)

        # Resolve input names and shapes.
        input_names = _resolve_input_names(source)
        sample_inputs = _resolve_sample_inputs(source, model)
        output_names = ["output"]

        # ── Dynamo path (PyTorch >= 2.1) ──
        if dynamo:
            try:
                if hasattr(torch.onnx, "dynamo_export"):
                    export_options = torch.onnx.ExportOptions(
                        dynamic_shapes=True,
                        onnx_registry=None,
                    )
                    onnx_program = torch.onnx.dynamo_export(
                        model,
                        *sample_inputs,
                        export_options=export_options,
                    )
                    output_path = context.artifact_dir / "model.onnx"
                    onnx_program.save(
                        str(output_path),
                        **(
                            {"external_weights_path": "model_weights.bin"}
                            if external_data
                            else {}
                        ),
                    )
                    logger.info(
                        "ONNX model exported via dynamo_export to %s",
                        output_path,
                    )
                else:
                    # No dynamo_export available — fall through to legacy.
                    raise NotImplementedError("torch.onnx.dynamo_export not available")
            except Exception as exc:
                logger.warning(
                    "torch.onnx.dynamo_export failed: %s — falling back to legacy export(dynamo=False)",
                    exc,
                )
                output_path = self._legacy_export(
                    model, sample_inputs, input_names, output_names,
                    opset, context.artifact_dir, external_data, use_dynamo=False,
                )
        else:
            # Legacy torch.onnx.export path — classic tracing backend.
            output_path = self._legacy_export(
                model, sample_inputs, input_names, output_names,
                opset, context.artifact_dir, external_data, use_dynamo=False,
            )

        # Determine which files were produced.
        files: list[DraftFile] = [DraftFile(relative_path="model.onnx", role="model")]
        if external_data:
            weights_path = context.artifact_dir / "model_weights.bin"
            if weights_path.exists():
                files.append(
                    DraftFile(relative_path="model_weights.bin", role="aux")
                )

        # Save model config for reconstruction.
        if source.model_config_data:
            config_path = context.artifact_dir / "model_config.json"
            config_path.write_text(
                json.dumps(source.model_config_data, indent=2, ensure_ascii=False)
            )
            files.append(DraftFile(relative_path="model_config.json", role="config"))

        # Restore original training state.
        if was_training:
            model.train()

        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="onnx-runtime-v1",
            variant="dynamo" if dynamo else "legacy",
            files=tuple(files),
            entrypoint="model.onnx",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "torch": torch.__version__,
                },
                effective_options={
                    "opset": opset,
                    "dynamo": dynamo,
                    "external_data": external_data,
                },
            ),
            derived_from=(),
        )

    @staticmethod
    def _legacy_export(
        model: Any,
        sample_inputs: tuple[Any, ...],
        input_names: list[str],
        output_names: list[str],
        opset: int,
        artifact_dir: Any,  # Path
        external_data: bool,
        use_dynamo: bool = False,
    ) -> Any:
        """Legacy ``torch.onnx.export`` path (model already in eval mode)."""
        import torch

        output_path = artifact_dir / "model.onnx"

        # When sample_inputs is a single-tensor tuple, unwrap to avoid
        # torch.onnx.export complaining about extra tuple nesting.
        if len(sample_inputs) == 1:
            model_input = sample_inputs[0]
        else:
            model_input = sample_inputs

        torch.onnx.export(
            model,
            model_input,
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                name: {0: "batch_size"} for name in input_names
            },
            dynamo=use_dynamo,
            export_params=True,
            do_constant_folding=True,
        )
        return output_path


def _resolve_input_names(source: ExportSource) -> list[str]:
    """Resolve input names from source metadata or feature schema."""
    feature_schema = source.feature_schema
    if feature_schema:
        if "feature_names" in feature_schema:
            return list(feature_schema["feature_names"])
        if "input_names" in feature_schema:
            return list(feature_schema["input_names"])
    return ["input"]


def _resolve_sample_inputs(source: ExportSource, model: Any) -> tuple[Any, ...]:
    """Resolve sample inputs for ONNX export."""
    import torch

    sample = source.sample_inputs
    if sample:
        if isinstance(sample, dict):
            return tuple(sample.values())
        if isinstance(sample, (list, tuple)):
            return tuple(sample)
        return (torch.tensor(sample),)

    # Generate dummy from config or model weights.
    model_config = source.model_config_data
    input_dim = model_config.get("input_dim", 64)
    return (torch.randn(1, input_dim),)
