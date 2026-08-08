"""HuggingFace transformer → ONNX exporter — ``ModelExporter`` protocol.

Uses ``optimum.onnxruntime.ORTModel`` or ``transformers.onnx.export``
to export a HuggingFace model to ONNX format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel

from tributo._common.dependencies import (
    TORCH,
    TRANSFORMERS,
    DependencyState,
    probe_dependency,
    require_dependency,
)
from tributo.exceptions import JobConfigurationError
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
from tributo.integrations.exporters.options import HFONNXOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class HuggingFaceONNXExporter:
    """Export a HuggingFace transformer model to ONNX.

    Uses ``optimum.onnxruntime`` when available (preferred),
    falling back to ``transformers.onnx`` + ``torch.onnx.export``.
    """

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "hf-onnx-v1"
    priority: ClassVar[int] = 85
    output_format: ClassVar[str] = "onnx"
    output_flavor_id: ClassVar[str] = "hf-onnx-v1"
    source_kinds: ClassVar[tuple[str, ...]] = (
        "hf_model",
        "huggingface_model",
        "transformers",
    )
    options_model: ClassVar[type[BaseModel]] = HFONNXOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=False),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is a HuggingFace model."""
        if request.source_kind not in ("hf_model", "huggingface_model", "transformers"):
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=f"Expected hf_model/transformers source_kind, got {request.source_kind!r}",
            )
        missing: list[str] = []
        if probe_dependency(TRANSFORMERS).state is not DependencyState.AVAILABLE:
            missing.append("transformers")
        if probe_dependency(TORCH).state is not DependencyState.AVAILABLE:
            missing.append("torch")
        if missing:
            requirements = {
                "transformers": f"transformers>={TRANSFORMERS.minimum_version}",
                "torch": f"torch>={TORCH.minimum_version}",
            }
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason=f"{'/'.join(requirements[name] for name in missing)} required",
                missing_dependencies=tuple(missing),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Export the HF model to ONNX."""
        preprocessor = source.metadata.get(
            "preprocessor", source.metadata.get("tokenizer")
        )
        torch = require_dependency(TORCH)

        model = source.model_object
        task: str | None = target.typed_options.get("task")
        opset: int | None = target.typed_options.get("opset")

        # Resolve task from metadata if not explicitly set (the provider
        # writes metadata["task"], not task_type).
        if task is None:
            task = source.metadata.get("task")
        if opset is None:
            opset = 14

        # Export via the transformers onnx exporter — operates on the
        # in-process model object (plan: no second from_pretrained call,
        # which would fail for in-process sources with no model_id).
        _export_with_transformers_onnx(
            model,
            preprocessor,
            task or "default",
            opset,
            context.artifact_dir,
        )

        # Save tokenizer config.
        tokenizer_files: list[DraftFile] = []
        tokenizer = source.metadata.get("tokenizer")
        if tokenizer is not None:
            try:
                tok_dir = context.artifact_dir / "tokenizer"
                tok_dir.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(tok_dir))
                for fp in tok_dir.rglob("*"):
                    if fp.is_file():
                        rel = str(fp.relative_to(context.artifact_dir))
                        tokenizer_files.append(
                            DraftFile(relative_path=rel, role="tokenizer")
                        )
            except Exception:
                logger.warning("Failed to save tokenizer", exc_info=True)

        # Save model config.
        if source.model_config_data:
            config_path = context.artifact_dir / "config.json"
            config_path.write_text(
                json.dumps(source.model_config_data, indent=2, ensure_ascii=False)
            )
            tokenizer_files.append(
                DraftFile(relative_path="config.json", role="config")
            )

        all_files = [DraftFile(relative_path="model.onnx", role="model")]
        all_files.extend(tokenizer_files)

        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="hf-onnx-v1",
            variant=task,
            files=tuple(all_files),
            entrypoint="model.onnx",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "torch": torch.__version__,
                    "transformers": _get_transformers_version(),
                },
                effective_options={
                    "opset": opset,
                    "task": task,
                },
            ),
            derived_from=(),
        )


def _export_with_transformers_onnx(
    model: Any,
    preprocessor: Any | None,
    task: str,
    opset: int,
    artifact_dir: Path,
) -> Path:
    """Use ``transformers.onnx`` after the caller's dependency check."""
    require_dependency(TRANSFORMERS)
    from transformers.onnx import FeaturesManager, export

    if hasattr(model, "config"):
        if preprocessor is None:
            raise JobConfigurationError(
                "Transformers ONNX export requires a preprocessor in "
                "source.metadata['preprocessor']; open the source with a "
                "provider that supplies one"
            )
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model, feature=task
        )
    else:
        # Fallback: direct torch.onnx.export with dummy inputs.
        return _export_torch_onnx_fallback(model, artifact_dir, opset)

    onnx_path = artifact_dir / "model.onnx"
    export(
        preprocessor=preprocessor,
        model=model,
        config=model_onnx_config(model.config),
        opset=opset,
        output=onnx_path,
    )

    logger.info("HF ONNX model exported to %s", onnx_path)
    return onnx_path


def _export_torch_onnx_fallback(
    model: Any,
    artifact_dir: Path,
    opset: int,
) -> Path:
    """Direct torch.onnx.export fallback."""
    torch = require_dependency(TORCH)

    model.eval()
    dummy = torch.zeros(1, 128, dtype=torch.long)
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    onnx_path = artifact_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy, attention_mask),
        str(onnx_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    logger.info("HF ONNX model exported (torch fallback) to %s", onnx_path)
    return onnx_path


def _get_transformers_version() -> str:
    """Return the Transformers version; the caller must require the package."""
    transformers = require_dependency(TRANSFORMERS)
    version = getattr(transformers, "__version__", None)
    return version if isinstance(version, str) else "unknown"
