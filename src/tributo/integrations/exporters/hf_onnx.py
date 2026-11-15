"""HuggingFace transformer → ONNX exporter — ``ModelExporter`` protocol.

Uses the Transformers exporter matching the installed version to export a
HuggingFace model to ONNX format.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, ClassVar, Iterator, Mapping

from packaging.version import InvalidVersion, Version
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

    Configured models use the Transformers ONNX exporter; an explicit sample
    is required when a safe text sample cannot be constructed.
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
        ValidatorBinding(validator_id="onnx-runtime-v1", required=True),
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
                reason=(
                    f"{'/'.join(requirements[name] for name in missing)} required; "
                    "install tributo[model-export-hf]"
                ),
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
            opset = 18

        # Export via the transformers onnx exporter — operates on the
        # in-process model object (plan: no second from_pretrained call,
        # which would fail for in-process sources with no model_id).
        was_training = getattr(model, "training", None)
        modules = getattr(model, "modules", None)
        training_states = (
            [(module, module.training) for module in modules()]
            if callable(modules)
            else []
        )
        try:
            if callable(getattr(model, "eval", None)):
                model.eval()
            _export_with_transformers_onnx(
                model,
                preprocessor,
                task or "default",
                opset,
                context.artifact_dir,
                source.sample_inputs,
            )
        finally:
            if isinstance(was_training, bool):
                model.train(was_training)
            for module, state in training_states:
                module.training = state

        model_files = _inspect_onnx_artifact(context.artifact_dir, opset)

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

        all_files = model_files
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
    sample_inputs: Mapping[str, Any] | None = None,
) -> Path:
    """Use the appropriate Transformers API for the in-memory model."""
    if not hasattr(model, "config"):
        return _export_torch_onnx_fallback(model, artifact_dir, opset)

    transformers = require_dependency(TRANSFORMERS)
    try:
        version = Version(transformers.__version__)
    except (InvalidVersion, AttributeError, TypeError) as exc:
        raise JobConfigurationError("Cannot determine Transformers version") from exc

    onnx_path = artifact_dir / "model.onnx"
    if version.major == 4:
        from transformers.onnx import FeaturesManager, export

        if preprocessor is None:
            raise JobConfigurationError(
                "Transformers ONNX export requires a preprocessor in "
                "source.metadata['preprocessor']; open the source with a "
                "provider that supplies one"
            )
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model, feature=task
        )
        export(
            preprocessor=preprocessor,
            model=model,
            config=model_onnx_config(model.config),
            opset=opset,
            output=onnx_path,
        )
    elif version.major == 5 and version >= Version("5.19.0"):
        try:
            from transformers.exporters import OnnxConfig, OnnxExporter

            config = OnnxConfig(
                output_path=str(onnx_path),
                opset_version=opset,
                dynamic=True,
                external_data=True,
            )
        except (ImportError, TypeError) as exc:
            raise JobConfigurationError(
                "Unsupported Transformers ONNX exporter API; use the tested "
                "Transformers 5.19.0 release"
            ) from exc
        inputs = _resolve_hf_inputs(model, preprocessor, task, sample_inputs or {})
        OnnxExporter().export(model, inputs, config=config)
    else:
        raise JobConfigurationError(
            f"Transformers {version} does not provide a supported ONNX exporter; "
            "use Transformers 4.x or the tested 5.19.0 release"
        )

    logger.info("HF ONNX model exported to %s", onnx_path)
    return onnx_path


def _resolve_hf_inputs(
    model: Any, preprocessor: Any, task: str, sample_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Use caller-provided tensors, or construct a representative text pair."""
    torch = require_dependency(TORCH)
    if sample_inputs:
        if not all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in sample_inputs.items()
        ):
            raise JobConfigurationError("sample_inputs must map names to tensors")
        inputs = dict(sample_inputs)
    else:
        if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            raise JobConfigurationError(
                "Encoder-decoder models require representative source.sample_inputs"
            )
        from transformers import PreTrainedTokenizerBase

        text_tasks = {
            "default",
            "sequence-classification",
            "text-classification",
            "question-answering",
            "token-classification",
            "masked-lm",
        }
        if (
            not isinstance(preprocessor, PreTrainedTokenizerBase)
            or task not in text_tasks
        ):
            raise JobConfigurationError(
                "Provide representative source.sample_inputs for this model"
            )
        parameters = inspect.signature(model.forward).parameters
        if "input_ids" not in parameters:
            raise JobConfigurationError(
                "Cannot infer text inputs; provide source.sample_inputs"
            )
        try:
            generated = preprocessor(
                ["a short example", "another example"],
                ["a second sentence", "another sentence"],
                padding=True,
                return_tensors="pt",
                return_token_type_ids="token_type_ids" in parameters,
            )
        except (TypeError, ValueError) as exc:
            raise JobConfigurationError(
                "Cannot construct text inputs; provide source.sample_inputs"
            ) from exc
        inputs = {k: v for k, v in generated.items() if k in parameters}
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        if not inputs or not required.issubset(inputs):
            raise JobConfigurationError(
                "Cannot infer model inputs; provide source.sample_inputs"
            )
    device = getattr(model, "device", None)
    return {
        name: value.to(device) if device is not None else value
        for name, value in inputs.items()
    }


def _inspect_onnx_artifact(artifact_dir: Path, opset: int) -> list[DraftFile]:
    """Check actual opset and declare safe external tensor files."""
    import onnx

    model = onnx.load_model(str(artifact_dir / "model.onnx"), load_external_data=False)
    versions = [
        entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")
    ]
    if versions != [opset]:
        raise JobConfigurationError(
            f"Requested ONNX opset {opset}, produced {versions}"
        )

    def tensors(message: Any) -> Iterator[Any]:
        if isinstance(message, onnx.TensorProto):
            yield message
            return
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else (value,):
                    yield from tensors(child)

    locations: set[str] = set()
    for tensor in tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        names = [item.value for item in tensor.external_data if item.key == "location"]
        if len(names) != 1:
            raise JobConfigurationError("Invalid ONNX external data location")
        location = names[0]
        path = PurePosixPath(location)
        if (
            not location
            or "\x00" in location
            or "\\" in location
            or path.is_absolute()
            or PureWindowsPath(location).drive
            or location == "model.onnx"
            or any(p in (".", "..") for p in path.parts)
        ):
            raise JobConfigurationError(
                f"Unsafe ONNX external data location: {location!r}"
            )
        current = artifact_dir
        for component in path.parts:
            current = current / component
            if current.is_symlink():
                raise JobConfigurationError(
                    f"ONNX external data is a symlink: {location!r}"
                )
        if not current.is_file() or not current.resolve().is_relative_to(
            artifact_dir.resolve()
        ):
            raise JobConfigurationError(
                f"Missing or unsafe ONNX external data: {location!r}"
            )
        locations.add(location)
    return [DraftFile(relative_path="model.onnx", role="model")] + [
        DraftFile(relative_path=location, role="aux") for location in sorted(locations)
    ]


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
