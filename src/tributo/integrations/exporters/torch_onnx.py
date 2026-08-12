"""PyTorch → ONNX exporter — ``ModelExporter`` protocol for torch.onnx.

Supports both ``torch.onnx.export(dynamo=False)`` and the current
``torch.onnx.export(dynamo=True)`` path. The dynamo path uses the TorchDynamo
ONNX exporter while preserving the requested opset and dynamic batch contract.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel

from tributo._common.dependencies import (
    TORCH,
    DependencyState,
    probe_dependency,
    require_dependency,
)
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
from tributo.integrations.exporters.options import TorchONNXOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class TorchONNXExporter:
    """Export a PyTorch module to ONNX.

    Uses ``torch.onnx.export(dynamo=True)`` when requested, falling back to
    ``torch.onnx.export(dynamo=False)``.

    The dynamo path:
    - Uses FX graph capture instead of tracing.
    - Supports Python control flow and data-dependent shapes.
    - Produces a fully-traced ONNX model with dynamic shapes support.
    """

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "torch-onnx-v1"
    priority: ClassVar[int] = 95
    output_format: ClassVar[str] = "onnx"
    output_flavor_id: ClassVar[str] = "onnx-runtime-v1"
    source_kinds: ClassVar[tuple[str, ...]] = (
        "dnn_result",
        "pu_result",
        "torch_module",
    )
    options_model: ClassVar[type[BaseModel]] = TorchONNXOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is a PyTorch module."""
        if request.source_kind not in ("dnn_result", "pu_result", "torch_module"):
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=(
                    "Expected dnn_result/pu_result/torch_module, "
                    f"got {request.source_kind!r}"
                ),
            )
        if probe_dependency(TORCH).state is not DependencyState.AVAILABLE:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason="torch>=2.5.0 required",
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
        torch = require_dependency(TORCH)

        model = source.model_object
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}")
        if (
            source.source_kind in ("dnn_result", "pu_result")
            and not source.preprocessing_state
        ):
            raise ValueError(
                f"{source.source_kind} requires non-empty preprocessing_state "
                "for Bundle publication"
            )
        if (
            source.source_kind in ("dnn_result", "pu_result")
            and not source.model_config_data
        ):
            raise ValueError(
                f"{source.source_kind} requires non-empty model_config_data "
                "for Bundle publication"
            )

        model_config_json = (
            _serialize_json_artifact(
                source.model_config_data,
                artifact_name="model_config.json",
                source_kind=source.source_kind,
            )
            if source.model_config_data
            else None
        )
        preprocessor_json: str | None = None
        if source.source_kind in ("dnn_result", "pu_result"):
            _validate_preprocessing_state(
                source.preprocessing_state,
                source_kind=source.source_kind,
            )
            preprocessor_json = _serialize_json_artifact(
                source.preprocessing_state,
                artifact_name="preprocessor.json",
                source_kind=source.source_kind,
            )

        # Save training state for restoration (mutates_source=False guarantee).
        was_training = model.training
        model.eval()

        opts: dict[str, Any] = target.typed_options
        opset: int = opts.get("opset", 18)
        dynamo: bool = opts.get("dynamo", True)
        external_data: bool = opts.get("external_data", False)

        # Resolve input names and shapes.
        input_names = _resolve_input_names(source)
        sample_inputs = _resolve_sample_inputs(source, input_names)
        output_names = ["output"]

        # Tracks which path actually ran — the dynamo path may fall back
        # to legacy export at runtime, and the manifest must record the
        # path that was really used.
        used_dynamo = False

        try:
            # ── Dynamo path (PyTorch >= 2.1) ──
            if dynamo:
                try:
                    # ``torch.onnx.export(..., dynamo=True)`` is the current
                    # public API and, unlike the deprecated ``dynamo_export``
                    # helper, accepts the requested opset explicitly.
                    output_path = self._legacy_export(
                        model,
                        sample_inputs,
                        input_names,
                        output_names,
                        opset,
                        context.artifact_dir,
                        external_data,
                        use_dynamo=True,
                    )
                    used_dynamo = True
                    logger.info(
                        "ONNX model exported via torch.onnx.export(dynamo=True) to %s",
                        output_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "torch.onnx.export(dynamo=True) failed: %s — falling back to legacy export(dynamo=False)",
                        exc,
                    )
                    output_path = self._legacy_export(
                        model,
                        sample_inputs,
                        input_names,
                        output_names,
                        opset,
                        context.artifact_dir,
                        external_data,
                        use_dynamo=False,
                    )
            else:
                # Legacy torch.onnx.export path — classic tracing backend.
                output_path = self._legacy_export(
                    model,
                    sample_inputs,
                    input_names,
                    output_names,
                    opset,
                    context.artifact_dir,
                    external_data,
                    use_dynamo=False,
                )

            # Determine which files were produced.
            files: list[DraftFile] = [
                DraftFile(relative_path="model.onnx", role="model")
            ]
            if external_data:
                weights_path = context.artifact_dir / "model_weights.bin"
                if weights_path.exists():
                    files.append(
                        DraftFile(relative_path="model_weights.bin", role="aux")
                    )

            # Save model config for reconstruction.
            if model_config_json is not None:
                config_path = context.artifact_dir / "model_config.json"
                config_path.write_text(model_config_json)
                files.append(
                    DraftFile(relative_path="model_config.json", role="config")
                )

            if source.source_kind in ("dnn_result", "pu_result"):
                assert preprocessor_json is not None
                preprocessor_path = context.artifact_dir / "preprocessor.json"
                preprocessor_path.write_text(preprocessor_json)
                files.append(
                    DraftFile(
                        relative_path="preprocessor.json",
                        role="preprocessor",
                    )
                )

            return ArtifactDraft(
                name=target.target.name,
                format="onnx",
                flavor_id="onnx-runtime-v1",
                variant="dynamo" if used_dynamo else "legacy",
                files=tuple(files),
                entrypoint="model.onnx",
                producer=ProducerInfo(
                    exporter_id=self.exporter_id,
                    framework_versions={
                        "torch": torch.__version__,
                    },
                    effective_options={
                        "opset": opset,
                        "dynamo": used_dynamo,
                        "external_data": external_data,
                    },
                ),
                derived_from=(),
            )
        finally:
            # Restore original training state — even if export raised.
            if was_training:
                model.train()

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
        torch = require_dependency(TORCH)

        output_path = artifact_dir / "model.onnx"

        # When sample_inputs is a single-tensor tuple, unwrap to avoid
        # torch.onnx.export complaining about extra tuple nesting.
        if len(sample_inputs) == 1:
            model_input = sample_inputs[0]
        else:
            model_input = sample_inputs

        export_kwargs: dict[str, Any] = {
            "opset_version": opset,
            "input_names": input_names,
            "output_names": output_names,
            "export_params": True,
            "do_constant_folding": True,
            "external_data": external_data,
        }
        # TORCH requires PyTorch >= 2.5.0, whose export API supports this
        # keyword unconditionally.
        export_kwargs["dynamo"] = use_dynamo
        if use_dynamo:
            batch_dim = torch.export.Dim("batch_size")
            export_kwargs["dynamic_shapes"] = tuple({0: batch_dim} for _ in input_names)
        else:
            export_kwargs["dynamic_axes"] = {
                name: {0: "batch_size"} for name in input_names
            }
        with warnings.catch_warnings():
            # PyTorch 2.13.0 still triggers this upstream pytree deprecation
            # while deep-copying the exported program.  Tributo treats all
            # other warnings normally; this exact warning must not turn a
            # successful dynamo conversion into a legacy fallback.
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated,.*",
                category=FutureWarning,
            )
            torch.onnx.export(model, model_input, str(output_path), **export_kwargs)
        return output_path


def _validate_preprocessing_state(
    state: Mapping[str, Any], *, source_kind: str
) -> None:
    """Validate the FeatureTransformer state required by DNN/PU consumers."""
    required_types: dict[str, type[Any]] = {
        "features": list,
        "label_encoders": dict,
        "norm_params": dict,
    }
    missing = [key for key in required_types if key not in state]
    if missing:
        raise ValueError(
            f"{source_kind} preprocessing_state is missing required key(s): {missing}"
        )
    for key, expected_type in required_types.items():
        value = state[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{source_kind} preprocessing_state[{key!r}] must be "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )
    features = state["features"]
    if not features:
        raise ValueError(
            f"{source_kind} preprocessing_state['features'] must not be empty"
        )
    if any(
        not isinstance(feature, dict)
        or not isinstance(feature.get("name"), str)
        or not feature["name"]
        for feature in features
    ):
        raise ValueError(
            f"{source_kind} preprocessing_state['features'] must contain named "
            "feature objects"
        )


def _serialize_json_artifact(
    value: Mapping[str, Any], *, artifact_name: str, source_kind: str
) -> str:
    """Serialize one metadata artifact without coercion or non-finite values."""
    try:
        return json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_kind} {artifact_name} must contain only finite JSON values: {exc}"
        ) from exc


def _resolve_input_names(source: ExportSource) -> list[str]:
    """Resolve input names from source metadata or feature schema."""
    feature_schema = source.feature_schema
    if feature_schema:
        if "feature_names" in feature_schema:
            return list(feature_schema["feature_names"])
        if "input_names" in feature_schema:
            return list(feature_schema["input_names"])
    return ["input"]


def _resolve_sample_inputs(
    source: ExportSource, input_names: list[str]
) -> tuple[Any, ...]:
    """Resolve sample inputs in the same order as the declared input names."""
    torch = require_dependency(TORCH)

    sample = source.sample_inputs
    if sample:
        missing = [name for name in input_names if name not in sample]
        unexpected = [name for name in sample if name not in input_names]
        if missing or unexpected:
            raise ValueError(
                "sample_inputs must match declared input names exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return tuple(sample[name] for name in input_names)

    # Generate dummy from config or model weights.
    if len(input_names) != 1:
        raise ValueError(
            "sample_inputs are required when exporting a multi-input PyTorch model"
        )
    model_config = source.model_config_data
    input_dim = model_config.get("input_dim", 64)
    return (torch.randn(1, input_dim),)
