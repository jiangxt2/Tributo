"""Pass-through exporter for already validated in-memory ONNX model bytes."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, ConfigDict

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


@PublicAPI(stability="beta")
class PrebuiltONNXOptions(BaseModel):
    """No-op options contract for pass-through ONNX publication."""

    model_config = ConfigDict(extra="forbid")


@PublicAPI(stability="beta")
class PrebuiltONNXExporter:
    """Stage trusted ONNX bytes produced by a constrained algorithm finalizer."""

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "prebuilt-onnx-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "onnx"
    output_flavor_id: ClassVar[str] = "onnx-runtime-v1"
    source_kinds: ClassVar[tuple[str, ...]] = ("prebuilt_onnx",)
    options_model: ClassVar[type[BaseModel]] = PrebuiltONNXOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Accept only the explicit prebuilt ONNX source contract."""
        if request.source_kind != "prebuilt_onnx":
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason="prebuilt-onnx-v1 requires source_kind='prebuilt_onnx'",
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Write the immutable model bytes into the isolated staging directory."""
        del upstream
        payload = source.model_object
        if not isinstance(payload, bytes) or not payload:
            raise TypeError("prebuilt ONNX source must contain non-empty bytes")
        output_path = context.artifact_dir / "model.onnx"
        output_path.write_bytes(payload)
        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id=self.output_flavor_id,
            files=(DraftFile(relative_path="model.onnx", role="model"),),
            entrypoint="model.onnx",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions=dict(source.metadata.get("framework_versions", {})),
                effective_options={},
            ),
        )


__all__ = ["PrebuiltONNXExporter", "PrebuiltONNXOptions"]
