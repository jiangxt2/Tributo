"""Export one fixed five-Booster X-Learner artifact."""

from __future__ import annotations

import json
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
from tributo.training.exporters.artifact_protocol import ARTIFACT_KIND_REPORT
from tributo.training.exporters.causal_report import CausalReportExporter
from tributo.training.x_learner import (
    X_LEARNER_FORMULA,
    X_LEARNER_QUADRANT_CODES,
    X_LEARNER_STAGES,
    XLearnerModel,
)
from tributo.util.annotations import PublicAPI


class XLearnerExporterOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class XLearnerCausalReportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


@PublicAPI(stability="alpha")
class XLearnerCausalReportExporter(CausalReportExporter):
    """Expose the existing causal JSON report through exporter API v2."""

    api_version: ClassVar[int] = 2
    output_format: ClassVar[str] = "json"
    output_flavor_id: ClassVar[str] = "report"
    artifact_kind: ClassVar[str] = ARTIFACT_KIND_REPORT
    priority: ClassVar[int] = 80
    source_kinds: ClassVar[tuple[str, ...]] = ("x_learner_result",)
    options_model: ClassVar[type[BaseModel]] = XLearnerCausalReportOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        if request.source_kind != "x_learner_result":
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason="Causal report requires an X-Learner result source",
            )
        return SupportResult(supported=True, code="OK")


@PublicAPI(stability="alpha")
class XLearnerExporter:
    """Write five native UBJ Boosters and their fixed composition metadata."""

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "x-learner-v1"
    output_format: ClassVar[str] = "x-learner"
    output_flavor_id: ClassVar[str] = "x-learner-v1"
    priority: ClassVar[int] = 80
    source_kinds: ClassVar[tuple[str, ...]] = ("x_learner_result",)
    options_model: ClassVar[type[BaseModel]] = XLearnerExporterOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(
            supported=request.source_kind == "x_learner_result",
            code="OK"
            if request.source_kind == "x_learner_result"
            else "UNSUPPORTED_SOURCE_KIND",
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del upstream
        model = source.model_object
        if not isinstance(model, XLearnerModel):
            raise TypeError("x-learner-v1 requires XLearnerModel")
        files = []
        for stage in X_LEARNER_STAGES:
            name = f"{stage}.ubj"
            (context.artifact_dir / name).write_bytes(
                bytes(model.boosters[stage].save_raw(raw_format="ubj"))
            )
            files.append(DraftFile(relative_path=name, role="model"))
        metadata = {
            "api_version": 1,
            "feature_names": list(model.feature_names),
            "response_threshold": model.response_threshold,
            "propensity_clip": list(model.propensity_clip),
            "components": {stage: f"{stage}.ubj" for stage in X_LEARNER_STAGES},
            "formula": X_LEARNER_FORMULA,
            "quadrant_codes": X_LEARNER_QUADRANT_CODES,
        }
        (context.artifact_dir / "x_learner.json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        files.append(DraftFile(relative_path="x_learner.json", role="config"))
        return ArtifactDraft(
            name=target.target.name,
            format=self.output_format,
            flavor_id=self.output_flavor_id,
            files=tuple(files),
            entrypoint="x_learner.json",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


__all__ = ["XLearnerCausalReportExporter", "XLearnerExporter"]
