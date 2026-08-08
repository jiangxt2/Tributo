"""XGBoost native exporters with one plugin per output format."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from pydantic import BaseModel

from tributo._common.dependencies import (
    XGBOOST,
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
from tributo.integrations.exporters.options import (
    XGBoostJSONOptions,
    XGBoostNativeOptions,
    XGBoostUBJOptions,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from xgboost import Booster as XGBoostBooster


class _XGBoostNativeExporterBase:
    """Shared writer; concrete subclasses freeze format and flavor identity."""

    api_version: ClassVar[int] = 2
    priority: ClassVar[int] = 80
    source_kinds: ClassVar[tuple[str, ...]] = ("xgboost_result",)
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    exporter_id: ClassVar[str]
    output_format: ClassVar[str]
    output_flavor_id: ClassVar[str]
    options_model: ClassVar[type[BaseModel]]
    _file_extension: ClassVar[str]

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is an XGBoost booster."""
        if request.source_kind != "xgboost_result":
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=(
                    "Expected source_kind='xgboost_result', got "
                    f"{request.source_kind!r}"
                ),
            )
        if probe_dependency(XGBOOST).state is not DependencyState.AVAILABLE:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason="xgboost>=2.1.0 required",
                missing_dependencies=("xgboost",),
            )
        return SupportResult(supported=True, code="OK")

    def _write_model(self, booster: XGBoostBooster, output_path: Path) -> None:
        """Write the concrete representation implemented by a subclass."""
        raise NotImplementedError

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Save one concrete XGBoost native format."""
        del upstream
        xgboost = require_dependency(XGBOOST)
        booster: XGBoostBooster = source.model_object
        output_path = context.artifact_dir / f"model.{self._file_extension}"
        self._write_model(booster, output_path)

        feature_names = booster.feature_names
        if feature_names:
            meta_path = context.artifact_dir / "feature_names.json"
            meta_path.write_text(json.dumps(list(feature_names)), encoding="utf-8")

        files: list[DraftFile] = [
            DraftFile(
                relative_path=f"model.{self._file_extension}",
                role="model",
            )
        ]
        if feature_names:
            files.append(DraftFile(relative_path="feature_names.json", role="config"))

        logger.info(
            "XGBoost %s model written to %s",
            self.output_format,
            output_path,
        )
        return ArtifactDraft(
            name=target.target.name,
            format=self.output_format,
            flavor_id=self.output_flavor_id,
            files=tuple(files),
            entrypoint=f"model.{self._file_extension}",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={"xgboost": xgboost.__version__},
                effective_options=dict(target.typed_options),
            ),
        )


@PublicAPI(stability="beta")
class XGBoostUBJExporter(_XGBoostNativeExporterBase):
    """Export an XGBoost Booster as Universal Binary JSON (UBJ)."""

    exporter_id: ClassVar[str] = "xgboost-ubj-v1"
    output_format: ClassVar[str] = "ubj"
    output_flavor_id: ClassVar[str] = "xgboost-ubj-v1"
    options_model: ClassVar[type[BaseModel]] = XGBoostUBJOptions
    _file_extension: ClassVar[str] = "ubj"

    def _write_model(self, booster: XGBoostBooster, output_path: Path) -> None:
        output_path.write_bytes(bytes(booster.save_raw(raw_format="ubj")))


@PublicAPI(stability="beta")
class XGBoostJSONExporter(_XGBoostNativeExporterBase):
    """Export an XGBoost Booster in XGBoost's JSON representation."""

    exporter_id: ClassVar[str] = "xgboost-json-v1"
    output_format: ClassVar[str] = "xgboost-json"
    output_flavor_id: ClassVar[str] = "xgboost-json-v1"
    options_model: ClassVar[type[BaseModel]] = XGBoostJSONOptions
    _file_extension: ClassVar[str] = "json"

    def _write_model(self, booster: XGBoostBooster, output_path: Path) -> None:
        booster.save_model(str(output_path))


@PublicAPI(stability="beta")
class XGBoostNativeExporter(_XGBoostNativeExporterBase):
    """Deprecated direct-import adapter for the former ``fmt`` option.

    This class is intentionally absent from default and entry-point
    registration.  ``ExportTarget`` converts legacy configuration to one of
    the concrete exporters before planning.
    """

    exporter_id: ClassVar[str] = "xgboost-native-v1"
    output_format: ClassVar[str] = "xgboost"
    output_flavor_id: ClassVar[str] = "xgboost-native-v1"
    options_model: ClassVar[type[BaseModel]] = XGBoostNativeOptions
    _file_extension: ClassVar[str] = "ubj"

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Preserve direct callers while canonical config uses new plugins."""
        fmt = target.typed_options.get("fmt", "ubj")
        exporter: _XGBoostNativeExporterBase
        if fmt == "json":
            exporter = XGBoostJSONExporter()
        else:
            exporter = XGBoostUBJExporter()
        draft = exporter.export(context, source, upstream, target)
        return draft.model_copy(
            update={
                "format": self.output_format,
                "flavor_id": self.output_flavor_id,
                "variant": fmt,
                "producer": draft.producer.model_copy(
                    update={
                        "exporter_id": self.exporter_id,
                        "effective_options": {
                            key: value
                            for key, value in target.typed_options.items()
                            if key != "fmt"
                        },
                    }
                ),
            }
        )


__all__ = [
    "XGBoostJSONExporter",
    "XGBoostNativeExporter",
    "XGBoostUBJExporter",
]
