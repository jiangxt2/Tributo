"""XGBoost native exporter — UBJ / JSON format via ``save_raw`` / ``save_model``.

Implements the ``ModelExporter`` protocol for the ``"xgboost"`` output
format.  Picks UBJ by default (compact binary, ~½ the size of JSON).
"""

from __future__ import annotations

import importlib.util
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
from tributo.exporting.options import XGBoostNativeOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class XGBoostNativeExporter:
    """Export an XGBoost Booster to native UBJ or JSON format.

    Uses ``booster.save_raw()`` (UBJ) or ``booster.save_model()`` (JSON).
    Does NOT mutate the source.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "xgboost-native-v1"
    priority: ClassVar[int] = 80
    output_format: ClassVar[str] = "xgboost"
    source_kinds: ClassVar[tuple[str, ...]] = ("xgboost_result",)
    options_model: ClassVar[type[BaseModel]] = XGBoostNativeOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is an XGBoost booster."""
        if request.source_kind != "xgboost_result":
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=f"Expected source_kind='xgboost_result', got {request.source_kind!r}",
            )
        if importlib.util.find_spec("xgboost") is None:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason="xgboost not available",
                missing_dependencies=("xgboost",),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Save the XGBoost booster to UBJ or JSON."""
        import xgboost

        booster: xgboost.Booster = source.model_object
        fmt = target.typed_options.get("fmt", "ubj")

        if fmt == "ubj":
            raw: bytes = bytes(booster.save_raw())
            ext = "ubj"
        else:
            ext = "json"
            # Use a temp path for JSON (save_model only writes to files).
            json_path = context.artifact_dir / f"model.{ext}"
            booster.save_model(str(json_path))
            raw = json_path.read_bytes()

        output_path = context.artifact_dir / f"model.{ext}"
        if fmt == "ubj":
            output_path.write_bytes(raw)
        # JSON already written above.

        # Save feature names as metadata.
        feature_names = booster.feature_names
        if feature_names:
            meta_path = context.artifact_dir / "feature_names.json"
            meta_path.write_text(json.dumps(list(feature_names)))

        files: list[DraftFile] = [
            DraftFile(relative_path=f"model.{ext}", role="model"),
        ]
        if feature_names:
            files.append(DraftFile(relative_path="feature_names.json", role="config"))

        logger.info(
            "XGBoost native model (%s) written to %s",
            fmt,
            output_path,
        )

        return ArtifactDraft(
            name=target.target.name,
            format="xgboost",
            flavor_id="xgboost-native-v1",
            variant=fmt,
            files=tuple(files),
            entrypoint=f"model.{ext}",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "xgboost": xgboost.__version__,
                },
                effective_options={
                    k: v for k, v in target.typed_options.items() if k not in ("fmt",)
                },
            ),
            derived_from=(),
        )
