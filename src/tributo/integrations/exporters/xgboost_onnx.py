"""XGBoost → ONNX exporter — ``ModelExporter`` protocol implementation.

Wraps the proven ``tributo.training.onnx_exporter.export_to_onnx``
conversion in a protocol-conformant class.
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
from tributo.exporting.options import XGBoostONNXOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class XGBoostONNXExporter:
    """Export an XGBoost Booster to ONNX format.

    Uses onnxmltools ``convert_xgboost`` under the hood.  Temporarily
    mutates ``booster.feature_names`` during export (``mutates_source=True``).
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "xgboost-onnx-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "onnx"
    source_kinds: ClassVar[tuple[str, ...]] = ("xgboost_result",)
    options_model: ClassVar[type[BaseModel]] = XGBoostONNXOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=False),
    )
    mutates_source: ClassVar[bool] = True
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is an XGBoost booster.

        ONNX export is only supported for numeric-feature classification
        (``binary:*`` / ``multi:*`` objectives).  Regression, ranking,
        count, survival, custom objectives and categorical features are
        rejected at plan time — before any work starts.
        """
        if request.source_kind != "xgboost_result":
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=f"Expected source_kind='xgboost_result', got {request.source_kind!r}",
            )
        # Verify onnxmltools + xgboost are available.
        if (
            importlib.util.find_spec("onnxmltools") is None
            or importlib.util.find_spec("xgboost") is None
        ):
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason="onnxmltools/xgboost not available",
                missing_dependencies=("onnxmltools", "xgboost"),
            )
        if request.source_metadata.get("has_categorical_features"):
            return SupportResult(
                supported=False,
                code="CATEGORICAL_FEATURES",
                reason=(
                    "ONNX export only supports numeric features; the source "
                    "has categorical feature types"
                ),
            )
        objective = request.source_metadata.get("objective", "")
        if not objective:
            return SupportResult(
                supported=False,
                code="UNKNOWN_OBJECTIVE",
                reason=(
                    "Cannot verify the XGBoost objective; ONNX export is "
                    "only supported for numeric-feature classification "
                    "(binary:* / multi:*)"
                ),
            )
        if not (objective.startswith("binary:") or objective.startswith("multi:")):
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_OBJECTIVE",
                reason=(
                    f"Objective {objective!r} is not supported for ONNX "
                    "export; only binary:* / multi:* classification with "
                    "numeric features is supported at plan time"
                ),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Convert the XGBoost booster to ONNX and write to *context.artifact_dir*."""
        import numpy as np
        import onnxmltools
        import xgboost
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        booster: xgboost.Booster = source.model_object

        # Infer n_features from the booster's feature_names.
        if booster.feature_names is not None:
            n_features = len(booster.feature_names)
        else:
            raise ValueError(
                "Cannot determine n_features: please set feature_names "
                "on the XGBoost booster before calling this exporter"
            )

        # Save original feature names for ONNX metadata.
        original_names = booster.feature_names
        if original_names is None:
            original_names = [f"f{i}" for i in range(n_features)]
        feature_names_json = json.dumps(original_names)

        # onnxmltools requires feature_names to be f%d format — temporarily rename.
        booster.feature_names = [f"f{i}" for i in range(n_features)]

        try:
            # Choose wrapper based on objective type.  Only binary:* /
            # multi:* reach this point — everything else is rejected at
            # plan time by supports() (numeric-feature classification
            # only, per the plan's first-phase scope).
            # binary:* → binary classification (XGBClassifier, n_classes=2)
            # multi:*  → multi-class classification
            num_class_raw = booster.attr("num_class")
            objective = booster.attr("objective") or ""
            is_classification = objective.startswith("binary:") or objective.startswith(
                "multi:"
            )

            if is_classification:
                wrapper = xgboost.XGBClassifier()
                wrapper._Booster = booster
                n_classes = int(num_class_raw or 0) or 2
                wrapper.__dict__["n_classes_"] = n_classes
                wrapper.__dict__["classes_"] = np.arange(n_classes)
            else:
                wrapper = xgboost.XGBRegressor()
                wrapper._Booster = booster

            initial_types = [("float_input", FloatTensorType([None, n_features]))]
            opset = target.typed_options.get("opset", 12)
            onnx_model = convert_xgboost(
                wrapper,
                initial_types=initial_types,
                target_opset=opset,
            )
        finally:
            # Restore original feature names.
            booster.feature_names = original_names

        # Write feature names to ONNX metadata.
        meta = onnx_model.metadata_props.add()
        meta.key = "feature_names"
        meta.value = feature_names_json

        # Normalise for deterministic bytes: onnxmltools stamps a random
        # UUID into model.graph.name (and doc_strings), which would make
        # every export of the same booster byte-different — breaking the
        # bundle idempotency contract (same request → same artifact).
        onnx_model.graph.name = "xgboost-onnx"
        onnx_model.doc_string = ""
        for node in onnx_model.graph.node:
            node.doc_string = ""

        # Write ONNX file.
        output_path = context.artifact_dir / "model.onnx"
        output_path.write_bytes(onnx_model.SerializeToString())
        logger.info("XGBoost ONNX model written to %s", output_path)

        # Produce artifact draft.  DraftFile only carries relative_path +
        # role; the ExportManager re-hashes every file from disk and
        # produces the trusted ArtifactFile with sha256/size_bytes.
        draft_file = DraftFile(
            relative_path="model.onnx",
            role="model",
        )

        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="onnx-runtime-v1",
            variant=None,
            files=(draft_file,),
            entrypoint="model.onnx",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "xgboost": xgboost.__version__,
                    "onnxmltools": getattr(onnxmltools, "__version__", "unknown"),
                },
                effective_options={
                    k: v
                    for k, v in target.typed_options.items()
                    if k not in ("n_features",)
                },
            ),
            derived_from=(),
        )
