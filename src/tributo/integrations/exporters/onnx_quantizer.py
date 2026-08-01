"""ONNX INT8 quantizer — artifact-to-artifact ``ModelExporter``.

Consumes an upstream FP32 ONNX artifact (declared via
``upstream_requirements``) and produces a dynamically-quantised INT8
ONNX model using ``onnxruntime.quantization.quantize_dynamic``.

The planner injects a ``publish=False`` FP32 ONNX node automatically when
no explicit FP32 target exists (per the plan's implicit-node design), so
a user can request ``cpu-int8`` alone and the FP32 upstream is produced
and discarded transparently.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
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
    UpstreamRequirement,
    ValidatorBinding,
)
from tributo.exporting.options import ONNXQuantizerOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class ONNXQuantizer:
    """Quantise an upstream FP32 ONNX artifact to INT8 (dynamic mode).

    Requires an explicit or implicit upstream target named ``model`` of
    format ``onnx`` (declared via ``upstream_requirements``).
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "onnx-quantizer-v1"
    # Above root exporters: a target that declares a non-explicit dep
    # matching this exporter's upstream_requirements is a transform request
    # and must select the quantizer, not a root ONNX exporter.  The
    # supports() MISSING_UPSTREAM gate keeps it out of plain requests.
    priority: ClassVar[int] = 110
    output_format: ClassVar[str] = "onnx"
    # Source kind this exporter consumes ("" for transform exporters).
    source_kinds: ClassVar[tuple[str, ...]] = ()
    options_model: ClassVar[type[BaseModel]] = ONNXQuantizerOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=False),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = (
        UpstreamRequirement(
            name="model",
            format="onnx",
            options={"quantization": None},
        ),
    )

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check that an FP32 ONNX upstream is available and onnxruntime is."""
        if "onnx" not in request.upstream_formats:
            return SupportResult(
                supported=False,
                code="MISSING_UPSTREAM",
                reason=(
                    "ONNX quantizer requires an upstream FP32 ONNX artifact "
                    "(format='onnx' in depends_on)"
                ),
            )
        # Probe the top-level package only: find_spec() on a dotted name
        # imports the parent first, so a missing onnxruntime raises
        # ModuleNotFoundError instead of returning None. Submodule presence
        # (onnxruntime.quantization) is validated at export time by the
        # real import.
        if importlib.util.find_spec("onnxruntime") is None:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason="onnxruntime not available",
                missing_dependencies=("onnxruntime",),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Quantise the upstream FP32 ONNX model to INT8 (dynamic)."""
        import onnxruntime
        import onnxruntime.quantization as ort_quant

        upstream_artifact = upstream.get("model")
        if upstream_artifact is None:
            raise RuntimeError(
                f"{self.exporter_id} requires an upstream 'model' artifact "
                "(FP32 ONNX); declare depends_on=['model']"
            )
        model_path = upstream_artifact.entrypoint_path

        output_path = context.artifact_dir / "model.onnx"
        ort_quant.quantize_dynamic(
            str(model_path),
            str(output_path),
            weight_type=ort_quant.QuantType.QInt8,
        )
        logger.info(
            "Quantised %s → INT8 %s (dynamic)",
            model_path,
            output_path,
        )

        files: list[DraftFile] = [DraftFile(relative_path="model.onnx", role="model")]

        # Transform exporters must copy the upstream's runtime-required
        # config/preprocessor files into their own artifact — derived_from
        # expresses lineage, not runtime file dependencies (plan §Manifest).
        for uf in upstream_artifact.descriptor.files:
            rel = uf.relative_path
            if rel == "model.onnx":
                continue
            src = upstream_artifact.path_for(rel)
            dst = context.artifact_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files.append(DraftFile(relative_path=rel, role=uf.role))

        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="onnx-int8-v1",
            variant="int8-dynamic",
            files=tuple(files),
            entrypoint="model.onnx",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "onnxruntime": onnxruntime.__version__,
                },
                effective_options={"mode": "dynamic-int8"},
            ),
            derived_from=(),
        )
