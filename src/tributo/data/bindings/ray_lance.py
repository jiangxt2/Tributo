"""Ray Data Lance Binding using the public Lance reader."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data._s3 import to_lance_storage_options
from tributo.data.bindings._shared import (
    canonical_engine_schema,
    residual_decisions,
    runtime_s3_profile,
)
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    BindingStageError,
    binding_stage,
)
from tributo.data.ingestion import (
    PhysicalSplitSummary,
    RayDataHandle,
    TransformDecision,
)
from tributo.data.refs import schema_fingerprint
from tributo.data.scan_plan import (
    AsOfVersionRef,
    NumericVersionRef,
    SnapshotVersionRef,
    TableScan,
    TagVersionRef,
    UriTableRef,
)
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _RayLanceNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


def _require_lance(plan: Any) -> TableScan:
    if (
        not isinstance(plan, TableScan)
        or plan.connector_id != "lance"
        or not isinstance(plan.table, UriTableRef)
    ):
        raise JobConfigurationError("Lance binding requires a URI-backed TableScan")
    return plan


def _reject_lance_snapshot_ref(plan: TableScan) -> None:
    """Reject Iceberg snapshot identifiers at the Lance binding boundary."""
    if isinstance(plan.version_ref, SnapshotVersionRef):
        raise BindingStageError.framework_diagnostic(
            "validate_capabilities",
            error_type=JobConfigurationError,
            diagnostic_code="unsupported_lance_snapshot_ref",
            diagnostic=(
                "Lance bindings accept numeric versions, tags, and supported "
                "as-of timestamps; Iceberg snapshot identifiers are not Lance "
                "versions"
            ),
        )


class RayLanceBinding:
    """Compile Lance requests into native lazy Ray Data plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = _require_lance(request.plan)
            _reject_lance_snapshot_ref(plan)
            if isinstance(plan.version_ref, AsOfVersionRef):
                raise BindingStageError.framework_diagnostic(
                    "validate_capabilities",
                    error_type=JobConfigurationError,
                    diagnostic_code="unsupported_lance_asof",
                    diagnostic="Ray Lance binding does not support as-of versions",
                )
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: TableScan) -> _RayLanceNativePlan:
        import ray.data

        if not isinstance(plan.table, UriTableRef):
            raise JobConfigurationError("Lance URI table is required")
        options: dict[str, Any] = {}
        if isinstance(plan.version_ref, NumericVersionRef):
            options["version"] = plan.version_ref.version
        elif isinstance(plan.version_ref, TagVersionRef):
            options["version"] = plan.version_ref.tag
        columns = plan.options.get("columns")
        if columns:
            options["columns"] = list(columns)
        row_filter = plan.options.get("filter")
        if row_filter:
            options["filter"] = row_filter
        if plan.table.uri.lower().startswith("s3://"):
            options["storage_options"] = to_lance_storage_options(
                runtime_s3_profile(request.runtime_options)
            )
            transport_id = "s3"
        else:
            transport_id = "local"
        if request.read_options.target_parallelism is not None:
            options["override_num_blocks"] = request.read_options.target_parallelism
        if request.read_options.concurrency is not None:
            options["concurrency"] = request.read_options.concurrency
        dataset = ray.data.read_lance(plan.table.uri, **options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayLanceNativePlan(dataset, schema, transforms, transport_id)

    @staticmethod
    def _wrap(
        native_plan: _RayLanceNativePlan,
        decisions: tuple[TransformDecision, ...],
    ) -> BindingCompilation:
        transformed = apply_pipeline_to_ray_ds(
            native_plan.transforms, native_plan.dataset
        )
        output_schema = (
            native_plan.transforms.steps[-1].output_schema
            if native_plan.transforms.steps
            else native_plan.input_schema
        )
        return BindingCompilation(
            handle=RayDataHandle(transformed),
            engine_version=importlib.metadata.version("ray"),
            reader_api="ray.data.read_lance",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="Lance fragments and blocks are delegated to Ray Data"
            ),
            diagnostics=("metadata I/O was used for schema inference",),
        )
