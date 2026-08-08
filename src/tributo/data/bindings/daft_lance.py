"""Daft Lance Binding using the public Lance reader."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data._s3 import to_daft_s3_kwargs
from tributo.data.bindings._shared import (
    canonical_engine_schema,
    residual_decisions,
    runtime_s3_profile,
)
from tributo.data.bindings.ray_lance import (
    _reject_lance_snapshot_ref,
    _require_lance,
)
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    binding_stage,
)
from tributo.data.ingestion import (
    DaftDataFrameHandle,
    PhysicalSplitSummary,
    TransformDecision,
)
from tributo.data.refs import schema_fingerprint
from tributo.data.scan_plan import (
    AsOfVersionRef,
    NumericVersionRef,
    TableScan,
    TagVersionRef,
    UriTableRef,
)
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_daft_df,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _DaftLanceNativePlan:
    dataframe: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class DaftLanceBinding:
    """Compile Lance requests into native lazy Daft plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = _require_lance(request.plan)
            _reject_lance_snapshot_ref(plan)
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: TableScan) -> _DaftLanceNativePlan:
        import daft

        if not isinstance(plan.table, UriTableRef):
            raise JobConfigurationError("Lance URI table is required")
        options: dict[str, Any] = {}
        if isinstance(plan.version_ref, NumericVersionRef):
            options["version"] = plan.version_ref.version
        elif isinstance(plan.version_ref, TagVersionRef):
            options["version"] = plan.version_ref.tag
        elif isinstance(plan.version_ref, AsOfVersionRef):
            options["asof"] = plan.version_ref.timestamp
        scan_options: dict[str, Any] = {}
        columns = plan.options.get("columns")
        if columns:
            scan_options["columns"] = list(columns)
        row_filter = plan.options.get("filter")
        if row_filter:
            scan_options["filter"] = row_filter
        if scan_options:
            options["default_scan_options"] = scan_options
        if plan.table.uri.lower().startswith("s3://"):
            options["io_config"] = daft.io.IOConfig(
                s3=daft.io.S3Config(
                    **to_daft_s3_kwargs(runtime_s3_profile(request.runtime_options))
                )
            )
            transport_id = "s3"
        else:
            transport_id = "local"
        dataframe = daft.read_lance(plan.table.uri, **options)
        schema = canonical_engine_schema(dataframe.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.DAFT, schema
        )
        return _DaftLanceNativePlan(dataframe, schema, transforms, transport_id)

    @staticmethod
    def _wrap(
        native_plan: _DaftLanceNativePlan,
        decisions: tuple[TransformDecision, ...],
    ) -> BindingCompilation:
        transformed = apply_pipeline_to_daft_df(
            native_plan.transforms, native_plan.dataframe
        )
        output_schema = (
            native_plan.transforms.steps[-1].output_schema
            if native_plan.transforms.steps
            else native_plan.input_schema
        )
        return BindingCompilation(
            handle=DaftDataFrameHandle(transformed),
            engine_version=importlib.metadata.version("daft"),
            reader_api="daft.read_lance",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="Lance fragments and partitions are delegated to Daft"
            ),
            diagnostics=("metadata I/O was used for schema inference",),
        )
