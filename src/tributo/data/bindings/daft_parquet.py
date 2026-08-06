"""Daft Parquet binding implemented only through public Daft APIs."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data._s3 import to_daft_s3_kwargs
from tributo.data.bindings._shared import (
    canonical_engine_schema,
    require_parquet_file_scan,
    residual_decisions,
    runtime_s3_profile,
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
from tributo.data.scan_plan import FileScan
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_daft_df,
)


@dataclass(frozen=True)
class _DaftNativePlan:
    dataframe: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class DaftParquetBinding:
    """Compile Parquet requests into native lazy Daft plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = self.validate_capabilities(request)
        with binding_stage("classify_transforms"):
            decisions = self.classify_transforms(request)
        with binding_stage("build_native_plan"):
            native_plan = self.build_native_plan(request, plan)
        with binding_stage("wrap_handle"):
            return self.wrap_handle(native_plan, decisions)

    def validate_capabilities(self, request: BindingCompileRequest) -> FileScan:
        """Validate the scan shape after registry-level hint negotiation."""
        return require_parquet_file_scan(request.plan)

    def classify_transforms(
        self, request: BindingCompileRequest
    ) -> tuple[TransformDecision, ...]:
        """Keep ordered ETL residual to the native Daft DataFrame plan."""
        return residual_decisions(request.transforms)

    def build_native_plan(
        self, request: BindingCompileRequest, plan: FileScan
    ) -> _DaftNativePlan:
        """Create a lazy plan through ``daft.read_parquet``."""
        import daft

        reader_options: dict[str, Any] = {}
        if plan.filesystem_id == "s3":
            profile = runtime_s3_profile(request.runtime_options)
            reader_options["io_config"] = daft.io.IOConfig(
                s3=daft.io.S3Config(**to_daft_s3_kwargs(profile))
            )

        dataframe = daft.read_parquet(plan.uri, **reader_options)
        columns = plan.options.get("columns")
        if columns:
            dataframe = dataframe.select(*columns)
        schema = canonical_engine_schema(dataframe.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.DAFT, schema
        )
        return _DaftNativePlan(dataframe, schema, transforms, plan.filesystem_id)

    def wrap_handle(
        self,
        native_plan: _DaftNativePlan,
        decisions: tuple[TransformDecision, ...],
    ) -> BindingCompilation:
        """Apply native residuals and expose a typed non-materialized handle."""
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
            reader_api="daft.read_parquet",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="physical partitions are delegated to Daft"
            ),
            diagnostics=("metadata I/O was used for schema inference",),
        )
