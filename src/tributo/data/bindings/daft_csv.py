"""Daft CSV Binding implemented through the public reader API."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data._s3 import to_daft_s3_kwargs
from tributo.data.bindings._shared import (
    canonical_engine_schema,
    require_file_scan,
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
class _DaftCsvNativePlan:
    dataframe: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class DaftCsvBinding:
    """Compile CSV requests into native lazy Daft plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_file_scan(
                request.plan,
                connector_id="csv",
                filesystem_ids=frozenset({"local", "file", "s3"}),
            )
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    def _build(
        self, request: BindingCompileRequest, plan: FileScan
    ) -> _DaftCsvNativePlan:
        import daft

        options: dict[str, Any] = {}
        if plan.filesystem_id == "s3":
            options["io_config"] = daft.io.IOConfig(
                s3=daft.io.S3Config(
                    **to_daft_s3_kwargs(runtime_s3_profile(request.runtime_options))
                )
            )
        dataframe = daft.read_csv(plan.uri, **options)
        columns = plan.options.get("columns")
        if columns:
            dataframe = dataframe.select(*columns)
        schema = canonical_engine_schema(dataframe.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.DAFT, schema
        )
        return _DaftCsvNativePlan(dataframe, schema, transforms, plan.filesystem_id)

    @staticmethod
    def _wrap(
        native_plan: _DaftCsvNativePlan,
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
            reader_api="daft.read_csv",
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
