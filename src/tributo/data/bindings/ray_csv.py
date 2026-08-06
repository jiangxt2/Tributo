"""Ray Data CSV Binding implemented through the public reader API."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.fs as pafs

from tributo.data._s3 import to_pyarrow_s3_kwargs
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
    PhysicalSplitSummary,
    RayDataHandle,
    TransformDecision,
)
from tributo.data.refs import schema_fingerprint
from tributo.data.scan_plan import FileScan
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)


@dataclass(frozen=True)
class _RayCsvNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class RayCsvBinding:
    """Compile CSV requests into native lazy Ray Data plans."""

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
    ) -> _RayCsvNativePlan:
        import ray.data

        options: dict[str, Any] = {}
        columns = plan.options.get("columns")
        if columns:
            options["convert_options"] = pacsv.ConvertOptions(
                include_columns=list(columns)
            )
        if request.read_options.target_parallelism is not None:
            options["override_num_blocks"] = request.read_options.target_parallelism
        if request.read_options.concurrency is not None:
            options["concurrency"] = request.read_options.concurrency
        path = plan.uri
        if plan.filesystem_id == "s3":
            options["filesystem"] = pafs.S3FileSystem(
                **to_pyarrow_s3_kwargs(runtime_s3_profile(request.runtime_options))
            )
            path = path.removeprefix("s3://")
        dataset = ray.data.read_csv(path, **options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayCsvNativePlan(dataset, schema, transforms, plan.filesystem_id)

    @staticmethod
    def _wrap(
        native_plan: _RayCsvNativePlan,
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
            reader_api="ray.data.read_csv",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="physical blocks are delegated to Ray Data"
            ),
            diagnostics=("metadata I/O was used for schema inference",),
        )
