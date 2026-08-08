"""Ray Data Parquet binding implemented only through public Ray APIs."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs

from tributo.data._s3 import to_pyarrow_s3_kwargs
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
class _RayNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class RayParquetBinding:
    """Compile Parquet requests into native lazy Ray Data plans."""

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
        """Keep ordered ETL residual to the native Ray Dataset plan."""
        return residual_decisions(request.transforms)

    def build_native_plan(
        self, request: BindingCompileRequest, plan: FileScan
    ) -> _RayNativePlan:
        """Create a lazy plan through ``ray.data.read_parquet``."""
        import ray.data

        reader_options: dict[str, Any] = {}
        columns = plan.options.get("columns")
        if columns:
            reader_options["columns"] = list(columns)
        if request.read_options.target_parallelism is not None:
            reader_options["override_num_blocks"] = (
                request.read_options.target_parallelism
            )
        if request.read_options.batch_size is not None:
            reader_options["batch_size"] = request.read_options.batch_size
        if request.read_options.concurrency is not None:
            reader_options["concurrency"] = request.read_options.concurrency

        path = plan.uri
        if plan.filesystem_id == "s3":
            profile = runtime_s3_profile(request.runtime_options)
            reader_options["filesystem"] = pafs.S3FileSystem(
                **to_pyarrow_s3_kwargs(profile)
            )
            path = path.removeprefix("s3://")

        dataset = ray.data.read_parquet(path, **reader_options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayNativePlan(dataset, schema, transforms, plan.filesystem_id)

    def wrap_handle(
        self,
        native_plan: _RayNativePlan,
        decisions: tuple[TransformDecision, ...],
    ) -> BindingCompilation:
        """Apply native residuals and expose a typed non-materialized handle."""
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
            reader_api="ray.data.read_parquet",
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
