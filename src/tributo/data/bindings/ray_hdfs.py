"""Ray Data HDFS file Bindings using PyArrow's public filesystem adapter."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.fs as pafs

from tributo.data.bindings._shared import (
    canonical_engine_schema,
    require_file_scan,
    residual_decisions,
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
class _RayHdfsNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline


class _RayHdfsFileBinding:
    connector_id: ClassVar[str]
    reader_api: ClassVar[str]

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_file_scan(
                request.plan,
                connector_id=self.connector_id,
                filesystem_ids=frozenset({"hdfs"}),
            )
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    def _build(
        self, request: BindingCompileRequest, plan: FileScan
    ) -> _RayHdfsNativePlan:
        import ray.data

        filesystem, path = pafs.HadoopFileSystem.from_uri(plan.uri)
        options: dict[str, Any] = {"filesystem": filesystem}
        if request.read_options.target_parallelism is not None:
            options["override_num_blocks"] = request.read_options.target_parallelism
        if request.read_options.concurrency is not None:
            options["concurrency"] = request.read_options.concurrency
        columns = plan.options.get("columns")
        if self.connector_id == "parquet":
            if columns:
                options["columns"] = list(columns)
            if request.read_options.batch_size is not None:
                options["batch_size"] = request.read_options.batch_size
            dataset = ray.data.read_parquet(path, **options)
        else:
            if columns:
                options["convert_options"] = pacsv.ConvertOptions(
                    include_columns=list(columns)
                )
            dataset = ray.data.read_csv(path, **options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayHdfsNativePlan(dataset, schema, transforms)

    def _wrap(
        self,
        native_plan: _RayHdfsNativePlan,
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
            reader_api=self.reader_api,
            transport_id="hdfs",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="HDFS discovery and physical blocks are delegated to Ray Data"
            ),
            diagnostics=("metadata I/O was used for schema inference",),
        )


class RayHdfsParquetBinding(_RayHdfsFileBinding):
    connector_id = "parquet"
    reader_api = "ray.data.read_parquet"


class RayHdfsCsvBinding(_RayHdfsFileBinding):
    connector_id = "csv"
    reader_api = "ray.data.read_csv"
