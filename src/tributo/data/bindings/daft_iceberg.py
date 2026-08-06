"""Daft Iceberg Binding using Daft and PyIceberg public APIs."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data._s3 import to_daft_s3_kwargs, to_iceberg_properties
from tributo.data.bindings._shared import (
    canonical_engine_schema,
    residual_decisions,
    runtime_s3_profile,
)
from tributo.data.bindings.ray_iceberg import _require_iceberg
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
from tributo.data.scan_plan import CatalogTableRef, SnapshotVersionRef, TableScan
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_daft_df,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _DaftIcebergNativePlan:
    dataframe: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline


class DaftIcebergBinding:
    """Compile Iceberg requests into native lazy Daft plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = _require_iceberg(request.plan)
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(
        request: BindingCompileRequest, plan: TableScan
    ) -> _DaftIcebergNativePlan:
        import daft
        from pyiceberg.catalog import load_catalog

        if not isinstance(plan.table, CatalogTableRef):
            raise JobConfigurationError("Iceberg catalog table is required")
        runtime = request.runtime_options
        profile = runtime_s3_profile(runtime)
        catalog_properties = dict(runtime.get("catalog_properties", {}))
        catalog_properties.update(to_iceberg_properties(profile))
        catalog_name = str(runtime.get("catalog_name") or plan.table.catalog_id)
        catalog = load_catalog(catalog_name, **catalog_properties)
        table_identifier = ".".join((*plan.table.namespace, plan.table.table))
        table = catalog.load_table(table_identifier)
        io_config = daft.io.IOConfig(s3=daft.io.S3Config(**to_daft_s3_kwargs(profile)))
        snapshot_id = (
            plan.version_ref.snapshot_id
            if isinstance(plan.version_ref, SnapshotVersionRef)
            else None
        )
        dataframe = daft.read_iceberg(
            table,
            snapshot_id=snapshot_id,
            io_config=io_config,
        )
        selected = plan.options.get("selected_fields")
        if selected:
            dataframe = dataframe.select(*selected)
        schema = canonical_engine_schema(dataframe.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.DAFT, schema
        )
        return _DaftIcebergNativePlan(dataframe, schema, transforms)

    @staticmethod
    def _wrap(
        native_plan: _DaftIcebergNativePlan,
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
            reader_api="daft.read_iceberg",
            transport_id="iceberg_catalog",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="snapshots, manifests, and data files are delegated to Daft"
            ),
            diagnostics=("catalog metadata I/O was used for schema inference",),
        )
