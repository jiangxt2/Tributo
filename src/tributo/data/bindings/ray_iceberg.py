"""Ray Data Iceberg Binding using the public catalog reader."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data._s3 import to_iceberg_properties
from tributo.data.bindings._shared import (
    canonical_engine_schema,
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
from tributo.data.scan_plan import CatalogTableRef, SnapshotVersionRef, TableScan
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _RayIcebergNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline


def _require_iceberg(plan: Any) -> TableScan:
    if (
        not isinstance(plan, TableScan)
        or plan.connector_id != "iceberg"
        or not isinstance(plan.table, CatalogTableRef)
    ):
        raise JobConfigurationError(
            "Iceberg binding requires a catalog-backed Iceberg TableScan"
        )
    return plan


class RayIcebergBinding:
    """Compile Iceberg requests into native lazy Ray Data plans."""

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
    ) -> _RayIcebergNativePlan:
        import ray.data

        if not isinstance(plan.table, CatalogTableRef):
            raise JobConfigurationError("Iceberg catalog table is required")
        table_identifier = ".".join((*plan.table.namespace, plan.table.table))
        runtime = request.runtime_options
        catalog_kwargs = dict(runtime.get("catalog_properties", {}))
        catalog_kwargs.update(
            to_iceberg_properties(runtime_s3_profile(request.runtime_options))
        )
        catalog_kwargs["name"] = str(
            runtime.get("catalog_name") or plan.table.catalog_id
        )
        options: dict[str, Any] = {
            "table_identifier": table_identifier,
            "catalog_kwargs": catalog_kwargs,
        }
        selected = plan.options.get("selected_fields")
        if selected:
            options["selected_fields"] = tuple(selected)
        row_filter = plan.options.get("row_filter")
        if row_filter:
            options["row_filter"] = row_filter
        if isinstance(plan.version_ref, SnapshotVersionRef):
            options["snapshot_id"] = plan.version_ref.snapshot_id
        if request.read_options.target_parallelism is not None:
            options["override_num_blocks"] = request.read_options.target_parallelism
        dataset = ray.data.read_iceberg(**options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayIcebergNativePlan(dataset, schema, transforms)

    @staticmethod
    def _wrap(
        native_plan: _RayIcebergNativePlan,
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
            reader_api="ray.data.read_iceberg",
            transport_id="iceberg_catalog",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="snapshots, manifests, and data files are delegated to Ray Data"
            ),
            diagnostics=("catalog metadata I/O was used for schema inference",),
        )
