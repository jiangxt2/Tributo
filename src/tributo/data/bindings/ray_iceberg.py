"""Ray Data Iceberg Binding using the public catalog reader."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._shared import (
    canonical_engine_schema,
    iceberg_catalog_properties,
    parse_iceberg_row_filter,
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
    empty_schema_from_catalog: bool = False


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
        catalog_kwargs = iceberg_catalog_properties(runtime)
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
            options["row_filter"] = parse_iceberg_row_filter(row_filter)
        if isinstance(plan.version_ref, SnapshotVersionRef):
            options["snapshot_id"] = plan.version_ref.snapshot_id
        if request.read_options.target_parallelism is not None:
            options["override_num_blocks"] = request.read_options.target_parallelism
        dataset = ray.data.read_iceberg(**options)
        engine_schema = dataset.schema(fetch_if_missing=False)
        empty_schema_from_catalog = False
        if engine_schema is None:
            empty_schema = _empty_iceberg_schema(
                catalog_name=str(runtime.get("catalog_name") or plan.table.catalog_id),
                catalog_properties={
                    key: value for key, value in catalog_kwargs.items() if key != "name"
                },
                table_identifier=table_identifier,
                selected_fields=tuple(selected) if selected else (),
                row_filter=options.get("row_filter"),
                snapshot_id=(
                    plan.version_ref.snapshot_id
                    if isinstance(plan.version_ref, SnapshotVersionRef)
                    else None
                ),
            )
            if empty_schema is not None:
                dataset = ray.data.from_arrow(
                    pa.Table.from_batches([], schema=empty_schema)
                )
                engine_schema = empty_schema
                empty_schema_from_catalog = True
            else:
                engine_schema = dataset.schema()
        schema = canonical_engine_schema(engine_schema)
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayIcebergNativePlan(
            dataset,
            schema,
            transforms,
            empty_schema_from_catalog=empty_schema_from_catalog,
        )

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
                detail=(
                    "empty-table schema was recovered from Iceberg catalog metadata"
                    if native_plan.empty_schema_from_catalog
                    else "snapshots, manifests, and data files are delegated to Ray Data"
                )
            ),
            diagnostics=(
                "catalog metadata I/O was used for schema inference",
                *(
                    ("empty table schema was preserved from Iceberg metadata",)
                    if native_plan.empty_schema_from_catalog
                    else ()
                ),
            ),
        )


def _empty_iceberg_schema(
    *,
    catalog_name: str,
    catalog_properties: dict[str, str],
    table_identifier: str,
    selected_fields: tuple[str, ...],
    row_filter: Any | None,
    snapshot_id: int | None,
) -> pa.Schema | None:
    """Return catalog schema only when PyIceberg plans no data files."""
    from pyiceberg.catalog import load_catalog

    table = load_catalog(catalog_name, **catalog_properties).load_table(
        table_identifier
    )
    scan_options: dict[str, Any] = {}
    if selected_fields:
        scan_options["selected_fields"] = selected_fields
    if row_filter:
        scan_options["row_filter"] = row_filter
    if snapshot_id is not None:
        scan_options["snapshot_id"] = snapshot_id
    if next(iter(table.scan(**scan_options).plan_files()), None) is not None:
        return None
    schema = table.schema()
    if selected_fields:
        schema = schema.select(*selected_fields)
    return schema.as_arrow()
