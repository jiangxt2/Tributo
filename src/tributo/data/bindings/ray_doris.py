"""Ray Doris Binding delegating to the independent ray-doris package."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._shared import canonical_engine_schema, residual_decisions
from tributo.data.bindings._sql_shared import require_sql_table, resolve_sql_target
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
from tributo.data.scan_plan import SqlScan
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)


@dataclass(frozen=True)
class _RayDorisNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class RayDorisBinding:
    """Compile Doris table reads into ray-doris native Dataset plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "doris")
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: SqlScan) -> _RayDorisNativePlan:
        from ray_doris import read_doris

        target = resolve_sql_target(plan, request.runtime_options)
        protocol = str(request.runtime_options.get("protocol") or "mysql")
        options: dict[str, Any] = {
            "table": f"{target.database}.{target.table}",
            "host": target.host,
            "mysql_port": target.port,
            "user": target.username,
            "password": target.password,
            "columns": target.columns or None,
            "transport": protocol,
        }
        http_port = request.runtime_options.get("http_port")
        flight_port = request.runtime_options.get("flight_port")
        if http_port is not None:
            options["http_port"] = int(http_port)
        if flight_port is not None:
            options["flight_port"] = int(flight_port)
        if request.read_options.batch_size is not None:
            options["batch_size"] = request.read_options.batch_size
        if request.read_options.concurrency is not None:
            options["concurrency"] = request.read_options.concurrency
        target_parallelism = (
            plan.sharding.target_partitions or request.read_options.target_parallelism
        )
        if target_parallelism is not None:
            options["override_num_blocks"] = target_parallelism
        dataset = read_doris(**options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayDorisNativePlan(dataset, schema, transforms, protocol)

    @staticmethod
    def _wrap(
        native_plan: _RayDorisNativePlan,
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
            reader_api="ray_doris.read_doris",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="Doris tablets and batches are delegated to ray-doris"
            ),
            diagnostics=("database metadata I/O was used for schema inference",),
        )
