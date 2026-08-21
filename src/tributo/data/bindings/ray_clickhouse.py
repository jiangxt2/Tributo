"""Distributed Ray Data binding for ClickHouse query sources."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._shared import canonical_engine_schema, residual_decisions
from tributo.data.bindings._sql_shared import require_parameterized_query
from tributo.data.clickhouse import ClickHouseDataConnector
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
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import DeveloperAPI


@dataclass(frozen=True)
class _NativePlan:
    dataset: Any
    schema: pa.Schema
    transforms: CompiledPipeline


@DeveloperAPI
class RayClickHouseBinding:
    """Compile a digest-only query plan into independently executed Ray tasks."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_parameterized_query(request.plan, "clickhouse")
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: SqlScan) -> _NativePlan:
        del plan
        options = dict(request.runtime_options)
        sql = options.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise JobConfigurationError("ClickHouse query text is unavailable")
        parallelism = options.get("parallelism")
        if parallelism is None and request.read_options.target_parallelism is not None:
            parallelism = request.read_options.target_parallelism
        dataset = ClickHouseDataConnector().read(
            host=options.get("host") or "localhost",
            port=options.get("port") or 8123,
            database=options.get("database") or "default",
            username=options.get("user") or "default",
            password=options.get("password"),
            sql=sql,
            params=options.get("params") or {},
            sort_key=options.get("sort_key"),
            parallelism=parallelism if parallelism is not None else -1,
        )
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _NativePlan(dataset, schema, transforms)

    @staticmethod
    def _wrap(
        native: _NativePlan, decisions: tuple[TransformDecision, ...]
    ) -> BindingCompilation:
        dataset = apply_pipeline_to_ray_ds(native.transforms, native.dataset)
        output_schema = (
            native.transforms.steps[-1].output_schema
            if native.transforms.steps
            else native.schema
        )
        return BindingCompilation(
            handle=RayDataHandle(dataset),
            engine_version=importlib.metadata.version("ray"),
            reader_api="ray.data.read_datasource",
            transport_id="clickhouse_native",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native.schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="ClickHouse query shards are delegated to Ray Data"
            ),
            diagnostics=(
                "database metadata I/O was used for schema and shard planning",
            ),
        )


__all__ = ["RayClickHouseBinding"]
