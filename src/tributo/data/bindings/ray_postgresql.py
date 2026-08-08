"""Ray Data PostgreSQL Binding using the public SQL reader."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._postgresql import (
    PsycopgConnectionFactory,
    compile_table_query,
)
from tributo.data.bindings._shared import canonical_engine_schema, residual_decisions
from tributo.data.bindings._sql_shared import require_sql_table, resolve_sql_target
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    BindingStageError,
    binding_stage,
)
from tributo.data.ingestion import (
    PhysicalSplitSummary,
    RayDataHandle,
    TransformDecision,
)
from tributo.data.refs import schema_fingerprint
from tributo.data.scan_plan import SqlScan, SqlShardMode
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _RayPostgreSqlNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline


class RayPostgreSqlBinding:
    """Compile structured PostgreSQL reads into native Ray Data plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "postgresql")
            if plan.sharding.mode is not SqlShardMode.SINGLE:
                raise BindingStageError.framework_diagnostic(
                    "validate_capabilities",
                    error_type=JobConfigurationError,
                    diagnostic_code="unsupported_postgresql_parallel_read",
                    diagnostic=(
                        "Ray Data PostgreSQL binding cannot honor automatic or "
                        "parallel shard requirements; select Daft with an "
                        "explicit partition column or use a single read"
                    ),
                )
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(
        request: BindingCompileRequest, plan: SqlScan
    ) -> _RayPostgreSqlNativePlan:
        import ray.data

        target = resolve_sql_target(plan, request.runtime_options)
        options: dict[str, Any] = {}
        if request.read_options.concurrency is not None:
            options["concurrency"] = request.read_options.concurrency
        dataset = ray.data.read_sql(
            compile_table_query(plan),
            PsycopgConnectionFactory(target),
            **options,
        )
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayPostgreSqlNativePlan(dataset, schema, transforms)

    @staticmethod
    def _wrap(
        native_plan: _RayPostgreSqlNativePlan,
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
            reader_api="ray.data.read_sql",
            transport_id="postgresql_dbapi",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                split_count=1,
                detail="PostgreSQL batches are delegated to Ray Data",
            ),
            diagnostics=("database metadata I/O was used for schema inference",),
        )
