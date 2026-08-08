"""Daft PostgreSQL Binding using the public SQL reader."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._postgresql import (
    SqlAlchemyConnectionFactory,
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
    DaftDataFrameHandle,
    PhysicalSplitSummary,
    TransformDecision,
)
from tributo.data.refs import schema_fingerprint
from tributo.data.scan_plan import SqlScan, SqlShardMode
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_daft_df,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _DaftPostgreSqlNativePlan:
    dataframe: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline


class DaftPostgreSqlBinding:
    """Compile structured PostgreSQL reads into native lazy Daft plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "postgresql")
            if plan.sharding.mode is SqlShardMode.AUTO:
                raise BindingStageError.framework_diagnostic(
                    "validate_capabilities",
                    error_type=JobConfigurationError,
                    diagnostic_code="unsupported_postgresql_auto_read",
                    diagnostic=(
                        "Daft PostgreSQL automatic sharding requires an explicit "
                        "partition column; use parallel mode with a column or a "
                        "single read"
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
    ) -> _DaftPostgreSqlNativePlan:
        import daft

        target = resolve_sql_target(plan, request.runtime_options)
        options: dict[str, Any] = {}
        if plan.sharding.mode is SqlShardMode.PARALLEL:
            options["partition_col"] = plan.sharding.columns[0]
            if plan.sharding.target_partitions is not None:
                options["num_partitions"] = plan.sharding.target_partitions
            options["partition_bound_strategy"] = str(
                plan.options.get("partition_bound_strategy") or "min-max"
            )
        dataframe = daft.read_sql(
            compile_table_query(plan),
            SqlAlchemyConnectionFactory(target),
            **options,
        )
        schema = canonical_engine_schema(dataframe.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.DAFT, schema
        )
        return _DaftPostgreSqlNativePlan(dataframe, schema, transforms)

    @staticmethod
    def _wrap(
        native_plan: _DaftPostgreSqlNativePlan,
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
            reader_api="daft.read_sql",
            transport_id="postgresql_sqlalchemy",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="PostgreSQL partitions and batches are delegated to Daft"
            ),
            diagnostics=("database metadata I/O was used for schema inference",),
        )
