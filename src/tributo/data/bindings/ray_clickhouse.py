"""Ray ClickHouse Binding delegated to the independent ray-clickhouse package."""

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
class _RayClickHouseNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    split_mode: str


def _split_options(
    request: BindingCompileRequest,
    plan: SqlScan,
) -> dict[str, Any]:
    sharding = plan.sharding
    if sharding.mode is SqlShardMode.SINGLE:
        if request.read_options.target_parallelism is not None:
            raise BindingStageError.framework_diagnostic(
                "validate_capabilities",
                error_type=JobConfigurationError,
                diagnostic_code="single_sql_read_rejects_parallelism_hint",
                diagnostic=(
                    "A single ClickHouse read cannot honor target_parallelism; "
                    "set partitioning.mode to 'auto' or 'parallel', or remove "
                    "target_parallelism"
                ),
            )
        return {"split": "single"}
    if sharding.mode is SqlShardMode.AUTO:
        return {"split": "partition"}
    if plan.options.get("partition_bound_strategy") not in (None, "min-max"):
        raise BindingStageError.framework_diagnostic(
            "validate_capabilities",
            error_type=JobConfigurationError,
            diagnostic_code="clickhouse_range_strategy_unsupported",
            diagnostic="ray-clickhouse range splitting supports min-max bounds only",
        )
    if len(sharding.columns) != 1:
        raise BindingStageError.framework_diagnostic(
            "validate_capabilities",
            error_type=JobConfigurationError,
            diagnostic_code="clickhouse_range_requires_one_column",
            diagnostic=(
                "ray-clickhouse range splitting requires exactly one integer "
                "partitioning column"
            ),
        )
    return {"split": "range", "range_column": sharding.columns[0]}


class RayClickHouseBinding:
    """Compile ClickHouse table reads into ray-clickhouse native Datasets."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "clickhouse")
            split_options = _split_options(request, plan)
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan, split_options)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(
        request: BindingCompileRequest,
        plan: SqlScan,
        split_options: dict[str, Any],
    ) -> _RayClickHouseNativePlan:
        from ray_clickhouse import read_clickhouse

        target = resolve_sql_target(plan, request.runtime_options)
        options: dict[str, Any] = {
            "host": target.host,
            "port": target.port,
            "database": target.database,
            "table": target.table,
            "username": target.username,
            "password": target.password,
            "columns": target.columns or None,
            **split_options,
        }
        read_options = request.read_options
        if read_options.batch_size is not None:
            options["batch_rows"] = read_options.batch_size
        if read_options.target_split_size_bytes is not None:
            options["batch_bytes"] = read_options.target_split_size_bytes
        if read_options.concurrency is not None:
            options["concurrency"] = read_options.concurrency
        target_tasks = (
            plan.sharding.target_partitions or read_options.target_parallelism
        )
        if target_tasks is not None:
            options.update(
                {
                    "target_tasks": target_tasks,
                    "max_tasks": target_tasks,
                    "override_num_blocks": target_tasks,
                }
            )

        dataset = read_clickhouse(**options)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayClickHouseNativePlan(
            dataset=dataset,
            input_schema=schema,
            transforms=transforms,
            split_mode=str(options["split"]),
        )

    @staticmethod
    def _wrap(
        native_plan: _RayClickHouseNativePlan,
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
            reader_api="ray_clickhouse.read_clickhouse",
            transport_id="clickhouse.http.arrow_stream",
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail=(
                    "ClickHouse split planning and bounded Arrow streaming are "
                    f"delegated to ray-clickhouse ({native_plan.split_mode})"
                )
            ),
            diagnostics=("ClickHouse schema metadata was fetched by ray-clickhouse",),
        )
