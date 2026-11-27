"""Ray ClickHouse Binding delegated to the independent ray-clickhouse package."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._shared import canonical_engine_schema
from tributo.data.bindings._sql_filters import (
    _SqlFilters,
    compile_sql_filters,
    require_transform_filters,
)
from tributo.data.bindings._sql_shared import (
    SqlBindingConnection,
    SqlBindingTarget,
    require_sql_table,
    resolve_sql_connection,
    resolve_sql_target,
)
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    BindingStageError,
    binding_stage,
    classify_transform_decisions,
)
from tributo.data.ingestion import (
    PhysicalSplitSummary,
    RayDataHandle,
    TransformDecision,
)
from tributo.data.refs import digest, schema_fingerprint
from tributo.data.scan_plan import ParameterizedQuery, SqlScan, SqlShardMode
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
    filters: _SqlFilters


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
    """Compile ClickHouse table and query reads into native Datasets."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            if (
                isinstance(request.plan, SqlScan)
                and request.plan.connector_id == "clickhouse"
                and isinstance(request.plan.target, ParameterizedQuery)
            ):
                plan = request.plan
            else:
                plan = require_sql_table(request.plan, "clickhouse")
            require_transform_filters(plan)
            if isinstance(plan.target, ParameterizedQuery) and (
                plan.sharding.mode is not SqlShardMode.SINGLE
            ):
                raise JobConfigurationError(
                    "ClickHouse query reads require single mode"
                )
            split_options = _split_options(request, plan)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan, split_options)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, native_plan.filters.decisions)

    @staticmethod
    def _build(
        request: BindingCompileRequest,
        plan: SqlScan,
        split_options: dict[str, Any],
    ) -> _RayClickHouseNativePlan:
        is_query = isinstance(plan.target, ParameterizedQuery)
        connection: SqlBindingConnection | SqlBindingTarget
        if isinstance(plan.target, ParameterizedQuery):
            query = request.runtime_options.get("sql")
            params = request.runtime_options.get("params")
            if not isinstance(query, str) or digest(query) != plan.target.query_digest:
                raise JobConfigurationError("ClickHouse query does not match its plan")
            if params is not None and not isinstance(params, Mapping):
                raise JobConfigurationError(
                    "ClickHouse query parameters must be a mapping"
                )
            if (digest(params) if params else None) != plan.options.get(
                "params_digest"
            ):
                raise JobConfigurationError(
                    "ClickHouse query parameters do not match its plan"
                )
            connection = resolve_sql_connection("clickhouse", request.runtime_options)
            source_options: dict[str, Any] = {"query": query}
            if params is not None:
                source_options["query_parameters"] = dict(params)
        else:
            target = resolve_sql_target(plan, request.runtime_options)
            connection = target
            source_options = {"table": target.table, "columns": target.columns or None}
        options: dict[str, Any] = {
            "host": connection.host,
            "port": connection.port,
            "database": connection.database,
            "username": connection.username,
            "password": connection.password,
            **source_options,
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

        from ray_clickhouse import read_clickhouse

        dataset = read_clickhouse(**options)
        schema = canonical_engine_schema(dataset.schema(fetch_if_missing=False))
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        with binding_stage("classify_transforms"):
            filters = (
                _SqlFilters(None, {}, classify_transform_decisions(request.transforms))
                if is_query
                else compile_sql_filters(request.transforms, schema, "clickhouse")
            )
        if filters.predicate is not None:
            dataset = read_clickhouse(
                **options,
                filter=filters.predicate,
                query_parameters=filters.parameters,
            )
            filtered_schema = canonical_engine_schema(
                dataset.schema(fetch_if_missing=False)
            )
            if not filtered_schema.equals(schema):
                raise JobConfigurationError(
                    "ClickHouse schema changed while planning the filtered read"
                )
        return _RayClickHouseNativePlan(
            dataset=dataset,
            input_schema=schema,
            transforms=transforms,
            split_mode=str(options["split"]),
            filters=filters,
        )

    @staticmethod
    def _wrap(
        native_plan: _RayClickHouseNativePlan,
        decisions: tuple[TransformDecision, ...],
    ) -> BindingCompilation:
        transformed = apply_pipeline_to_ray_ds(
            native_plan.transforms,
            native_plan.dataset,
            ordinals=native_plan.filters.residual_ordinals,
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
