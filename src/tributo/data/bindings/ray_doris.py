"""Ray Doris Binding delegating to the independent ray-doris package."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from pydantic import ValidationError

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
from tributo.data.source_config import RayReadTaskOptions
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _RayDorisNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str
    filters: _SqlFilters
    is_query: bool = False


def _validated_doris_read_options(
    runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only the validated Doris-specific reader arguments."""
    tablet_size = runtime_options.get("tablet_size")
    if tablet_size is not None and (type(tablet_size) is not int or tablet_size <= 0):
        raise JobConfigurationError(
            "Doris option 'tablet_size' must be a strict positive integer"
        )

    query_plan_policy = runtime_options.get("on_query_plan_error")
    if query_plan_policy is not None and (
        not isinstance(query_plan_policy, str)
        or query_plan_policy not in {"single_task", "error"}
    ):
        raise JobConfigurationError(
            "Doris option 'on_query_plan_error' must be 'single_task' or 'error'"
        )

    raw_remote_args = runtime_options.get("ray_remote_args")
    if raw_remote_args is None:
        ray_remote_args = None
    else:
        try:
            ray_remote_args = RayReadTaskOptions.model_validate(raw_remote_args)
        except ValidationError:
            raise JobConfigurationError(
                "Doris option 'ray_remote_args' contains unsupported Ray task options"
            ) from None

    validated: dict[str, Any] = {}
    if tablet_size is not None:
        validated["tablet_size"] = tablet_size
    if query_plan_policy is not None:
        validated["on_query_plan_error"] = query_plan_policy
    if ray_remote_args is not None:
        validated["ray_remote_args"] = ray_remote_args.model_dump(
            mode="json", exclude_none=True
        )
    return validated


class RayDorisBinding:
    """Compile Doris table and query reads into ray-doris Dataset plans."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            if (
                isinstance(request.plan, SqlScan)
                and request.plan.connector_id == "doris"
                and isinstance(request.plan.target, ParameterizedQuery)
            ):
                plan = request.plan
                if (
                    plan.sharding.mode is not SqlShardMode.SINGLE
                    or request.read_options.target_parallelism is not None
                ):
                    raise JobConfigurationError(
                        "Doris query reads require single mode without target_parallelism"
                    )
            else:
                plan = require_sql_table(request.plan, "doris")
            require_transform_filters(plan)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, native_plan.filters.decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: SqlScan) -> _RayDorisNativePlan:
        validated_read_options = _validated_doris_read_options(request.runtime_options)
        is_query = isinstance(plan.target, ParameterizedQuery)
        protocol = str(request.runtime_options.get("protocol") or "mysql")
        connection: SqlBindingConnection | SqlBindingTarget
        if isinstance(plan.target, ParameterizedQuery):
            query = request.runtime_options.get("sql")
            if not isinstance(query, str) or digest(query) != plan.target.query_digest:
                raise JobConfigurationError("Doris query does not match its plan")
            if (
                request.runtime_options.get("params") is not None
                or protocol != "mysql"
                or request.runtime_options.get("http_port") is not None
                or request.runtime_options.get("flight_port") is not None
                or validated_read_options.get("tablet_size") not in (None, 1)
                or validated_read_options.get("on_query_plan_error")
                not in (None, "single_task")
            ):
                raise JobConfigurationError(
                    "Doris query reads require MySQL and do not support "
                    "parameters or table-specific options"
                )
            connection = resolve_sql_connection(
                "doris", request.runtime_options, require_database=False
            )
            source_options: dict[str, Any] = {"query": query}
        else:
            target = resolve_sql_target(plan, request.runtime_options)
            connection = target
            source_options = {
                "table": f"{target.database}.{target.table}",
                "columns": target.columns or None,
                "transport": protocol,
            }
        options: dict[str, Any] = {
            "host": connection.host,
            "mysql_port": connection.port,
            "user": connection.username,
            "password": connection.password,
            **source_options,
        }
        http_port = request.runtime_options.get("http_port")
        flight_port = request.runtime_options.get("flight_port")
        if http_port is not None:
            options["http_port"] = int(http_port)
        if flight_port is not None:
            options["flight_port"] = int(flight_port)
        options.update(validated_read_options)
        if request.read_options.batch_size is not None:
            options["batch_size"] = request.read_options.batch_size
        if request.read_options.concurrency is not None:
            options["concurrency"] = request.read_options.concurrency
        target_parallelism = (
            plan.sharding.target_partitions or request.read_options.target_parallelism
        )
        if target_parallelism is not None:
            options["override_num_blocks"] = target_parallelism
        from ray_doris import read_doris

        dataset = read_doris(**options)
        schema = canonical_engine_schema(dataset.schema(fetch_if_missing=is_query))
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        with binding_stage("classify_transforms"):
            filters = (
                _SqlFilters(None, {}, classify_transform_decisions(request.transforms))
                if is_query
                else compile_sql_filters(request.transforms, schema, "doris")
            )
        if filters.predicate is not None:
            dataset = read_doris(**options, filter=filters.predicate)
            filtered_schema = canonical_engine_schema(
                dataset.schema(fetch_if_missing=False)
            )
            if not filtered_schema.equals(schema):
                raise JobConfigurationError(
                    "Doris schema changed while planning the filtered read"
                )
        return _RayDorisNativePlan(
            dataset, schema, transforms, protocol, filters, is_query
        )

    @staticmethod
    def _wrap(
        native_plan: _RayDorisNativePlan,
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
            reader_api="ray_doris.read_doris",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail=(
                    "Doris query uses one ray-doris read task"
                    if native_plan.is_query
                    else "Doris tablets and batches are delegated to ray-doris"
                )
            ),
            diagnostics=("database metadata I/O was used for schema inference",),
        )
