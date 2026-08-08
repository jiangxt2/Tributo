"""Daft OLAP Bindings delegating to the independent connector package."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, ClassVar

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
class _DaftOlapNativePlan:
    dataframe: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    reader_api: str
    transport_id: str


class _DaftOlapBinding:
    connector_id: ClassVar[str]
    reader_api: ClassVar[str]

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, self.connector_id)
            if (
                plan.sharding.mode is SqlShardMode.SINGLE
                and request.read_options.target_parallelism is not None
            ):
                raise BindingStageError.framework_diagnostic(
                    "validate_capabilities",
                    error_type=JobConfigurationError,
                    diagnostic_code="single_sql_read_rejects_parallelism_hint",
                    diagnostic=(
                        "A single SQL read cannot honor target_parallelism; "
                        "set partitioning.mode to 'auto' or 'parallel', or "
                        "remove target_parallelism"
                    ),
                )
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    def _build(
        self, request: BindingCompileRequest, plan: SqlScan
    ) -> _DaftOlapNativePlan:
        from daft_olap import read_clickhouse, read_doris

        target = resolve_sql_target(plan, request.runtime_options)
        options: dict[str, Any] = {
            "host": target.host,
            "database": target.database,
            "table": target.table,
            "username": target.username,
            "password": target.password,
            "columns": target.columns or None,
            "split": (
                "single" if plan.sharding.mode is SqlShardMode.SINGLE else "auto"
            ),
        }
        if request.read_options.batch_size is not None:
            options["batch_rows"] = request.read_options.batch_size
        target_tasks = (
            plan.sharding.target_partitions or request.read_options.target_parallelism
        )
        if target_tasks is not None:
            options["target_tasks"] = target_tasks
        if self.connector_id == "clickhouse":
            options["port"] = target.port
            dataframe = read_clickhouse(**options)
            transport_id = "clickhouse_native"
        else:
            protocol = str(request.runtime_options.get("protocol") or "mysql")
            options["transport"] = protocol
            options["mysql_port"] = target.port
            http_port = request.runtime_options.get("http_port")
            flight_port = request.runtime_options.get("flight_port")
            if http_port is not None:
                options["http_port"] = int(http_port)
            if flight_port is not None:
                options["flight_port"] = int(flight_port)
            dataframe = read_doris(**options)
            transport_id = protocol
        schema = canonical_engine_schema(dataframe.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.DAFT, schema
        )
        return _DaftOlapNativePlan(
            dataframe,
            schema,
            transforms,
            self.reader_api,
            transport_id,
        )

    @staticmethod
    def _wrap(
        native_plan: _DaftOlapNativePlan,
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
            reader_api=native_plan.reader_api,
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="database splits and batches are delegated to daft-olap-connectors"
            ),
            diagnostics=("database metadata I/O was used for schema inference",),
        )


class DaftClickHouseBinding(_DaftOlapBinding):
    connector_id = "clickhouse"
    reader_api = "daft_olap.read_clickhouse"


class DaftDorisBinding(_DaftOlapBinding):
    connector_id = "doris"
    reader_api = "daft_olap.read_doris"
