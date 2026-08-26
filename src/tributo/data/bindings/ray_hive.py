"""Ray HiveServer2 Binding delegated to the independent ray-hive package."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from tributo.data.bindings._shared import canonical_engine_schema, residual_decisions
from tributo.data.bindings._sql_shared import require_sql_table
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
from tributo.data.scan_plan import SqlScan, SqlTableRead
from tributo.data.transform_compiler import (
    CompiledPipeline,
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.exceptions import JobConfigurationError


@dataclass(frozen=True)
class _RayHiveNativePlan:
    dataset: Any
    input_schema: pa.Schema
    transforms: CompiledPipeline
    transport_id: str


class RayHiveBinding:
    """Compile structured Hive table reads into ray-hive Ray Datasets."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        with binding_stage("validate_capabilities"):
            plan = require_sql_table(request.plan, "hive")
        with binding_stage("classify_transforms"):
            decisions = residual_decisions(request.transforms)
        with binding_stage("build_native_plan"):
            native_plan = self._build(request, plan)
        with binding_stage("wrap_handle"):
            return self._wrap(native_plan, decisions)

    @staticmethod
    def _build(request: BindingCompileRequest, plan: SqlScan) -> _RayHiveNativePlan:
        from ray_hive import (
            HiveConnectionOptions,
            HiveReadOptions,
            HiveTableIdentifier,
            SecretRef,
            read_hive,
        )

        target = plan.target
        if not isinstance(target, SqlTableRead):
            raise JobConfigurationError("Hive Binding requires a structured table")
        if target.schema is None:
            raise JobConfigurationError("Hive table reads require a database schema")
        runtime = request.runtime_options
        password_env = runtime.get("password_env")
        credentials = SecretRef.env(str(password_env)) if password_env else None
        # read_hive() builds a lazy Ray Dataset; block reads happen when the
        # returned handle is consumed, while schema metadata is planned here.
        connection = HiveConnectionOptions(
            host=str(runtime["host"]),
            port=int(runtime["port"]),
            database=target.schema,
            transport=str(runtime.get("transport") or "binary"),
            auth=str(runtime.get("auth") or "NOSASL"),
            username=(
                str(runtime["username"])
                if runtime.get("username") is not None
                else None
            ),
            credentials=credentials,
            session_options=dict(runtime.get("session_options") or {}),
            connect_timeout=float(runtime.get("connect_timeout") or 10.0),
            rpc_timeout=float(runtime.get("rpc_timeout") or 60.0),
        )
        read = HiveReadOptions(
            table=HiveTableIdentifier(target.schema, target.table),
            columns=target.projection or None,
            fetch_rows=int(runtime.get("fetch_rows") or 10_000),
            target_batch_bytes=int(
                runtime.get("target_batch_bytes") or 64 * 1024 * 1024
            ),
            query_timeout=float(runtime.get("query_timeout") or 600.0),
        )
        dataset = read_hive(connection=connection, read=read)
        schema = canonical_engine_schema(dataset.schema())
        transforms = ConcreteTransformCompiler().compile(
            request.transforms, TransformBackend.RAY, schema
        )
        return _RayHiveNativePlan(
            dataset=dataset,
            input_schema=schema,
            transforms=transforms,
            transport_id=f"hiveserver2.{connection.transport}",
        )

    @staticmethod
    def _wrap(
        native_plan: _RayHiveNativePlan,
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
            reader_api="ray_hive.read_hive",
            transport_id=native_plan.transport_id,
            transform_decisions=decisions,
            input_schema_fingerprint=schema_fingerprint(native_plan.input_schema),
            schema_fingerprint=schema_fingerprint(output_schema),
            metadata_fetched=True,
            physical_splits=PhysicalSplitSummary(
                detail="Strict HiveServer2 query planning is delegated to ray-hive"
            ),
            diagnostics=("Hive schema metadata was fetched through HiveServer2",),
        )
