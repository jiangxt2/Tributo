"""Built-in Ray and Daft write bindings.

Each binding is deliberately thin: it validates the typed handle, prepares
engine-native options, invokes the corresponding public ``write_*`` API, and
returns a small credential-free receipt.  It never implements file, fragment,
manifest, or snapshot data-plane logic itself.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from collections.abc import Mapping
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit

from tributo._common.runtime_versions import (
    SUPPORTED_DAFT_VERSION as DAFT_ENGINE_VERSION,
)
from tributo._common.runtime_versions import (
    SUPPORTED_RAY_VERSION as RAY_ENGINE_VERSION,
)
from tributo.data.contracts.handles import DaftDataFrameHandle, RayDataHandle
from tributo.data.contracts.modes import WriteMode
from tributo.data.writing._native import (
    daft_io_config,
    daft_path,
    descriptor,
    ensure_iceberg_table,
    iceberg_context,
    lance_path,
    lance_storage_options,
    native_path,
    ray_filesystem,
    ray_mode,
    write_receipt,
)
from tributo.data.writing.capabilities import WriteCapability
from tributo.data.writing.contracts import (
    WriteAmbiguousError,
    WriteCapabilityError,
    WriteDescriptor,
    WriteExecutionContext,
    WriteHandle,
    WriteReceipt,
)
from tributo.data.writing.targets import LogicalWritePlan


def _ray_descriptor(
    target_kind: str, binding_id: str, capabilities: WriteCapability
) -> WriteDescriptor:
    dependencies: tuple[str, ...] = (
        ("pyiceberg",)
        if target_kind == "iceberg"
        else ("lance-ray", "pylance")
        if target_kind == "lance"
        else ("ray-clickhouse",)
        if target_kind == "clickhouse"
        else ("ray-doris",)
        if target_kind == "doris"
        else ()
    )
    return descriptor(
        engine_id="tributo.ray_data",
        target_kind=target_kind,
        binding_id=binding_id,
        engine_version=RAY_ENGINE_VERSION,
        dependency_distributions=dependencies,
        capabilities=capabilities,
        installation_hint=(
            "Install ray-clickhouse: uv sync --locked --extra clickhouse"
            if target_kind == "clickhouse"
            else "Install ray-doris: uv sync --locked --extra mysql"
            if target_kind == "doris"
            else None
        ),
    )


def _daft_descriptor(
    target_kind: str, binding_id: str, capabilities: WriteCapability
) -> WriteDescriptor:
    dependencies: tuple[str, ...] = (
        ("pyiceberg",)
        if target_kind == "iceberg"
        else ("pylance", "daft-lance")
        if target_kind == "lance"
        else ()
    )
    return descriptor(
        engine_id="tributo.daft",
        target_kind=target_kind,
        binding_id=binding_id,
        engine_version=DAFT_ENGINE_VERSION,
        dependency_distributions=dependencies,
        capabilities=capabilities,
    )


_FILE_MODES = frozenset({WriteMode.APPEND, WriteMode.OVERWRITE})
_LANCE_MODES = frozenset({WriteMode.CREATE, WriteMode.APPEND, WriteMode.OVERWRITE})
_CLICKHOUSE_MODES = frozenset({WriteMode.APPEND})
_PARQUET_OPTIONS = frozenset({"compression", "min_rows_per_file"})
_ICEBERG_OPTIONS = frozenset({"snapshot_properties"})
_LANCE_OPTIONS = frozenset(
    {"min_rows_per_file", "max_rows_per_file", "data_storage_version"}
)
_CLICKHOUSE_OPTIONS = frozenset(
    {
        "batch_bytes",
        "batch_rows",
        "columns",
        "concurrency",
        "connect_timeout_seconds",
        "insert_mode",
        "query_timeout_seconds",
        "secure",
    }
)
_DORIS_CONNECTION_OPTIONS = frozenset(
    {
        "mysql_port",
        "http_secure",
        "http_ca_file",
        "mysql_ca_file",
        "connect_timeout_seconds",
        "redirect_hosts",
        "redirect_ports",
        "redirect_policy",
    }
)
_DORIS_WRITE_OPTIONS = frozenset(
    {
        "operation",
        "format",
        "batch_rows",
        "batch_bytes",
        "max_filter_ratio",
        "strict_mode",
        "request_timeout_seconds",
        "load_properties",
        "concurrency",
    }
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RayParquetWriteBinding:
    """Delegate Parquet writes to ``ray.data.Dataset.write_parquet``."""

    binding_id = "tributo.ray.parquet"
    _descriptor = _ray_descriptor(
        "parquet",
        binding_id,
        WriteCapability(
            supported_modes=_FILE_MODES,
            supported_options=_PARQUET_OPTIONS,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_ray(plan, input_handle, "parquet")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_ray(plan, input_handle, "parquet")
        kwargs: dict[str, Any] = {
            "compression": str(plan.options.get("compression", "zstd")),
            "mode": ray_mode(plan.mode),
        }
        min_rows = plan.options.get("min_rows_per_file")
        if min_rows is not None:
            kwargs["min_rows_per_file"] = int(min_rows)
        filesystem = ray_filesystem(
            context.runtime_options,
            required=plan.target.lower().startswith("s3://"),
        )
        if filesystem is not None:
            kwargs["filesystem"] = filesystem
        input_handle.dataset.write_parquet(native_path(plan.target), **kwargs)
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="ray.data.Dataset.write_parquet",
        )


class RayCsvWriteBinding:
    """Delegate CSV writes to ``ray.data.Dataset.write_csv``."""

    binding_id = "tributo.ray.csv"
    _descriptor = _ray_descriptor(
        "csv",
        binding_id,
        WriteCapability(
            supported_modes=_FILE_MODES,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_ray(plan, input_handle, "csv")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_ray(plan, input_handle, "csv")
        kwargs: dict[str, Any] = {"mode": ray_mode(plan.mode)}
        filesystem = ray_filesystem(
            context.runtime_options,
            required=plan.target.lower().startswith("s3://"),
        )
        if filesystem is not None:
            kwargs["filesystem"] = filesystem
        input_handle.dataset.write_csv(native_path(plan.target), **kwargs)
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="ray.data.Dataset.write_csv",
        )


class RayIcebergWriteBinding:
    """Delegate Iceberg writes to ``ray.data.Dataset.write_iceberg``."""

    binding_id = "tributo.ray.iceberg"
    _descriptor = _ray_descriptor(
        "iceberg",
        binding_id,
        WriteCapability(
            supported_modes=frozenset({WriteMode.APPEND, WriteMode.OVERWRITE}),
            supported_options=_ICEBERG_OPTIONS,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_ray(plan, input_handle, "iceberg")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_ray(plan, input_handle, "iceberg")
        catalog_name, table_identifier, properties = iceberg_context(
            context.runtime_options, plan.target
        )
        ensure_iceberg_table(
            input_handle, "tributo.ray_data", context.runtime_options, plan.target
        )
        catalog_kwargs = {"name": catalog_name, **properties}
        kwargs: dict[str, Any] = {
            "table_identifier": table_identifier,
            "catalog_kwargs": catalog_kwargs,
            "mode": ray_mode(plan.mode),
        }
        if "snapshot_properties" in plan.options:
            kwargs["snapshot_properties"] = dict(plan.options["snapshot_properties"])
        input_handle.dataset.write_iceberg(**kwargs)
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="ray.data.Dataset.write_iceberg",
        )


class RayLanceWriteBinding:
    """Delegate Lance writes to the official Lance-Ray integration."""

    binding_id = "tributo.ray.lance"
    _descriptor = _ray_descriptor(
        "lance",
        binding_id,
        WriteCapability(
            supported_modes=_LANCE_MODES,
            supported_options=_LANCE_OPTIONS,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_ray(plan, input_handle, "lance")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_ray(plan, input_handle, "lance")
        import lance_ray

        kwargs: dict[str, Any] = {
            "mode": cast(Literal["create", "append", "overwrite"], plan.mode.value),
            "storage_options": lance_storage_options(context.runtime_options),
            "stream": False,
        }
        for option in (
            "min_rows_per_file",
            "max_rows_per_file",
            "data_storage_version",
        ):
            if option in plan.options:
                kwargs[option] = plan.options[option]
        lance_ray.write_lance(
            input_handle.dataset,
            lance_path(plan.target),
            **kwargs,
        )
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="lance_ray.write_lance",
        )


class RayClickHouseWriteBinding:
    """Delegate append-only table writes to the public ray-clickhouse facade."""

    binding_id = "tributo.ray.clickhouse"
    _descriptor = _ray_descriptor(
        "clickhouse",
        binding_id,
        WriteCapability(
            supported_modes=_CLICKHOUSE_MODES,
            supported_options=_CLICKHOUSE_OPTIONS,
            distributed=True,
            native_metrics=True,
            requires_existing_target=True,
            can_create_target=False,
            supports_empty_input=True,
        ),
    )

    @staticmethod
    def is_available() -> bool:
        try:
            importlib.metadata.version("ray-clickhouse")
        except importlib.metadata.PackageNotFoundError:
            return False
        return True

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        _require_ray(plan, input_handle, "clickhouse")
        _clickhouse_target(plan.target)
        _clickhouse_username(plan.runtime_options)
        _clickhouse_password_env(plan.runtime_options)
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_ray(plan, input_handle, "clickhouse")
        host, port, database, table = _clickhouse_target(plan.target)
        password_env = _clickhouse_password_env(context.runtime_options)
        username = _clickhouse_username(context.runtime_options)

        from ray_clickhouse import write_clickhouse

        kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "database": database,
            "table": table,
            "username": username,
            "password_env": password_env,
            "write_mode": "append",
        }
        for name in _CLICKHOUSE_OPTIONS:
            if name in plan.options:
                kwargs[name] = plan.options[name]
        native = write_clickhouse(input_handle.dataset, **kwargs)
        if str(native.status) != "confirmed" or int(native.ambiguous_batches) != 0:
            raise WriteCapabilityError(
                "ray-clickhouse write completed without an unambiguous confirmation"
            )
        return WriteReceipt(
            request_digest=plan.request_digest,
            engine_id=plan.engine_id,
            binding_id=self.binding_id,
            target_kind=plan.target_kind,
            target_ref=plan.target,
            mode=plan.mode,
            committed=True,
            rows_written=int(native.rows_written),
            bytes_written=int(native.bytes_written),
            diagnostics=("ray_clickhouse.write_clickhouse",),
            metadata={
                "batches_written": int(native.batches_written),
                "ambiguous_batches": int(native.ambiguous_batches),
                "native_status": str(native.status),
            },
        )


class RayDorisWriteBinding:
    """Delegate writes to the public ray-doris Stream Load facade."""

    binding_id = "tributo.ray.doris"
    _descriptor = _ray_descriptor(
        "doris",
        binding_id,
        WriteCapability(
            supported_modes=frozenset({WriteMode.APPEND}),
            supported_options=_DORIS_CONNECTION_OPTIONS | _DORIS_WRITE_OPTIONS,
            distributed=True,
            native_metrics=True,
            requires_existing_target=True,
            can_create_target=False,
            supports_empty_input=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        _require_ray(plan, input_handle, "doris")
        _doris_target(plan.target)
        _doris_username(plan.runtime_options)
        _doris_password_env(plan.runtime_options)
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        handle = _require_ray(plan, input_handle, "doris")
        host, port, database, table = _doris_target(plan.target)
        from ray.exceptions import RayTaskError, WorkerCrashedError
        from ray_doris import (
            DorisAmbiguousWriteError,
            DorisConnection,
            DorisTable,
            write_doris,
        )

        connection = DorisConnection(
            host=host,
            http_port=port,
            username=_doris_username(context.runtime_options),
            password_env=_doris_password_env(context.runtime_options),
            **{
                name: plan.options[name]
                for name in _DORIS_CONNECTION_OPTIONS
                if name in plan.options
            },
        )
        try:
            result = write_doris(
                handle.dataset,
                connection=connection,
                table=DorisTable(database, table),
                **{
                    name: plan.options[name]
                    for name in _DORIS_WRITE_OPTIONS
                    if name in plan.options
                },
            )
        except Exception as exc:
            cause = exc.cause if isinstance(exc, RayTaskError) else exc
            if isinstance(cause, DorisAmbiguousWriteError):
                raise WriteAmbiguousError(
                    "Doris may have accepted a batch; reconcile before retrying",
                    source_error_type="DorisAmbiguousWriteError",
                ) from None
            if isinstance(cause, WorkerCrashedError):
                raise WriteAmbiguousError(
                    "A Ray worker exited during the write; reconcile before retrying",
                    source_error_type="WorkerCrashedError",
                ) from None
            raise
        if result.status in {"publish_timeout", "mixed"}:
            raise WriteAmbiguousError(
                "Doris returned a publish timeout; reconcile before retrying",
                source_error_type="DorisPublishTimeout",
            )
        if result.status != "success":
            raise WriteAmbiguousError(
                "Doris returned an unrecognized write status; reconcile before retrying",
                source_error_type="DorisWriteResult",
            )
        return WriteReceipt(
            request_digest=plan.request_digest,
            engine_id=plan.engine_id,
            binding_id=self.binding_id,
            target_kind=plan.target_kind,
            target_ref=plan.target,
            mode=plan.mode,
            committed=True,
            rows_written=result.loaded_rows,
            bytes_written=result.uploaded_bytes,
            diagnostics=("ray_doris.write_doris",),
            metadata={
                "native_status": result.status,
                "batches": result.batches,
                "attempted_rows": result.attempted_rows,
                "filtered_rows": result.filtered_rows,
            },
        )


class DaftParquetWriteBinding:
    """Delegate Parquet writes to ``daft.DataFrame.write_parquet``."""

    binding_id = "tributo.daft.parquet"
    _descriptor = _daft_descriptor(
        "parquet",
        binding_id,
        WriteCapability(
            supported_modes=_FILE_MODES,
            supported_options=frozenset({"compression"}),
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_daft(plan, input_handle, "parquet")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_daft(plan, input_handle, "parquet")
        kwargs: dict[str, Any] = {
            "compression": str(plan.options.get("compression", "zstd")),
            "write_mode": plan.mode.value,
        }
        io_config = daft_io_config(
            context.runtime_options,
            required=plan.target.lower().startswith("s3://"),
        )
        if io_config is not None:
            kwargs["io_config"] = io_config
        input_handle.dataframe.write_parquet(daft_path(plan.target), **kwargs)
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="daft.DataFrame.write_parquet",
        )


class DaftCsvWriteBinding:
    """Delegate CSV writes to ``daft.DataFrame.write_csv``."""

    binding_id = "tributo.daft.csv"
    _descriptor = _daft_descriptor(
        "csv",
        binding_id,
        WriteCapability(
            supported_modes=_FILE_MODES,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_daft(plan, input_handle, "csv")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_daft(plan, input_handle, "csv")
        kwargs: dict[str, Any] = {"write_mode": plan.mode.value}
        io_config = daft_io_config(
            context.runtime_options,
            required=plan.target.lower().startswith("s3://"),
        )
        if io_config is not None:
            kwargs["io_config"] = io_config
        input_handle.dataframe.write_csv(daft_path(plan.target), **kwargs)
        return write_receipt(
            plan=plan, binding_id=self.binding_id, native_api="daft.DataFrame.write_csv"
        )


class DaftIcebergWriteBinding:
    """Delegate Iceberg writes to ``daft.DataFrame.write_iceberg``."""

    binding_id = "tributo.daft.iceberg"
    _descriptor = _daft_descriptor(
        "iceberg",
        binding_id,
        WriteCapability(
            supported_modes=frozenset({WriteMode.APPEND, WriteMode.OVERWRITE}),
            supported_options=_ICEBERG_OPTIONS,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_daft(plan, input_handle, "iceberg")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_daft(plan, input_handle, "iceberg")
        table, _, _ = ensure_iceberg_table(
            input_handle, "tributo.daft", context.runtime_options, plan.target
        )
        kwargs: dict[str, Any] = {
            "mode": plan.mode.value,
            "io_config": daft_io_config(
                context.runtime_options,
                required=plan.target.lower().startswith("s3://"),
            ),
        }
        if "snapshot_properties" in plan.options:
            kwargs["snapshot_properties"] = dict(plan.options["snapshot_properties"])
        input_handle.dataframe.write_iceberg(table, **kwargs)
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="daft.DataFrame.write_iceberg",
        )


class DaftLanceWriteBinding:
    """Delegate Lance writes to ``daft.DataFrame.write_lance``."""

    binding_id = "tributo.daft.lance"
    _descriptor = _daft_descriptor(
        "lance",
        binding_id,
        WriteCapability(
            supported_modes=_LANCE_MODES,
            distributed=True,
            native_metrics=False,
            can_create_target=True,
        ),
    )

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        input_handle = _require_daft(plan, input_handle, "lance")
        return self._descriptor.model_copy(deep=True)

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        input_handle = _require_daft(plan, input_handle, "lance")
        input_handle.dataframe.write_lance(
            daft_path(plan.target),
            mode=cast(
                Literal["create", "append", "overwrite", "merge"], plan.mode.value
            ),
            io_config=daft_io_config(
                context.runtime_options,
                required=plan.target.lower().startswith("s3://"),
            ),
        )
        return write_receipt(
            plan=plan,
            binding_id=self.binding_id,
            native_api="daft.DataFrame.write_lance",
        )


def _clickhouse_target(target: str) -> tuple[str, int, str, str]:
    parsed = urlsplit(target)
    if (
        parsed.scheme.lower() != "clickhouse"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
    ):
        raise WriteCapabilityError(
            "ClickHouse write target must be clickhouse://host:port/database/table "
            "without credentials, query, or fragment"
        )
    try:
        parsed_port = parsed.port
    except ValueError:
        raise WriteCapabilityError("ClickHouse write target port is invalid") from None
    port = 8123 if parsed_port is None else parsed_port
    if port < 1 or port > 65_535:
        raise WriteCapabilityError("ClickHouse write target port is invalid")
    raw_path = parsed.path.split("/")
    if len(raw_path) != 3 or raw_path[0] or not raw_path[1] or not raw_path[2]:
        raise WriteCapabilityError(
            "ClickHouse write target must contain exactly database/table"
        )
    path = tuple(unquote(value) for value in raw_path[1:])
    if any("/" in value for value in path):
        raise WriteCapabilityError(
            "ClickHouse write target database/table must be path segments"
        )
    return parsed.hostname, port, path[0], path[1]


def _doris_target(target: str) -> tuple[str, int, str, str]:
    parsed = urlsplit(target)
    if (
        parsed.scheme.lower() != "doris"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
    ):
        raise WriteCapabilityError(
            "Doris write target must be doris://host:port/database/table without credentials, query, or fragment"
        )
    try:
        port = 8030 if parsed.port is None else parsed.port
    except ValueError:
        raise WriteCapabilityError("Doris write target port is invalid") from None
    parts = parsed.path.split("/")
    if not 1 <= port <= 65535 or len(parts) != 3 or not parts[1] or not parts[2]:
        raise WriteCapabilityError(
            "Doris write target must contain a valid port and database/table"
        )
    database, table = (unquote(part) for part in parts[1:])
    if any("/" in part or "\x00" in part for part in (database, table)):
        raise WriteCapabilityError(
            "Doris write target database/table must be path segments"
        )
    return parsed.hostname, port, database, table


def _doris_username(runtime_options: Mapping[str, Any]) -> str:
    user = runtime_options.get("user", os.getenv("TRIBUTO_DORIS_USER", "root"))
    if not isinstance(user, str) or not user:
        raise WriteCapabilityError("Doris write user must be a non-empty string")
    return user


def _doris_password_env(runtime_options: Mapping[str, Any]) -> str | None:
    reference = runtime_options.get("credential_ref")
    if reference is None:
        return (
            "TRIBUTO_DORIS_PASSWORD" if "TRIBUTO_DORIS_PASSWORD" in os.environ else None
        )
    if not isinstance(reference, str) or not reference.startswith("env://"):
        raise WriteCapabilityError("Doris write credential_ref must use env://VARIABLE")
    name = reference.removeprefix("env://")
    if _ENV_NAME.fullmatch(name) is None:
        raise WriteCapabilityError(
            "Doris write credential_ref must name a portable environment variable"
        )
    return name


def _clickhouse_password_env(runtime_options: Mapping[str, Any]) -> str | None:
    reference = runtime_options.get("credential_ref")
    if reference is None:
        return (
            "TRIBUTO_CLICKHOUSE_PASSWORD"
            if "TRIBUTO_CLICKHOUSE_PASSWORD" in os.environ
            else None
        )
    if not isinstance(reference, str) or not reference.startswith("env://"):
        raise WriteCapabilityError(
            "ClickHouse write credential_ref must use env://VARIABLE"
        )
    name = reference.removeprefix("env://")
    if _ENV_NAME.fullmatch(name) is None:
        raise WriteCapabilityError(
            "ClickHouse write credential_ref must name a portable environment variable"
        )
    return name


def _clickhouse_username(runtime_options: Mapping[str, Any]) -> str:
    configured = runtime_options.get("user")
    if configured is None:
        configured = os.getenv("TRIBUTO_CLICKHOUSE_USER", "default")
    if not isinstance(configured, str) or not configured:
        raise WriteCapabilityError("ClickHouse write user must be a non-empty string")
    return configured


def _require_ray(
    plan: LogicalWritePlan, handle: WriteHandle, target_kind: str
) -> RayDataHandle:
    if plan.engine_id != "tributo.ray_data" or plan.target_kind != target_kind:
        raise WriteCapabilityError("Ray write binding received an incompatible plan")
    if not isinstance(handle, RayDataHandle):
        raise WriteCapabilityError("Ray write binding requires a RayDataHandle")
    return handle


def _require_daft(
    plan: LogicalWritePlan, handle: WriteHandle, target_kind: str
) -> DaftDataFrameHandle:
    if plan.engine_id != "tributo.daft" or plan.target_kind != target_kind:
        raise WriteCapabilityError("Daft write binding received an incompatible plan")
    if not isinstance(handle, DaftDataFrameHandle):
        raise WriteCapabilityError("Daft write binding requires a DaftDataFrameHandle")
    return handle
