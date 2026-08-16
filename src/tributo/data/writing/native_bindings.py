"""Built-in Ray and Daft write bindings.

Each binding is deliberately thin: it validates the typed handle, prepares
engine-native options, invokes the corresponding public ``write_*`` API, and
returns a small credential-free receipt.  It never implements file, fragment,
manifest, or snapshot data-plane logic itself.
"""

from __future__ import annotations

from typing import Any, Literal, cast

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
    WriteCapabilityError,
    WriteDescriptor,
    WriteExecutionContext,
    WriteHandle,
    WriteReceipt,
)
from tributo.data.writing.targets import LogicalWritePlan

RAY_ENGINE_VERSION = "2.55.1"
DAFT_ENGINE_VERSION = "0.7.23"


def _ray_descriptor(
    target_kind: str, binding_id: str, capabilities: WriteCapability
) -> WriteDescriptor:
    dependencies: tuple[str, ...] = (
        ("pyiceberg",)
        if target_kind == "iceberg"
        else ("lance-ray", "pylance")
        if target_kind == "lance"
        else ()
    )
    return descriptor(
        engine_id="tributo.ray_data",
        target_kind=target_kind,
        binding_id=binding_id,
        engine_version=RAY_ENGINE_VERSION,
        dependency_distributions=dependencies,
        capabilities=capabilities,
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
_PARQUET_OPTIONS = frozenset({"compression", "min_rows_per_file"})
_ICEBERG_OPTIONS = frozenset({"snapshot_properties"})
_LANCE_OPTIONS = frozenset(
    {"min_rows_per_file", "max_rows_per_file", "data_storage_version"}
)


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
