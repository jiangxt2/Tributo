"""Shared helpers for thin Ray and Daft native write bindings.

This module prepares engine-native configuration and target metadata only.  It
does not write files, construct fragments, or commit table snapshots; those
operations remain owned by the selected Ray or Daft writer API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from tributo.data._s3 import (
    merge_iceberg_properties,
    to_daft_s3_kwargs,
    to_lance_storage_options,
    to_pyarrow_s3_kwargs,
)
from tributo.data.contracts.modes import WriteMode
from tributo.data.runtime_credentials import coerce_s3_runtime
from tributo.data.writing.contracts import (
    WriteCapabilityError,
    WriteDescriptor,
    WriteReceipt,
)


def descriptor(
    *,
    engine_id: str,
    target_kind: str,
    binding_id: str,
    engine_version: str,
    dependency_distributions: tuple[str, ...] = (),
    capabilities: Any,
    installation_hint: str | None = None,
) -> WriteDescriptor:
    """Build the stable descriptor shared by ``describe`` and registration."""
    from tributo import __version__

    return WriteDescriptor(
        engine_id=engine_id,
        target_kind=target_kind,
        binding_id=binding_id,
        engine_version_spec=f"=={engine_version}",
        binding_distribution="tributo",
        binding_distribution_version=__version__,
        dependency_distributions=dependency_distributions,
        capabilities=capabilities,
        installation_hint=installation_hint,
    )


def runtime_s3(runtime_options: Mapping[str, Any]) -> Any | None:
    """Return the approved native S3 runtime object, if one was supplied."""
    value = runtime_options.get("s3")
    if value is None:
        return None
    try:
        return coerce_s3_runtime(value)
    except ValueError:
        raise WriteCapabilityError("runtime s3 configuration is invalid") from None


def is_s3_target(target: str) -> bool:
    """Return whether a target is an S3 URI."""
    return urlsplit(target).scheme.lower() == "s3"


def native_path(target: str) -> str:
    """Convert a URI target to the path form expected by Ray file writers."""
    parsed = urlsplit(target)
    if parsed.scheme.lower() == "s3":
        return f"{parsed.netloc}{parsed.path}".lstrip("/")
    if parsed.scheme.lower() == "file":
        return unquote(parsed.path)
    return target


def daft_path(target: str) -> str:
    """Preserve S3 URIs for Daft and decode local file URIs."""
    parsed = urlsplit(target)
    if parsed.scheme.lower() == "file":
        return unquote(parsed.path)
    return target


def lance_path(target: str) -> str:
    """Preserve object-store URIs and decode local file URIs for Lance."""
    parsed = urlsplit(target)
    if parsed.scheme.lower() == "file":
        return unquote(parsed.path)
    return target


def ray_filesystem(
    runtime_options: Mapping[str, Any], *, required: bool = False
) -> Any | None:
    """Build Ray's accepted PyArrow filesystem for an S3 target."""
    if runtime_s3(runtime_options) is None and not required:
        return None
    import pyarrow.fs as pafs

    return pafs.S3FileSystem(**to_pyarrow_s3_kwargs(runtime_s3(runtime_options)))


def daft_io_config(
    runtime_options: Mapping[str, Any], *, required: bool = False
) -> Any | None:
    """Build Daft's public IOConfig for an S3 target."""
    if runtime_s3(runtime_options) is None and not required:
        return None
    import daft

    return daft.io.IOConfig(
        s3=daft.io.S3Config(**to_daft_s3_kwargs(runtime_s3(runtime_options)))
    )


def lance_storage_options(runtime_options: Mapping[str, Any]) -> dict[str, str] | None:
    """Build native Lance storage options from the approved runtime object."""
    return to_lance_storage_options(runtime_s3(runtime_options))


def arrow_schema(input_handle: Any, engine_id: str) -> Any:
    """Read only the input schema needed for a missing table preflight."""
    if engine_id == "tributo.ray_data":
        schema = input_handle.dataset.schema()
        return getattr(schema, "base_schema", schema)
    schema = input_handle.dataframe.schema()
    return schema.to_pyarrow_schema()


def iceberg_context(
    runtime_options: Mapping[str, Any], target: str
) -> tuple[str, str, dict[str, str]]:
    """Resolve the catalog control-plane inputs without writing data."""
    raw_properties = runtime_options.get("catalog_properties", {})
    if not isinstance(raw_properties, Mapping):
        raise WriteCapabilityError("catalog_properties must be a mapping")
    properties = {str(key): str(value) for key, value in raw_properties.items()}
    catalog_name = str(runtime_options.get("catalog_name") or "default")
    table_identifier = str(runtime_options.get("table_identifier") or target)
    merged = merge_iceberg_properties(properties, source=runtime_s3(runtime_options))
    return catalog_name, table_identifier, merged


def ensure_iceberg_table(
    input_handle: Any,
    engine_id: str,
    runtime_options: Mapping[str, Any],
    target: str,
) -> tuple[Any, str, dict[str, str]]:
    """Load or create an Iceberg table using catalog metadata only.

    Creating the table is a control-plane operation.  Data files, manifests,
    snapshots, and commits are still produced exclusively by the engine-native
    writer that consumes the returned table/catalog configuration.
    """
    from pyiceberg.catalog import load_catalog
    from pyiceberg.exceptions import NoSuchTableError

    catalog_name, table_identifier, properties = iceberg_context(
        runtime_options, target
    )
    catalog = load_catalog(catalog_name, **properties)
    try:
        table = catalog.load_table(table_identifier)
    except NoSuchTableError:
        table = catalog.create_table(
            identifier=table_identifier,
            schema=arrow_schema(input_handle, engine_id),
            location=runtime_options.get("location"),
        )
    return table, table_identifier, properties


def write_receipt(
    *,
    plan: Any,
    binding_id: str,
    native_api: str,
) -> WriteReceipt:
    """Create the minimum reliable receipt after a blocking native call."""
    return WriteReceipt(
        request_digest=plan.request_digest,
        engine_id=plan.engine_id,
        binding_id=binding_id,
        target_kind=plan.target_kind,
        target_ref=plan.target,
        mode=plan.mode,
        committed=True,
        diagnostics=(native_api,),
    )


def ray_mode(mode: WriteMode) -> Any:
    """Map the public mode to Ray's public SaveMode enum."""
    from ray.data import SaveMode

    return SaveMode(mode.value)
