"""Shared parameter mapping for built-in engine bindings.

This module may construct engine configuration objects, but it must never
discover files, decode formats, execute SQL, or move row batches.
"""

from __future__ import annotations

from typing import Any, Mapping

import pyarrow as pa

from tributo._common.storage_profiles import StorageProfile, StorageProfileResolver
from tributo.data._s3 import (
    ICEBERG_FILE_IO_PROPERTY,
    PYARROW_ICEBERG_FILE_IO,
    merge_iceberg_properties,
)
from tributo.data.base import S3Config
from tributo.data.engine_binding import BindingStageError, classify_transform_decisions
from tributo.data.ingestion import TransformDecision
from tributo.data.scan_plan import FileScan, LogicalScanPlan
from tributo.data.transform_ir import TransformPipeline
from tributo.exceptions import JobConfigurationError


def require_file_scan(
    plan: LogicalScanPlan,
    *,
    connector_id: str,
    filesystem_ids: frozenset[str],
) -> FileScan:
    """Narrow a Binding request to one declared file-format scope."""
    if not isinstance(plan, FileScan) or plan.connector_id != connector_id:
        raise JobConfigurationError(
            f"{connector_id} binding requires a {connector_id} FileScan"
        )
    if plan.filesystem_id not in filesystem_ids:
        raise JobConfigurationError(
            f"{connector_id} binding does not support filesystem {plan.filesystem_id!r}"
        )
    return plan


def require_parquet_file_scan(plan: LogicalScanPlan) -> FileScan:
    """Narrow a Binding request to the built-in Parquet file-plan shape."""
    return require_file_scan(
        plan,
        connector_id="parquet",
        filesystem_ids=frozenset({"local", "file", "s3"}),
    )


def _runtime_s3_source(runtime_options: Mapping[str, Any]) -> S3Config | None:
    """Validate and return source-local S3 settings, when present."""
    raw = runtime_options.get("s3")
    if raw is None:
        return None
    if isinstance(raw, S3Config):
        return raw
    if isinstance(raw, Mapping):
        return S3Config.model_validate(dict(raw))
    raise JobConfigurationError("runtime option 's3' must be an S3Config")


def _runtime_storage_profile(
    runtime_options: Mapping[str, Any],
) -> StorageProfile | None:
    """Validate the profile resolved by the ingestion Gateway."""
    runtime_profile = runtime_options.get("s3_profile")
    if runtime_profile is not None and not isinstance(runtime_profile, StorageProfile):
        raise JobConfigurationError("runtime option 's3_profile' must be resolved")
    return runtime_profile


def runtime_s3_profile(runtime_options: Mapping[str, Any]) -> StorageProfile:
    """Resolve the shared storage profile, then overlay source-local settings."""
    profile = _runtime_storage_profile(runtime_options)
    profile = profile or StorageProfileResolver().resolve(None)
    source = _runtime_s3_source(runtime_options)
    if source is None:
        return profile
    source_has_credentials = bool(source.access_key_id or source.secret_access_key)
    return StorageProfile(
        endpoint=source.endpoint or profile.endpoint,
        region=source.region or profile.region,
        access_key_id=(
            source.access_key_id if source_has_credentials else profile.access_key_id
        ),
        secret_access_key=(
            source.secret_access_key
            if source_has_credentials
            else profile.secret_access_key
        ),
        use_ssl=profile.use_ssl,
        path_style=profile.path_style,
        profile_name=None if source_has_credentials else profile.profile_name,
    )


def iceberg_catalog_properties(runtime_options: Mapping[str, Any]) -> dict[str, str]:
    """Resolve catalog properties with source-aware S3 precedence."""
    raw_properties = runtime_options.get("catalog_properties", {})
    if not isinstance(raw_properties, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_properties.items()
    ):
        raise JobConfigurationError(
            "runtime option 'catalog_properties' must map strings to strings"
        )
    file_io = raw_properties.get(ICEBERG_FILE_IO_PROPERTY)
    if file_io is not None and file_io != PYARROW_ICEBERG_FILE_IO:
        raise BindingStageError.framework_diagnostic(
            "build_native_plan",
            error_type=JobConfigurationError,
            diagnostic_code="unsupported_iceberg_file_io",
            diagnostic=(
                "Built-in Iceberg bindings require PyArrowFileIO; "
                "other py-io-impl values are unsupported"
            ),
        )
    return merge_iceberg_properties(
        raw_properties,
        profile=_runtime_storage_profile(runtime_options),
        source=_runtime_s3_source(runtime_options),
    )


def parse_iceberg_row_filter(value: Any) -> Any | None:
    """Validate the shared filter subset with PyIceberg's public parser."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise JobConfigurationError("Iceberg row_filter must be a non-empty string")
    from pyiceberg.expressions.parser import ParseException, parse

    try:
        return parse(value)
    except ParseException:
        raise JobConfigurationError(
            "Iceberg row_filter must use the supported PyIceberg expression syntax"
        ) from None


def arrow_schema(value: Any) -> pa.Schema:
    """Extract an Arrow schema from a documented engine schema wrapper."""
    if isinstance(value, pa.Schema):
        return value
    base_schema = getattr(value, "base_schema", None)
    if isinstance(base_schema, pa.Schema):
        return base_schema
    to_arrow = getattr(value, "to_pyarrow_schema", None)
    if callable(to_arrow):
        converted = to_arrow()
        if isinstance(converted, pa.Schema):
            return converted
    raise JobConfigurationError(
        f"Engine returned unsupported schema type {type(value).__name__!r}"
    )


def canonical_engine_schema(value: Any) -> pa.Schema:
    """Return a conservative engine-neutral Arrow schema.

    Ray Data and Daft can expose different nullability and private metadata
    for the same Parquet schema. Neither distinction is portable across the
    two public readers, so the ingestion contract retains field order/types,
    widens fields to nullable, and drops engine-specific metadata.
    """
    schema = arrow_schema(value)
    return pa.schema([_canonical_arrow_field(field) for field in schema])


def _canonical_arrow_field(field: pa.Field) -> pa.Field:
    return pa.field(
        field.name,
        _canonical_arrow_type(field.type),
        nullable=True,
    )


def _canonical_arrow_type(data_type: pa.DataType) -> pa.DataType:
    """Normalize Arrow offset-width choices that do not change data semantics."""
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return pa.string()
    if pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        return pa.binary()
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return pa.list_(_canonical_arrow_field(data_type.value_field))
    if pa.types.is_fixed_size_list(data_type):
        return pa.list_(
            _canonical_arrow_field(data_type.value_field), data_type.list_size
        )
    if pa.types.is_struct(data_type):
        return pa.struct([_canonical_arrow_field(field) for field in data_type])
    if pa.types.is_map(data_type):
        return pa.map_(
            _canonical_arrow_type(data_type.key_type),
            _canonical_arrow_type(data_type.item_type),
            keys_sorted=data_type.keys_sorted,
        )
    if pa.types.is_dictionary(data_type):
        return _canonical_arrow_type(data_type.value_type)
    return data_type


def residual_decisions(pipeline: TransformPipeline) -> tuple[TransformDecision, ...]:
    """Record ordered transforms conservatively as engine-level residuals."""
    return classify_transform_decisions(pipeline)
