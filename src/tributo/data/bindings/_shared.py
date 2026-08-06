"""Shared parameter mapping for built-in engine bindings.

This module may construct engine configuration objects, but it must never
discover files, decode formats, execute SQL, or move row batches.
"""

from __future__ import annotations

from typing import Any, Mapping

import pyarrow as pa

from tributo._common.storage_profiles import StorageProfile, StorageProfileResolver
from tributo.data.base import S3Config
from tributo.data.engine_binding import classify_transform_decisions
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


def runtime_s3_profile(runtime_options: Mapping[str, Any]) -> StorageProfile:
    """Resolve the shared storage profile, then overlay source-local settings."""
    runtime_profile = runtime_options.get("s3_profile")
    if runtime_profile is not None and not isinstance(runtime_profile, StorageProfile):
        raise JobConfigurationError("runtime option 's3_profile' must be resolved")
    profile = runtime_profile or StorageProfileResolver().resolve(None)
    raw = runtime_options.get("s3")
    if raw is None:
        return profile
    if isinstance(raw, S3Config):
        source = raw
    elif isinstance(raw, Mapping):
        source = S3Config.model_validate(dict(raw))
    else:
        raise JobConfigurationError("runtime option 's3' must be an S3Config")
    return StorageProfile(
        endpoint=source.endpoint or profile.endpoint,
        region=source.region or profile.region,
        access_key_id=source.access_key_id or profile.access_key_id,
        secret_access_key=source.secret_access_key or profile.secret_access_key,
        use_ssl=profile.use_ssl,
        path_style=profile.path_style,
        profile_name=profile.profile_name,
    )


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
