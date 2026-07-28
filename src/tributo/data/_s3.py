"""Unified S3 authentication resolution.

Supports two environment variable naming conventions:
- Tributo standard: ``S3_ENDPOINT``, ``AWS_ACCESS_KEY_ID``,
  ``AWS_SECRET_ACCESS_KEY``, ``AWS_REGION``
- Lance / AWS standard: ``AWS_ENDPOINT_URL`` (equivalent to ``S3_ENDPOINT``)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tributo.data.base import S3Config


def resolve_endpoint(explicit: str | None = None) -> str | None:
    """Resolve S3 endpoint, supporting ``S3_ENDPOINT`` and ``AWS_ENDPOINT_URL``."""
    return (
        explicit or os.environ.get("S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    )


def resolve_access_key_id(explicit: str | None = None) -> str | None:
    """Resolve AWS Access Key ID."""
    return explicit or os.environ.get("AWS_ACCESS_KEY_ID")


def resolve_secret_access_key(explicit: str | None = None) -> str | None:
    """Resolve AWS Secret Access Key."""
    return explicit or os.environ.get("AWS_SECRET_ACCESS_KEY")


def resolve_region(explicit: str | None = None) -> str:
    """Resolve AWS Region, defaulting to ``"us-east-1"``."""
    return explicit or os.environ.get("AWS_REGION", "us-east-1")


def _resolve_s3_fields(
    s3_config: S3Config | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve (endpoint, key_id, access_key, region) from config + env.

    Shared helper that eliminates the triple copy-paste of the
    ``s3_config.field if s3_config else None`` pattern across the three
    ``to_*`` backends.  Does **not** apply a ``"us-east-1"`` default for
    region — callers that need one (only ``to_pyarrow_s3_kwargs``) should
    apply it themselves.
    """
    return (
        resolve_endpoint(s3_config.endpoint if s3_config else None),
        resolve_access_key_id(s3_config.access_key_id if s3_config else None),
        resolve_secret_access_key(s3_config.secret_access_key if s3_config else None),
        (s3_config.region if s3_config else None) or os.environ.get("AWS_REGION"),
    )


def to_pyarrow_s3_kwargs(s3_config: S3Config | None = None) -> dict[str, str]:
    """Convert ``S3Config`` to kwargs for ``pyarrow.fs.S3FileSystem``.

    Args:
        s3_config: S3 configuration.  When ``None``, values are resolved
            from environment variables.

    Returns:
        Keyword arguments for ``pyarrow.fs.S3FileSystem``.
    """
    endpoint, key_id, access_key, region = _resolve_s3_fields(s3_config)
    # Only PyArrow needs a region default; the other backends omit the field.
    if not region:
        region = "us-east-1"

    kwargs: dict[str, str] = {}
    if key_id:
        kwargs["access_key"] = key_id
    if access_key:
        kwargs["secret_key"] = access_key
    if region:
        kwargs["region"] = region
    if endpoint:
        raw = endpoint.replace("http://", "").replace("https://", "")
        kwargs["endpoint_override"] = raw
        if endpoint.startswith("http://"):
            kwargs["scheme"] = "http"
    return kwargs


def to_lance_storage_options(
    s3_config: S3Config | None = None,
) -> dict[str, str] | None:
    """Convert ``S3Config`` to Lance ``storage_options`` dict.

    Args:
        s3_config: S3 configuration.  When ``None``, values are resolved
            from environment variables.

    Returns:
        Lance ``storage_options`` dict, or ``None`` if no S3 config is
        available.
    """
    endpoint, key_id, access_key, region = _resolve_s3_fields(s3_config)

    opts: dict[str, str] = {}
    if key_id:
        opts["access_key_id"] = key_id
    if access_key:
        opts["secret_access_key"] = access_key
    if endpoint:
        opts["endpoint"] = endpoint
        if endpoint.startswith("http://"):
            opts["allow_http"] = "true"
    if region:
        opts["region"] = region
    return opts if opts else None


def to_iceberg_properties(s3_config: S3Config | None = None) -> dict[str, str]:
    """Convert ``S3Config`` to PyIceberg catalog S3 properties.

    Consistent with ``to_lance_storage_options``: properties without
    explicit configuration are omitted so PyIceberg uses its internal
    defaults.

    Args:
        s3_config: S3 configuration.  When ``None``, values are resolved
            from environment variables.

    Returns:
        PyIceberg S3 property dict (``s3.endpoint``, ``s3.access-key-id``,
        etc.).
    """
    endpoint, key_id, access_key, region = _resolve_s3_fields(s3_config)

    props: dict[str, str] = {}
    if key_id:
        props["s3.access-key-id"] = key_id
    if access_key:
        props["s3.secret-access-key"] = access_key
    if endpoint:
        props["s3.endpoint"] = endpoint
    if region:
        props["s3.region"] = region
    return props
