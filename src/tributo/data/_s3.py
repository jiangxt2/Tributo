"""Unified S3 authentication resolution.

Supports two environment variable naming conventions:
- Tributo standard: ``S3_ENDPOINT``, ``AWS_ACCESS_KEY_ID``,
  ``AWS_SECRET_ACCESS_KEY``, ``AWS_REGION``
- Lance / AWS standard: ``AWS_ENDPOINT_URL`` (equivalent to ``S3_ENDPOINT``)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping, TypedDict

from tributo.exceptions import JobConfigurationError

if TYPE_CHECKING:
    from tributo._common.storage_profiles import StorageProfile
    from tributo.data.base import S3Config

    S3Settings = S3Config | StorageProfile
else:
    S3Settings = Any

_ICEBERG_CREDENTIAL_PROPERTIES = frozenset(
    {
        "s3.access-key-id",
        "s3.secret-access-key",
        "s3.session-token",
        "s3.profile-name",
    }
)
ICEBERG_FILE_IO_PROPERTY = "py-io-impl"
PYARROW_ICEBERG_FILE_IO = "pyiceberg.io.pyarrow.PyArrowFileIO"


class DaftS3Kwargs(TypedDict, total=False):
    """Typed subset passed to Daft's public ``S3Config`` constructor."""

    region_name: str
    endpoint_url: str
    key_id: str
    session_token: str
    access_key: str
    use_ssl: bool
    force_virtual_addressing: bool
    profile_name: str


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
    s3_config: S3Settings | None = None,
    *,
    use_environment: bool = True,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    bool,
    bool,
    str | None,
]:
    """Resolve endpoint, atomic credentials, region, and transport settings.

    Shared helper that eliminates the triple copy-paste of the
    ``s3_config.field if s3_config else None`` pattern across the three
    ``to_*`` backends.  Does **not** apply a ``"us-east-1"`` default for
    region — callers that need one (only ``to_pyarrow_s3_kwargs``) should
    apply it themselves.
    """
    endpoint = s3_config.endpoint if s3_config else None
    key_id = s3_config.access_key_id if s3_config else None
    access_key = s3_config.secret_access_key if s3_config else None
    session_token: str | None = None
    region = s3_config.region if s3_config else None
    profile_name = getattr(s3_config, "profile_name", None)
    if use_environment and not profile_name and not (key_id or access_key):
        key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN")
    return (
        resolve_endpoint(endpoint) if use_environment else endpoint,
        key_id,
        access_key,
        session_token,
        region or (os.environ.get("AWS_REGION") if use_environment else None),
        getattr(s3_config, "use_ssl", True),
        getattr(s3_config, "path_style", False),
        profile_name,
    )


def _named_profile_credentials(
    profile_name: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve one named boto3 profile for backends without profile support."""
    if not profile_name:
        return None, None, None
    try:
        import boto3
        from botocore.exceptions import BotoCoreError
    except ImportError:
        raise JobConfigurationError(
            "Named S3 profiles require the optional S3 dependencies; "
            "install them with: pip install 'tributo[s3]'"
        ) from None

    failed = False
    key_id: str | None = None
    access_key: str | None = None
    session_token: str | None = None
    try:
        credentials = boto3.Session(profile_name=profile_name).get_credentials()
        if credentials is not None:
            frozen = credentials.get_frozen_credentials()
            key_id = frozen.access_key
            access_key = frozen.secret_key
            session_token = frozen.token
    except BotoCoreError:
        failed = True
    if failed or not key_id or not access_key:
        raise JobConfigurationError(
            "Configured S3 profile could not be resolved"
        ) from None
    return key_id, access_key, session_token


def to_pyarrow_s3_kwargs(
    s3_config: S3Settings | None = None,
) -> dict[str, str | bool]:
    """Convert ``S3Config`` to kwargs for ``pyarrow.fs.S3FileSystem``.

    Args:
        s3_config: S3 configuration.  When ``None``, values are resolved
            from environment variables.

    Returns:
        Keyword arguments for ``pyarrow.fs.S3FileSystem``.
    """
    (
        endpoint,
        key_id,
        access_key,
        session_token,
        region,
        use_ssl,
        path_style,
        profile_name,
    ) = _resolve_s3_fields(s3_config)
    if profile_name and not key_id and not access_key:
        key_id, access_key, session_token = _named_profile_credentials(profile_name)
    # Only PyArrow needs a region default; the other backends omit the field.
    if not region:
        region = "us-east-1"

    kwargs: dict[str, str | bool] = {}
    if key_id:
        kwargs["access_key"] = key_id
    if access_key:
        kwargs["secret_key"] = access_key
    if session_token:
        kwargs["session_token"] = session_token
    if region:
        kwargs["region"] = region
    if endpoint:
        raw = endpoint.replace("http://", "").replace("https://", "")
        kwargs["endpoint_override"] = raw
        if endpoint.startswith("http://") or not use_ssl:
            kwargs["scheme"] = "http"
    if path_style:
        kwargs["force_virtual_addressing"] = False
    return kwargs


def to_daft_s3_kwargs(s3_config: S3Settings | None = None) -> DaftS3Kwargs:
    """Convert ``S3Config`` and environment fallbacks to Daft S3 names."""
    (
        endpoint,
        key_id,
        access_key,
        session_token,
        region,
        use_ssl,
        path_style,
        profile_name,
    ) = _resolve_s3_fields(s3_config)
    kwargs: DaftS3Kwargs = {}
    if region:
        kwargs["region_name"] = region
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if key_id:
        kwargs["key_id"] = key_id
    if access_key:
        kwargs["access_key"] = access_key
    if session_token:
        kwargs["session_token"] = session_token
    if (endpoint and endpoint.startswith("http://")) or not use_ssl:
        kwargs["use_ssl"] = False
    if path_style:
        kwargs["force_virtual_addressing"] = False
    if profile_name and not key_id and not access_key:
        kwargs["profile_name"] = profile_name
    return kwargs


def to_lance_storage_options(
    s3_config: S3Settings | None = None,
) -> dict[str, str] | None:
    """Convert ``S3Config`` to Lance ``storage_options`` dict.

    Args:
        s3_config: S3 configuration.  When ``None``, values are resolved
            from environment variables.

    Returns:
        Lance ``storage_options`` dict, or ``None`` if no S3 config is
        available.
    """
    (
        endpoint,
        key_id,
        access_key,
        session_token,
        region,
        use_ssl,
        path_style,
        profile_name,
    ) = _resolve_s3_fields(s3_config)
    if profile_name and not key_id and not access_key:
        key_id, access_key, session_token = _named_profile_credentials(profile_name)

    opts: dict[str, str] = {}
    if key_id:
        opts["access_key_id"] = key_id
    if access_key:
        opts["secret_access_key"] = access_key
    if session_token:
        opts["session_token"] = session_token
    if endpoint:
        if not use_ssl and endpoint.startswith("https://"):
            endpoint = "http://" + endpoint[len("https://") :]
        opts["endpoint"] = endpoint
        if endpoint.startswith("http://") or not use_ssl:
            opts["allow_http"] = "true"
    if region:
        opts["region"] = region
    if path_style:
        opts["virtual_hosted_style_request"] = "false"
    return opts if opts else None


def to_iceberg_properties(
    s3_config: S3Settings | None = None,
    *,
    use_environment: bool = True,
) -> dict[str, str]:
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
    (
        endpoint,
        key_id,
        access_key,
        session_token,
        region,
        use_ssl,
        path_style,
        profile_name,
    ) = _resolve_s3_fields(s3_config, use_environment=use_environment)
    if profile_name and not key_id and not access_key:
        key_id, access_key, session_token = _named_profile_credentials(profile_name)

    props: dict[str, str] = {}
    if key_id:
        props["s3.access-key-id"] = key_id
    if access_key:
        props["s3.secret-access-key"] = access_key
    if session_token:
        props["s3.session-token"] = session_token
    if endpoint:
        if not use_ssl and endpoint.startswith("https://"):
            endpoint = "http://" + endpoint[len("https://") :]
        props["s3.endpoint"] = endpoint
    if region:
        props["s3.region"] = region
    if profile_name:
        props["s3.profile-name"] = profile_name
    if path_style:
        props["s3.force-virtual-addressing"] = "false"
    return props


def merge_iceberg_properties(
    catalog_properties: Mapping[str, str],
    *,
    profile: S3Settings | None = None,
    source: S3Settings | None = None,
) -> dict[str, str]:
    """Merge Iceberg S3 settings using explicit, source-aware precedence.

    Environment values provide defaults, a named runtime profile overrides those
    defaults, explicit catalog properties override both, and source-local S3
    fields are the most specific values.
    """
    merged = to_iceberg_properties(profile)
    if _ICEBERG_CREDENTIAL_PROPERTIES.intersection(catalog_properties):
        for key in _ICEBERG_CREDENTIAL_PROPERTIES:
            merged.pop(key, None)
    merged.update(catalog_properties)
    if source is not None:
        source_properties = to_iceberg_properties(source, use_environment=False)
        if _ICEBERG_CREDENTIAL_PROPERTIES.intersection(source_properties):
            for key in _ICEBERG_CREDENTIAL_PROPERTIES:
                merged.pop(key, None)
        merged.update(source_properties)
    _materialize_iceberg_profile_credentials(merged)
    file_io = merged.setdefault(
        ICEBERG_FILE_IO_PROPERTY,
        PYARROW_ICEBERG_FILE_IO,
    )
    if file_io != PYARROW_ICEBERG_FILE_IO:
        raise JobConfigurationError(
            "Built-in Iceberg bindings require PyArrowFileIO; "
            "other py-io-impl values are unsupported"
        )
    return merged


def _materialize_iceberg_profile_credentials(properties: dict[str, str]) -> None:
    """Resolve a profile for PyArrowFileIO without mixing credential layers."""
    profile_name = properties.get("s3.profile-name")
    if not profile_name or any(
        properties.get(key)
        for key in (
            "s3.access-key-id",
            "s3.secret-access-key",
            "s3.session-token",
        )
    ):
        return
    key_id, access_key, session_token = _named_profile_credentials(profile_name)
    if key_id:
        properties["s3.access-key-id"] = key_id
    if access_key:
        properties["s3.secret-access-key"] = access_key
    if session_token:
        properties["s3.session-token"] = session_token
