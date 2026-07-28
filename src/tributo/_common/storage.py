"""Unified storage backend abstraction.

Provides cross-module shared S3/local file operation utilities,
eliminating the duplicated ``if path.startswith("s3://"): ... else: ...``
branching logic found in each module.

Design principles (inspired by Ray _common/utils.py):
- All S3 credential resolution goes through ``resolve_*`` functions.
- All boto3 client creation goes through ``get_boto3_client``.
- Higher-level functions (``write_json``, ``write_bytes``, ``download_from_s3``)
  compose these building blocks.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from tributo.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)


# ── S3 URL parsing ──


@DeveloperAPI
def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key).

    Args:
        s3_url: S3 URI (e.g., ``s3://my-bucket/path/to/file``).

    Returns:
        ``(bucket, key)`` tuple.

    Raises:
        ValueError: URL does not start with ``s3://``.
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    parts = s3_url[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


# ── S3 credential resolution (compatible with two env var naming conventions) ──


def resolve_endpoint(explicit: str | None = None) -> str | None:
    """Resolve S3 endpoint, compatible with S3_ENDPOINT and AWS_ENDPOINT_URL."""
    return (
        explicit or os.environ.get("S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    )


def resolve_access_key_id(explicit: str | None = None) -> str | None:
    """Resolve AWS Access Key ID."""
    return explicit or os.environ.get("AWS_ACCESS_KEY_ID")


def resolve_secret_access_key(explicit: str | None = None) -> str | None:
    """Resolve AWS Secret Access Key."""
    return explicit or os.environ.get("AWS_SECRET_ACCESS_KEY")


def resolve_region(explicit: str | None = None) -> str | None:
    """Resolve AWS Region (no default; returns None to let boto3 use its default)."""
    return explicit or os.environ.get("AWS_REGION")


# ── boto3 client creation ──


@DeveloperAPI
def get_boto3_client(
    endpoint: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region: str | None = None,
) -> Any:
    """Create a boto3 S3 client with unified credential resolution.

    Priority: explicit parameters > environment variables > boto3 default
    credential chain.

    Args:
        endpoint: S3-compatible endpoint URL (e.g., MinIO).
        access_key_id: AWS Access Key ID.
        secret_access_key: AWS Secret Access Key.
        region: AWS region.

    Returns:
        A boto3 S3 client instance.
    """
    import boto3

    kwargs: dict[str, str] = {}
    resolved_endpoint = resolve_endpoint(endpoint)
    if resolved_endpoint:
        kwargs["endpoint_url"] = resolved_endpoint
    resolved_key_id = resolve_access_key_id(access_key_id)
    if resolved_key_id:
        kwargs["aws_access_key_id"] = resolved_key_id
    resolved_secret = resolve_secret_access_key(secret_access_key)
    if resolved_secret:
        kwargs["aws_secret_access_key"] = resolved_secret
    resolved_region = resolve_region(region)
    if resolved_region:
        kwargs["region_name"] = resolved_region

    return boto3.client("s3", **kwargs)


@DeveloperAPI
def get_boto3_client_from_config(s3_cfg: dict[str, Any] | None = None) -> Any:
    """Create a boto3 S3 client from a config dict.

    Args:
        s3_cfg: S3 config dict with keys access_key_id/secret_access_key/
            endpoint/region.

    Returns:
        A boto3 S3 client instance.
    """
    if s3_cfg is None:
        s3_cfg = {}
    return get_boto3_client(
        endpoint=s3_cfg.get("endpoint"),
        access_key_id=s3_cfg.get("access_key_id"),
        secret_access_key=s3_cfg.get("secret_access_key"),
        region=s3_cfg.get("region"),
    )


# ── Unified write operations ──


@DeveloperAPI
def write_json(
    path: str,
    data: dict,
    *,
    s3_cfg: dict[str, Any] | None = None,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON data to local path or S3.

    Automatically detects the path type:
    - ``s3://`` prefix → writes to S3
    - Otherwise → writes to a local file

    Args:
        path: Target path (local path or s3:// URI).
        data: Dict data to write.
        s3_cfg: S3 config dict (required for S3 paths).
        indent: JSON indentation level.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    if path.startswith("s3://"):
        bucket, key = parse_s3_url(path)
        client = get_boto3_client_from_config(s3_cfg)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=indent, ensure_ascii=ensure_ascii).encode(
                "utf-8"
            ),
        )
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


@DeveloperAPI
def write_bytes(
    path: str,
    data: bytes,
    *,
    s3_cfg: dict[str, Any] | None = None,
) -> None:
    """Write bytes to local path or S3.

    Args:
        path: Target path (local path or s3:// URI).
        data: Bytes to write.
        s3_cfg: S3 config dict (required for S3 paths).
    """
    if path.startswith("s3://"):
        bucket, key = parse_s3_url(path)
        client = get_boto3_client_from_config(s3_cfg)
        client.put_object(Bucket=bucket, Key=key, Body=data)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


@DeveloperAPI
def upload_file(
    local_path: str,
    s3_url: str,
    *,
    s3_cfg: dict[str, Any] | None = None,
) -> None:
    """Upload a local file to S3.

    Args:
        local_path: Local file path.
        s3_url: Target S3 URI.
        s3_cfg: S3 config dict.
    """
    bucket, key = parse_s3_url(s3_url)
    client = get_boto3_client_from_config(s3_cfg)
    client.upload_file(local_path, bucket, key)


# ── S3 download to local ──


@DeveloperAPI
def download_from_s3(
    s3_uri: str,
    *,
    local_dir: str | None = None,
    filename: str | None = None,
    s3_cfg: dict[str, Any] | None = None,
) -> Path:
    """Download a file from S3 to a local temporary directory.

    Args:
        s3_uri: S3 URI (e.g., ``s3://bucket/path/to/model.onnx``).
        local_dir: Local cache directory; uses system temp dir when ``None``.
        filename: Local filename; uses the S3 key's basename when ``None``.
        s3_cfg: S3 config dict.

    Returns:
        Local file path.

    Raises:
        ImportError: boto3 is not installed.
    """
    try:
        import boto3  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "boto3 is required for S3 download. Install with: pip install boto3"
        ) from e

    bucket, key = parse_s3_url(s3_uri)
    if local_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "tributo_cache" / str(uuid.uuid4())
    else:
        cache_dir = Path(local_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = Path(key).name or "downloaded_file"
    local_path = cache_dir / filename

    client = get_boto3_client_from_config(s3_cfg)
    client.download_file(bucket, key, str(local_path))
    logger.info("Downloaded %s → %s", s3_uri, local_path)
    return local_path


@DeveloperAPI
def resolve_to_local(
    uri: str,
    *,
    s3_cfg: dict[str, Any] | None = None,
) -> Path:
    """Resolve a URI to a local path.

    - Local path → returns ``Path`` directly
    - ``s3://`` URI → downloads to a temp dir and returns the local path

    Args:
        uri: Local path or ``s3://`` URI.
        s3_cfg: S3 config dict (required for S3 paths).

    Returns:
        Local file path.
    """
    if uri.startswith("s3://"):
        return download_from_s3(uri, s3_cfg=s3_cfg)
    return Path(uri)
