"""First-party local and S3 bundle repository adapters.

Responsibilities (and only these):
- Commit assembled bundles atomically to local filesystems or manifest-last
  to S3-compatible object stores.
- Read exact committed manifests and materialize bounded local artifacts.
- Resolve and update storage-level alias v1 documents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Iterator

from tributo._common.storage import (
    get_boto3_client,
    parse_s3_url,
)
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exceptions import BundleCommitBusyError
from tributo.exporting.manifest import ExportManifest, ManifestSchemaRegistry
from tributo.exporting.models import (
    AliasConfig,
    BundleRef,
    FailureInfo,
    LogicalArtifact,
    ResolvedArtifact,
)
from tributo.exporting.repository import (
    AliasUpdate,
    ReaderResourceLimits,
    RepositoryCommit,
    StagedBundle,
    enforce_artifact_limits,
    verify_materialized_artifact,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_LEASE_TTL_SECONDS = 300  # 5 minutes
_PUBLISH_LEASE_WAIT_SECONDS = _LEASE_TTL_SECONDS
_LEASE_CAS_RETRIES = 3
_LEASE_HEARTBEAT_INTERVAL_SECONDS = 30.0
_LEASE_RENEW_RETRIES = 3
_LEASE_STOP_TIMEOUT_SECONDS = 5.0
_ALIAS_MAX_RETRIES = 3
_S3_SINGLE_PUT_MAX_BYTES = 5_000_000_000
_CONTROL_DOCUMENT_MAX_BYTES = 10 * 1024 * 1024


# ── S3 helpers ─────────────────────────────────────────────────────────────────


def _read_local_bytes_bounded(path: Path, limit: int, kind: str) -> bytes:
    """Read at most *limit* bytes, rejecting oversized files before allocation."""
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f"{kind} size {size} exceeds limit {limit}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{kind} size exceeds limit {limit}")
    return data


def _read_s3_body_bounded(response: Any, limit: int, kind: str) -> bytes:
    """Read one S3 response body without trusting optional size metadata."""
    content_length = response.get("ContentLength")
    if isinstance(content_length, int) and content_length > limit:
        raise ValueError(f"{kind} size {content_length} exceeds limit {limit}")
    data: bytes = response["Body"].read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{kind} size exceeds limit {limit}")
    return data


def _decode_json_object(data: bytes, kind: str) -> dict[str, Any]:
    """Decode a bounded UTF-8 JSON control document and require an object."""
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return decoded


def _s3_client_from_profile(
    resolver: StorageProfileResolver, storage_profile: str | None
) -> Any:
    """Create a boto3 S3 client from a storage profile name."""
    profile = resolver.resolve(storage_profile)
    return get_boto3_client(
        endpoint=profile.endpoint,
        access_key_id=profile.access_key_id,
        secret_access_key=profile.secret_access_key,
        region=profile.region,
        use_ssl=profile.use_ssl,
        path_style=profile.path_style,
        profile_name=profile.profile_name,
    )


def _s3_head(client: Any, bucket: str, key: str) -> Any:
    """Return S3 object metadata dict, or None if not found."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        return {
            "content_length": resp.get("ContentLength", 0),
            "etag": resp.get("ETag", "").strip('"'),
            # botocore normalises x-amz-meta-* header keys to Title-Case
            # (e.g. "Tributo-Sha256"); normalise to lowercase for lookup.
            "metadata": {k.lower(): v for k, v in (resp.get("Metadata") or {}).items()},
        }
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return None
        raise


def _s3_put_json(
    client: Any,
    bucket: str,
    key: str,
    data: dict[str, Any],
    *,
    if_none_match: bool = False,
    if_match: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Any:  # botocore returns Any
    """Put a JSON object to S3 with optional conditional headers.

    Returns the response dict.
    """
    extra: dict[str, Any] = {}
    if if_none_match:
        extra["IfNoneMatch"] = "*"
    if if_match is not None:
        # re-add quotes — S3/MinIO requires quoted ETag in If-Match header
        tag = if_match.strip('"')
        extra["IfMatch"] = f'"{tag}"'
    if metadata:
        extra["Metadata"] = metadata

    body = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
        **extra,
    )


def _s3_put_bytes(
    client: Any,
    bucket: str,
    key: str,
    data: bytes,
    *,
    if_none_match: bool = False,
    content_type: str = "application/octet-stream",
    metadata: dict[str, str] | None = None,
    checksum_algorithm: str | None = None,
) -> Any:
    """Put bytes to S3 with optional conditional headers."""
    extra: dict[str, Any] = {}
    if if_none_match:
        extra["IfNoneMatch"] = "*"
    if metadata:
        extra["Metadata"] = metadata
    if checksum_algorithm:
        # Server-side integrity check of the upload payload.
        extra["ChecksumAlgorithm"] = checksum_algorithm

    return client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        **extra,
    )


def _s3_put_stream(
    client: Any,
    bucket: str,
    key: str,
    stream: Any,
    *,
    if_none_match: bool = False,
    metadata: dict[str, str] | None = None,
    checksum_algorithm: str | None = None,
) -> Any:
    """Stream one object without materializing the payload in memory."""
    extra: dict[str, Any] = {}
    if if_none_match:
        extra["IfNoneMatch"] = "*"
    if metadata:
        extra["Metadata"] = metadata
    if checksum_algorithm:
        extra["ChecksumAlgorithm"] = checksum_algorithm
    return client.put_object(
        Bucket=bucket,
        Key=key,
        Body=stream,
        ContentType="application/octet-stream",
        **extra,
    )


def _s3_get_json_with_etag(
    client: Any,
    bucket: str,
    key: str,
    max_bytes: int = _CONTROL_DOCUMENT_MAX_BYTES,
) -> tuple[dict[str, Any], str] | None:
    """Get one JSON object and its ETag from the same S3 object version."""
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        raw = _read_s3_body_bounded(response, max_bytes, "S3 control document")
        return (
            _decode_json_object(raw, "S3 control document"),
            str(response["ETag"]).strip('"'),
        )
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return None
        raise


def _s3_get_bytes(
    client: Any,
    bucket: str,
    key: str,
    max_bytes: int = _CONTROL_DOCUMENT_MAX_BYTES,
) -> bytes | None:
    """Read an object's body as bytes, or None if not found."""
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return _read_s3_body_bounded(resp, max_bytes, "S3 control document")
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def _s3_delete(client: Any, bucket: str, key: str) -> None:
    """Best-effort delete of an S3 object."""
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        logger.debug("Failed to delete s3://%s/%s", bucket, key, exc_info=True)


# ── Local publish ──────────────────────────────────────────────────────────────


def _fsync_dir(path: Path) -> None:
    """fsync directory after writing files (ensures metadata is durable)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass  # Best effort on platforms that don't support dir fsync.


def _copy_artifact_fsync(src: Path, dst: Path, artifact: LogicalArtifact) -> None:
    """Copy only manifest-declared artifact files with per-file fsync."""
    source_root = src.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for artifact_file in artifact.files:
        item = (source_root / artifact_file.relative_path).resolve()
        if not item.is_relative_to(source_root):
            raise ValueError(
                f"Artifact path escapes staging root: {artifact_file.relative_path!r}"
            )
        target = dst / artifact_file.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        with open(target, "rb") as stream:
            os.fsync(stream.fileno())
        _fsync_dir(target.parent)


def _local_identical_manifest(
    final_dir: Path,
    manifest: ExportManifest,
    manifest_bytes: bytes,
) -> bytes | None:
    """Return the existing manifest bytes if *final_dir* holds the same bundle.

    Uses the same logical comparison as S3 so roles, source identity, and
    artifact descriptors cannot silently drift across an idempotent retry.
    Returns None when the directory is missing, has no manifest, or holds
    different content.
    """
    existing_manifest_path = final_dir / "manifest.json"
    if not existing_manifest_path.is_file():
        return None
    try:
        existing_bytes = _read_local_bytes_bounded(
            existing_manifest_path,
            _CONTROL_DOCUMENT_MAX_BYTES,
            "Manifest",
        )
    except OSError:
        return None
    if not _manifest_logically_equal(existing_bytes, manifest_bytes):
        return None
    for artifact in manifest.artifacts:
        verify_materialized_artifact(
            artifact,
            final_dir / "artifacts" / artifact.name,
        )
    return existing_bytes


def _local_publish(
    *,
    bundle_dir: Path,
    staging_root: Path,
    manifest: ExportManifest,
    manifest_bytes: bytes,
    manifest_sha256: str,
    bundle_id: str,
    execution_id: str,
) -> tuple[Path, str, bytes]:
    """Publish bundle to a local directory using atomic rename.

    Returns ``(final_dir, sha256, manifest_bytes)``. On an idempotent retry the
    bytes are from the manifest that originally won the atomic commit.
    """
    final_dir = bundle_dir.resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Check for existing identical bundle (idempotency).
    existing_bytes = _local_identical_manifest(final_dir, manifest, manifest_bytes)
    if existing_bytes is not None:
        logger.info("Bundle %s already exists — idempotent", final_dir)
        return final_dir, hashlib.sha256(existing_bytes).hexdigest(), existing_bytes
    if final_dir.exists():
        raise RuntimeError(
            f"Bundle directory {final_dir} exists but has no manifest or "
            "different content — collision"
        )

    # Create temp directory in the same filesystem as final.
    temp_dir = parent / f".tmp-{bundle_id}-{execution_id}-{uuid.uuid4().hex[:8]}"

    try:
        temp_dir.mkdir(parents=True)

        # Copy artifacts from staging.
        artifacts_dir = temp_dir / "artifacts"
        for artifact in manifest.artifacts:
            artifact_src = staging_root / "nodes" / artifact.name / "artifact"
            artifact_dst = artifacts_dir / artifact.name
            _copy_artifact_fsync(artifact_src, artifact_dst, artifact)

        # Write metadata.
        metadata_dir = temp_dir / "metadata"
        metadata_dir.mkdir(parents=True)

        # Write manifest.
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        with open(manifest_path, "rb") as f:
            os.fsync(f.fileno())

        # fsync temp dir.
        _fsync_dir(temp_dir)

        # Atomic rename.
        try:
            os.rename(str(temp_dir), str(final_dir))
        except OSError:
            # Rename failed — a concurrent writer may have created the
            # final directory.  Re-read and compare instead of falling
            # back to a copy (which could silently overwrite).
            existing_bytes = _local_identical_manifest(
                final_dir, manifest, manifest_bytes
            )
            if existing_bytes is not None:
                logger.info("Bundle %s appeared during rename — idempotent", final_dir)
                return (
                    final_dir,
                    hashlib.sha256(existing_bytes).hexdigest(),
                    existing_bytes,
                )
            raise

        _fsync_dir(parent)
        return final_dir, manifest_sha256, manifest_bytes

    except Exception:
        # Clean up temp on failure.
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# ── S3 publish ─────────────────────────────────────────────────────────────────


def _s3_lease_acquire(
    client: Any,
    bucket: str,
    leases_prefix: str,
    bundle_id: str,
    owner: str,
    ttl: int = _LEASE_TTL_SECONDS,
) -> str:
    """Acquire or renew a lease, retrying conditional-write races."""
    lease_key = f"{leases_prefix}{bundle_id}.json"
    for _ in range(_LEASE_CAS_RETRIES):
        now = time.time()
        lease_data = {
            "owner": owner,
            "created_at": now,
            "expires_at": now + ttl,
            "bundle_id": bundle_id,
        }
        existing_version = _s3_get_json_with_etag(client, bucket, lease_key)
        if existing_version is None:
            try:
                _s3_put_json(
                    client,
                    bucket,
                    lease_key,
                    lease_data,
                    if_none_match=True,
                )
                return lease_key
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "PreconditionFailed":
                    continue
                raise

        existing, existing_etag = existing_version
        existing_owner = existing.get("owner")
        expires_at = float(existing.get("expires_at", 0))
        if existing_owner != owner and now <= expires_at:
            raise BundleCommitBusyError(
                f"Lease {lease_key} held by {existing_owner} until "
                f"epoch {expires_at:.3f}"
            )

        try:
            _s3_put_json(
                client,
                bucket,
                lease_key,
                lease_data,
                if_match=existing_etag,
            )
            if existing_owner != owner:
                logger.info(
                    "Took over expired lease %s (previous owner %s)",
                    lease_key,
                    existing_owner,
                )
            return lease_key
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "PreconditionFailed":
                continue
            raise

    raise BundleCommitBusyError(
        f"Lease {lease_key} changed during {_LEASE_CAS_RETRIES} acquisition attempts"
    )


class _S3LeaseHeartbeat:
    """Renew an owned lease while a blocking object upload is in progress."""

    def __init__(
        self,
        client: Any,
        bucket: str,
        leases_prefix: str,
        bundle_id: str,
        owner: str,
        *,
        interval_seconds: float,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._leases_prefix = leases_prefix
        self._bundle_id = bundle_id
        self._owner = owner
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._failure: Exception | None = None
        self._last_success = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name=f"tributo-s3-lease-{bundle_id}",
            daemon=True,
        )

    def start(self) -> None:
        """Start periodic renewal."""
        self._thread.start()

    def stop(self) -> bool:
        """Request shutdown and report whether the heartbeat thread exited."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=_LEASE_STOP_TIMEOUT_SECONDS)
        return not self._thread.is_alive()

    def renew_now(self) -> None:
        """Synchronously renew, retrying transient storage failures."""
        self._raise_if_failed()
        last_error: Exception | None = None
        for attempt in range(_LEASE_RENEW_RETRIES):
            try:
                self._renew_once()
                return
            except BundleCommitBusyError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < _LEASE_RENEW_RETRIES:
                    time.sleep(0.05 * (attempt + 1))
        assert last_error is not None
        self._mark_failure(last_error)
        raise BundleCommitBusyError(
            f"Unable to prove publish lease ownership for {self._bundle_id}"
        ) from last_error

    def _renew_once(self) -> None:
        """Perform one renewal without holding the heartbeat state lock."""
        try:
            _s3_lease_acquire(
                self._client,
                self._bucket,
                self._leases_prefix,
                self._bundle_id,
                self._owner,
            )
        except BundleCommitBusyError as exc:
            self._mark_failure(exc)
            raise BundleCommitBusyError(
                f"Lost publish lease for {self._bundle_id}"
            ) from exc
        with self._lock:
            self._last_success = time.monotonic()

    def _mark_failure(self, exc: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = exc

    def _raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise BundleCommitBusyError(
                f"Lost publish lease for {self._bundle_id}"
            ) from failure

    @property
    def failed(self) -> bool:
        """Return whether a renewal has failed."""
        with self._lock:
            return self._failure is not None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._renew_once()
            except BundleCommitBusyError:
                logger.warning(
                    "Publish lease ownership was lost for %s",
                    self._bundle_id,
                    exc_info=True,
                )
                self._stop.set()
                return
            except Exception as exc:
                with self._lock:
                    elapsed = time.monotonic() - self._last_success
                if elapsed >= _LEASE_TTL_SECONDS / 2:
                    self._mark_failure(exc)
                    logger.warning(
                        "Publish lease for %s could not be renewed for %.1fs",
                        self._bundle_id,
                        elapsed,
                        exc_info=True,
                    )
                    self._stop.set()
                    return
                logger.warning(
                    "Transient publish lease renewal failure for %s; retrying",
                    self._bundle_id,
                    exc_info=True,
                )


def _s3_lease_release(client: Any, bucket: str, lease_key: str, owner: str) -> None:
    """Release only the lease version still owned by *owner*."""
    lease: dict[str, Any] | None = None
    etag: str | None = None
    try:
        response = client.get_object(Bucket=bucket, Key=lease_key)
        raw = _read_s3_body_bounded(
            response,
            _CONTROL_DOCUMENT_MAX_BYTES,
            "Lease",
        )
        decoded_lease = _decode_json_object(raw, "Lease")
        if decoded_lease.get("owner") != owner:
            return
        lease = decoded_lease
        etag = str(response.get("ETag", "")).strip('"')
        if not etag:
            head = _s3_head(client, bucket, lease_key)
            if head is None:
                return
            etag = head["etag"]
        client.delete_object(
            Bucket=bucket,
            Key=lease_key,
            IfMatch=f'"{etag}"',
        )
    except client.exceptions.ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if (
            error_code in {"NotImplemented", "InvalidRequest"}
            and lease is not None
            and etag is not None
        ):
            expired_lease = dict(lease)
            expired_lease["expires_at"] = 0
            try:
                _s3_put_json(
                    client,
                    bucket,
                    lease_key,
                    expired_lease,
                    if_match=etag,
                )
            except Exception:
                logger.warning(
                    "Could not conditionally expire lease s3://%s/%s",
                    bucket,
                    lease_key,
                    exc_info=True,
                )
            return
        if error_code not in {
            "404",
            "NoSuchKey",
            "PreconditionFailed",
        }:
            logger.warning(
                "Failed to release lease s3://%s/%s",
                bucket,
                lease_key,
                exc_info=True,
            )
    except Exception:
        logger.debug(
            "Failed to release lease s3://%s/%s",
            bucket,
            lease_key,
            exc_info=True,
        )


def _s3_upload_artifact_files(
    client: Any,
    bucket: str,
    prefix: str,
    artifact: LogicalArtifact,
    staging_root: Path,
    *,
    created_keys: list[str] | None = None,
) -> list[str]:
    """Upload all files of a single artifact to S3.

    Returns the list of uploaded S3 keys.
    """
    artifact_src = staging_root / "nodes" / artifact.name / "artifact"
    keys: list[str] = []

    for af in artifact.files:
        src_file = artifact_src / af.relative_path
        key = f"{prefix}artifacts/{artifact.name}/{af.relative_path}"

        # Check for existing identical object.
        existing = _s3_head(client, bucket, key)
        if existing is not None:
            existing_len = existing["content_length"]
            existing_meta = existing.get("metadata", {})
            existing_sha = existing_meta.get("tributo-sha256", "")
            if existing_len == af.size_bytes and existing_sha == af.sha256:
                logger.debug(
                    "Object s3://%s/%s already exists — idempotent", bucket, key
                )
                keys.append(key)
                continue
            else:
                raise RuntimeError(
                    f"Object s3://{bucket}/{key} exists but content differs — collision"
                )

        if af.size_bytes > _S3_SINGLE_PUT_MAX_BYTES:
            raise ValueError(
                f"Artifact file {af.relative_path!r} is {af.size_bytes} bytes; "
                "S3 atomic publication supports files up to 5 GB. Split or "
                "shard larger model weights before export."
            )

        # A conditional single-object write preserves immutable create-only
        # semantics. Multipart completion does not provide a portable
        # If-None-Match contract across supported S3-compatible stores.
        metadata = {"tributo-sha256": af.sha256}
        try:
            with src_file.open("rb") as stream:
                _s3_put_stream(
                    client,
                    bucket,
                    key,
                    stream,
                    if_none_match=True,
                    metadata=metadata,
                    checksum_algorithm="SHA256",
                )
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "PreconditionFailed":
                # Race — re-check.
                existing = _s3_head(client, bucket, key)
                if existing is not None:
                    existing_len = existing["content_length"]
                    existing_meta = existing.get("metadata", {})
                    existing_sha = existing_meta.get("tributo-sha256", "")
                    if existing_len == af.size_bytes and existing_sha == af.sha256:
                        logger.debug(
                            "Object s3://%s/%s created by concurrent upload — idempotent",
                            bucket,
                            key,
                        )
                        keys.append(key)
                        continue
                raise RuntimeError(
                    f"Object s3://{bucket}/{key} created by concurrent upload "
                    f"but content differs — collision"
                ) from exc
            raise
        keys.append(key)
        if created_keys is not None:
            created_keys.append(key)

    return keys


def _manifest_logically_equal(existing_bytes: bytes, new_bytes: bytes) -> bool:
    """Compare two manifests ignoring per-run advisory fields.

    Idempotent retries of the same request rebuild the manifest with fresh
    ``created_at``, node ``duration_ms``, and validation metrics.  The
    logical bundle content (artifacts, tree digests, roles, source info)
    must match.
    """

    def normalize(raw: bytes) -> Any:
        data = json.loads(raw.decode("utf-8"))
        data.pop("created_at", None)
        for node in data.get("execution", {}).get("nodes", []):
            node.pop("duration_ms", None)
        for art in data.get("artifacts", []):
            for v in art.get("validation", []):
                v.pop("metrics", None)
        return data

    try:
        return bool(normalize(existing_bytes) == normalize(new_bytes))
    except (json.JSONDecodeError, TypeError):
        return False


def _s3_existing_manifest_digest(
    client: Any,
    bucket: str,
    manifest_key: str,
    candidate_bytes: bytes,
) -> tuple[str, bytes] | None:
    """Return committed digest and bytes, or fail on a manifest conflict."""
    existing_body = _s3_get_bytes(client, bucket, manifest_key)
    if existing_body is None:
        return None
    if not _manifest_logically_equal(existing_body, candidate_bytes):
        raise RuntimeError(
            f"Manifest s3://{bucket}/{manifest_key} exists but differs — collision"
        )
    return hashlib.sha256(existing_body).hexdigest(), existing_body


def _s3_verify_committed_artifacts(
    client: Any,
    bucket: str,
    bundle_prefix: str,
    manifest: ExportManifest,
) -> None:
    """Verify that an idempotently reused manifest still anchors its objects."""
    for artifact in manifest.artifacts:
        for artifact_file in artifact.files:
            key = (
                f"{bundle_prefix}artifacts/{artifact.name}/"
                f"{artifact_file.relative_path}"
            )
            head = _s3_head(client, bucket, key)
            if head is None:
                raise RuntimeError(f"Committed object s3://{bucket}/{key} is missing")
            metadata = head.get("metadata", {})
            if (
                head["content_length"] != artifact_file.size_bytes
                or metadata.get("tributo-sha256") != artifact_file.sha256
            ):
                raise RuntimeError(
                    f"Committed object s3://{bucket}/{key} does not match manifest"
                )


def _s3_publish(
    *,
    client: Any,
    bucket: str,
    bundle_prefix: str,
    staging_root: Path,
    manifest: ExportManifest,
    manifest_bytes: bytes,
    manifest_sha256: str,
    bundle_id: str,
    execution_id: str,
) -> Any:
    """Publish bundle to S3 using manifest-last protocol.

    Returns a dict with publish metadata.
    """
    owner = f"{execution_id}-{uuid.uuid4().hex[:8]}"
    # Store-level leases ({store_prefix}.leases/) so the publisher and the
    # orphan GC agree on the same key — a bundle-internal lease would let
    # GC delete an in-flight publish.
    store_prefix = bundle_prefix[: -len(bundle_id) - 1]
    leases_prefix = f"{store_prefix}.leases/"
    manifest_key = f"{bundle_prefix}manifest.json"

    # Phase 1: Acquire an exclusive lease.  A concurrent idempotent caller
    # observes the first caller's manifest instead of uploading in parallel.
    wait_deadline = time.monotonic() + _PUBLISH_LEASE_WAIT_SECONDS
    while True:
        existing_commit = _s3_existing_manifest_digest(
            client, bucket, manifest_key, manifest_bytes
        )
        if existing_commit is not None:
            existing_digest, existing_bytes = existing_commit
            _s3_verify_committed_artifacts(client, bucket, bundle_prefix, manifest)
            return {
                "manifest_key": manifest_key,
                "uploaded_count": 0,
                "manifest_sha256": existing_digest,
                "manifest_bytes": existing_bytes,
            }
        try:
            lease_key = _s3_lease_acquire(
                client,
                bucket,
                leases_prefix,
                bundle_id,
                owner,
            )
            break
        except BundleCommitBusyError:
            if time.monotonic() >= wait_deadline:
                raise
            time.sleep(0.05)
    logger.info("Acquired publish lease %s", lease_key)

    uploaded_keys: list[str] = []
    created_keys: list[str] = []
    manifest_committed = False
    heartbeat = _S3LeaseHeartbeat(
        client,
        bucket,
        leases_prefix,
        bundle_id,
        owner,
        interval_seconds=min(
            _LEASE_HEARTBEAT_INTERVAL_SECONDS,
            _LEASE_TTL_SECONDS / 3,
        ),
    )
    try:
        # Phase 2: renew independently of blocking file uploads so no single
        # artifact can outlive the lease unnoticed.
        heartbeat.start()
        for artifact in manifest.artifacts:
            keys = _s3_upload_artifact_files(
                client,
                bucket,
                bundle_prefix,
                artifact,
                staging_root,
                created_keys=created_keys,
            )
            uploaded_keys.extend(keys)
            if heartbeat.failed:
                heartbeat.renew_now()

        # Phase 3: stop background renewal and synchronously prove ownership
        # immediately before the manifest-last commit.
        if not heartbeat.stop():
            raise BundleCommitBusyError(
                f"Lease heartbeat for {bundle_id} did not stop before commit"
            )
        heartbeat.renew_now()
        existing_commit = _s3_existing_manifest_digest(
            client, bucket, manifest_key, manifest_bytes
        )
        if existing_commit is not None:
            existing_digest, existing_bytes = existing_commit
            _s3_verify_committed_artifacts(client, bucket, bundle_prefix, manifest)
            manifest_sha256 = existing_digest
            manifest_bytes = existing_bytes
            manifest_committed = True
        else:
            metadata = {"tributo-sha256": manifest_sha256}
            try:
                _s3_put_bytes(
                    client,
                    bucket,
                    manifest_key,
                    manifest_bytes,
                    if_none_match=True,
                    content_type="application/json",
                    metadata=metadata,
                )
                logger.info("Published manifest s3://%s/%s", bucket, manifest_key)
                manifest_committed = True
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] != "PreconditionFailed":
                    raise
                existing_commit = _s3_existing_manifest_digest(
                    client, bucket, manifest_key, manifest_bytes
                )
                if existing_commit is None:
                    raise RuntimeError(
                        f"Manifest s3://{bucket}/{manifest_key} disappeared "
                        "after a conditional-write conflict"
                    ) from exc
                manifest_sha256, manifest_bytes = existing_commit
                manifest_committed = True

        return {
            "manifest_key": manifest_key,
            "uploaded_count": len(uploaded_keys),
            # On idempotent retries this is the sha of the manifest already
            # on disk, not the freshly computed one.
            "manifest_sha256": manifest_sha256,
            "manifest_bytes": manifest_bytes,
        }

    except Exception:
        heartbeat_stopped = heartbeat.stop()
        # Cleanup is safe only after confirming that this attempt still owns
        # the lease. Otherwise the objects are left for TTL-based GC so an old
        # publisher cannot delete a new owner's data.
        cleanup_owned = False
        if heartbeat_stopped and not heartbeat.failed:
            try:
                heartbeat.renew_now()
                cleanup_owned = True
            except BundleCommitBusyError:
                pass
        if not manifest_committed and cleanup_owned:
            for key in created_keys:
                _s3_delete(client, bucket, key)
        raise

    finally:
        if heartbeat.stop():
            _s3_lease_release(client, bucket, lease_key, owner)
        else:
            logger.warning(
                "Leaving lease %s to expire because its heartbeat is still active",
                lease_key,
            )


# ── Alias ──────────────────────────────────────────────────────────────────────


def _update_alias_s3(
    client: Any,
    bucket: str,
    alias_key_path: str,
    alias_config: AliasConfig,
    manifest_uri: str,
    manifest_sha256: str,
    bundle_id: str,
    created_at: str,
) -> tuple[str, str | None]:
    """Update or create an alias pointer on S3.

    Returns ``(status, failure_code_or_none)``.
    Status is one of: ``"updated"``, ``"unchanged"``, ``"failed"``.
    """
    alias_data = {
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_sha256,
        "bundle_id": bundle_id,
        "created_at": created_at,
    }

    for _ in range(_ALIAS_MAX_RETRIES):
        existing_version = _s3_get_json_with_etag(client, bucket, alias_key_path)
        if existing_version is None:
            existing = None
            existing_etag = None
        else:
            existing, existing_etag = existing_version

        if existing is None:
            # CAS with expected digest requires the alias to already exist —
            # nothing to compare against, so this is an error.
            if (
                alias_config.policy == "compare_and_swap"
                and alias_config.expected_manifest_sha256
            ):
                return "failed", "ALIAS_NOT_FOUND"
            # CAS create-only (expected_manifest_sha256 is None) — create.
            try:
                _s3_put_json(
                    client, bucket, alias_key_path, alias_data, if_none_match=True
                )
                return "updated", None
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "PreconditionFailed":
                    continue  # Race — retry.
                return "failed", exc.response["Error"]["Code"]

        else:
            # Alias exists — check policy.
            if alias_config.policy == "newer":
                existing_bundle_id = existing.get("bundle_id", "")
                # Compare by created_at then bundle_id.
                # For legacy aliases without created_at (empty string),
                # the empty string sorts before any ISO datetime, so the
                # candidate always wins — which is the desired behaviour.
                existing_created_at = existing.get("created_at", "")
                if existing_created_at > created_at:
                    return "unchanged", None
                if (
                    existing_created_at == created_at
                    and existing_bundle_id >= bundle_id
                ):
                    return "unchanged", None

            elif alias_config.policy == "compare_and_swap":
                if alias_config.expected_manifest_sha256 is None:
                    # Create-only but alias exists.
                    return "failed", "ALIAS_EXISTS"
                current_sha = existing.get("manifest_sha256", "")
                if current_sha != alias_config.expected_manifest_sha256:
                    return "failed", "CAS_MISMATCH"

            # Update with If-Match.
            assert existing_etag is not None
            try:
                _s3_put_json(
                    client,
                    bucket,
                    alias_key_path,
                    alias_data,
                    if_match=existing_etag,
                )
                return "updated", None
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "PreconditionFailed":
                    continue  # Race — retry.
                return "failed", exc.response["Error"]["Code"]

    return "failed", "RETRY_EXHAUSTED"


class LocalBundleRepository:
    """Atomic immutable-bundle repository for local filesystems."""

    api_version: ClassVar[int] = 1
    repository_id: ClassVar[str] = "local-v1"
    schemes: ClassVar[tuple[str, ...]] = ("file",)

    def __init__(self, storage_resolver: StorageProfileResolver | None = None) -> None:
        del storage_resolver

    def commit(
        self,
        staged_bundle: StagedBundle,
        *,
        bundle_uri: str,
        storage_profile: str | None,
    ) -> RepositoryCommit:
        """Commit by same-filesystem atomic rename."""
        del storage_profile
        _validate_staged_bundle(staged_bundle, bundle_uri)
        local_uri = _strip_file_scheme(bundle_uri)
        final_dir, manifest_sha256, committed_manifest_bytes = _local_publish(
            bundle_dir=Path(local_uri) / staged_bundle.bundle_id,
            staging_root=staged_bundle.staging_root,
            manifest=staged_bundle.manifest,
            manifest_bytes=staged_bundle.manifest_bytes,
            manifest_sha256=staged_bundle.manifest_sha256,
            bundle_id=staged_bundle.bundle_id,
            execution_id=staged_bundle.execution_id,
        )
        return RepositoryCommit(
            bundle_ref=BundleRef(
                canonical_uri=staged_bundle.manifest.canonical_uri,
                bundle_id=staged_bundle.bundle_id,
                manifest_sha256=manifest_sha256,
            ),
            manifest_uri=f"{staged_bundle.manifest.canonical_uri}/manifest.json",
            manifest_bytes=committed_manifest_bytes,
            local_bundle_dir=final_dir,
            local_dir_ephemeral=False,
        )

    def read_manifest(
        self,
        bundle_uri: str,
        *,
        storage_profile: str | None,
        max_manifest_bytes: int,
        manifest_registry: ManifestSchemaRegistry,
    ) -> tuple[ExportManifest, bytes]:
        """Read exact local manifest bytes with a size bound."""
        del storage_profile
        path = _resolve_local_manifest_path(Path(_strip_file_scheme(bundle_uri)))
        data = _read_local_bytes_bounded(path, max_manifest_bytes, "Manifest")
        return _parse_manifest(data, manifest_registry), data

    @contextmanager
    def materialize_artifact(
        self,
        bundle_uri: str,
        manifest: ExportManifest,
        artifact: LogicalArtifact,
        *,
        storage_profile: str | None,
        limits: ReaderResourceLimits,
        cache_dir: Path,
    ) -> Iterator[ResolvedArtifact]:
        """Expose an artifact already present on the local filesystem."""
        del storage_profile, cache_dir
        _validate_materialization_request(manifest, artifact, limits)
        manifest_path = _resolve_local_manifest_path(
            Path(_strip_file_scheme(bundle_uri))
        )
        bundle_dir = manifest_path.parent.resolve()
        artifacts_dir = (bundle_dir / "artifacts").resolve()
        artifact_dir = (artifacts_dir / artifact.name).resolve()
        if not artifact_dir.is_relative_to(artifacts_dir):
            raise ValueError(f"Artifact name escapes bundle: {artifact.name!r}")
        verify_materialized_artifact(artifact, artifact_dir)
        yield ResolvedArtifact(descriptor=artifact, root_dir=artifact_dir)


class S3BundleRepository:
    """Manifest-last immutable-bundle repository for S3-compatible stores."""

    api_version: ClassVar[int] = 1
    repository_id: ClassVar[str] = "s3-v1"
    schemes: ClassVar[tuple[str, ...]] = ("s3",)

    def __init__(self, storage_resolver: StorageProfileResolver | None = None) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()

    def commit(
        self,
        staged_bundle: StagedBundle,
        *,
        bundle_uri: str,
        storage_profile: str | None,
    ) -> RepositoryCommit:
        """Commit artifacts first and the manifest last."""
        _validate_staged_bundle(staged_bundle, bundle_uri)
        bucket, store_prefix_raw = parse_s3_url(bundle_uri)
        store_prefix = store_prefix_raw.rstrip("/") + "/" if store_prefix_raw else ""
        bundle_prefix = f"{store_prefix}{staged_bundle.bundle_id}/"
        client = _s3_client_from_profile(self._storage_resolver, storage_profile)
        publish_meta = _s3_publish(
            client=client,
            bucket=bucket,
            bundle_prefix=bundle_prefix,
            staging_root=staged_bundle.staging_root,
            manifest=staged_bundle.manifest,
            manifest_bytes=staged_bundle.manifest_bytes,
            manifest_sha256=staged_bundle.manifest_sha256,
            bundle_id=staged_bundle.bundle_id,
            execution_id=staged_bundle.execution_id,
        )
        canonical_uri = staged_bundle.manifest.canonical_uri
        return RepositoryCommit(
            bundle_ref=BundleRef(
                canonical_uri=canonical_uri,
                bundle_id=staged_bundle.bundle_id,
                manifest_sha256=publish_meta["manifest_sha256"],
            ),
            manifest_uri=f"{canonical_uri}/manifest.json",
            manifest_bytes=publish_meta["manifest_bytes"],
            local_bundle_dir=staged_bundle.staging_root,
            local_dir_ephemeral=True,
        )

    def read_manifest(
        self,
        bundle_uri: str,
        *,
        storage_profile: str | None,
        max_manifest_bytes: int,
        manifest_registry: ManifestSchemaRegistry,
    ) -> tuple[ExportManifest, bytes]:
        """Read exact S3 manifest bytes with a pre-download size bound."""
        bucket, key = _resolve_s3_manifest_key(bundle_uri)
        client = _s3_client_from_profile(self._storage_resolver, storage_profile)
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                raise FileNotFoundError(
                    f"Manifest not found: s3://{bucket}/{key}"
                ) from exc
            raise
        data = _read_s3_body_bounded(response, max_manifest_bytes, "Manifest")
        return _parse_manifest(data, manifest_registry), data

    @contextmanager
    def materialize_artifact(
        self,
        bundle_uri: str,
        manifest: ExportManifest,
        artifact: LogicalArtifact,
        *,
        storage_profile: str | None,
        limits: ReaderResourceLimits,
        cache_dir: Path,
    ) -> Iterator[ResolvedArtifact]:
        """Download an S3 artifact through a locked atomic digest cache."""
        _validate_materialization_request(manifest, artifact, limits)
        bucket, manifest_key = _resolve_s3_manifest_key(bundle_uri)
        bundle_key = manifest_key.removesuffix("manifest.json").rstrip("/")
        artifact_prefix = f"{bundle_key}/artifacts/{artifact.name}/"
        cache_base = cache_dir.resolve()
        cache_base.mkdir(parents=True, exist_ok=True)
        cache_root = (cache_base / artifact.tree_digest).resolve()
        if not cache_root.is_relative_to(cache_base):
            raise ValueError(
                f"Artifact tree digest escapes cache: {artifact.tree_digest!r}"
            )
        client = _s3_client_from_profile(self._storage_resolver, storage_profile)
        lock_path = cache_base / f".{artifact.tree_digest}.lock"
        context_root = Path(tempfile.mkdtemp(prefix="tributo-bundle-artifact-"))
        context_dir = context_root / "artifact"
        temporary_cache: Path | None = None
        invalid_cache: Path | None = None

        try:
            with _exclusive_file_lock(lock_path):
                cache_valid = False
                if cache_root.exists():
                    try:
                        if cache_root.is_dir():
                            verify_materialized_artifact(artifact, cache_root)
                            cache_valid = True
                    except (FileNotFoundError, ValueError, OSError):
                        pass
                    if not cache_valid:
                        invalid_cache = cache_base / (
                            f".invalid-{artifact.tree_digest}-{uuid.uuid4().hex[:8]}"
                        )
                        os.replace(cache_root, invalid_cache)

                if not cache_valid:
                    temporary_cache = Path(
                        tempfile.mkdtemp(
                            prefix=f".download-{artifact.tree_digest}-",
                            dir=cache_base,
                        )
                    ).resolve()
                    for artifact_file in artifact.files:
                        key = f"{artifact_prefix}{artifact_file.relative_path}"
                        local_path = (
                            temporary_cache / artifact_file.relative_path
                        ).resolve()
                        if not local_path.is_relative_to(temporary_cache):
                            raise ValueError(
                                "Artifact file parent escapes cache: "
                                f"{artifact_file.relative_path!r}"
                            )
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        head = client.head_object(Bucket=bucket, Key=key)
                        remote_size = head.get("ContentLength", 0)
                        if remote_size != artifact_file.size_bytes:
                            raise ValueError(
                                f"S3 object {key} size {remote_size} does not "
                                f"match manifest size {artifact_file.size_bytes}"
                            )
                        if remote_size > limits.max_single_file_bytes:
                            raise ValueError(
                                f"S3 object {key} size {remote_size} exceeds limit "
                                f"{limits.max_single_file_bytes}"
                            )
                        client.download_file(bucket, key, str(local_path))

                    verify_materialized_artifact(artifact, temporary_cache)
                    os.replace(temporary_cache, cache_root)
                    temporary_cache = None

                # Consumers receive a private snapshot. Copying under the
                # digest lock prevents another repair from changing the tree
                # during copytree.
                shutil.copytree(cache_root, context_dir)

            yield ResolvedArtifact(descriptor=artifact, root_dir=context_dir)
        finally:
            if temporary_cache is not None:
                shutil.rmtree(temporary_cache, ignore_errors=True)
            if invalid_cache is not None:
                shutil.rmtree(invalid_cache, ignore_errors=True)
            shutil.rmtree(context_root, ignore_errors=True)


class LocalBundleAliasStore:
    """Alias v1 storage for local bundle stores."""

    api_version: ClassVar[int] = 1
    alias_store_id: ClassVar[str] = "local-alias-v1"
    schemes: ClassVar[tuple[str, ...]] = ("file",)

    def __init__(self, storage_resolver: StorageProfileResolver | None = None) -> None:
        del storage_resolver

    def is_alias_uri(self, uri: str) -> bool:
        """Return whether *uri* is a local ``aliases/*.json`` path."""
        path = Path(_strip_file_scheme(uri))
        return path.suffix == ".json" and path.parent.name == "aliases"

    def resolve(
        self,
        alias_uri: str,
        *,
        storage_profile: str | None,
        max_alias_bytes: int,
    ) -> BundleRef:
        """Resolve an existing local alias v1 document."""
        del storage_profile
        path = Path(_strip_file_scheme(alias_uri))
        if not path.is_file():
            raise FileNotFoundError(f"Alias not found: {path}")
        data = _read_local_bytes_bounded(path, max_alias_bytes, "Alias")
        return _bundle_ref_from_alias(
            _decode_json_object(data, f"Alias {alias_uri}"), alias_uri
        )

    def update(
        self,
        *,
        bundle_uri: str,
        alias_config: AliasConfig,
        bundle_ref: BundleRef,
        manifest_uri: str,
        created_at: str,
        storage_profile: str | None,
    ) -> AliasUpdate:
        """Atomically apply local newer or compare-and-set semantics."""
        del storage_profile
        alias_dir = Path(_strip_file_scheme(bundle_uri)) / "aliases"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_path = alias_dir / f"{alias_config.name}.json"
        with _local_alias_lock(alias_dir):
            return _update_local_alias(
                alias_path=alias_path,
                alias_config=alias_config,
                bundle_ref=bundle_ref,
                manifest_uri=manifest_uri,
                created_at=created_at,
            )


class S3BundleAliasStore:
    """Alias v1 storage for S3-compatible bundle stores."""

    api_version: ClassVar[int] = 1
    alias_store_id: ClassVar[str] = "s3-alias-v1"
    schemes: ClassVar[tuple[str, ...]] = ("s3",)

    def __init__(self, storage_resolver: StorageProfileResolver | None = None) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()

    def is_alias_uri(self, uri: str) -> bool:
        """Return whether *uri* is an S3 ``aliases/*.json`` key."""
        _, key = parse_s3_url(uri)
        parts = key.split("/")
        return (
            len(parts) >= 2 and parts[-2] == "aliases" and parts[-1].endswith(".json")
        )

    def resolve(
        self,
        alias_uri: str,
        *,
        storage_profile: str | None,
        max_alias_bytes: int,
    ) -> BundleRef:
        """Resolve an existing S3 alias v1 document."""
        bucket, key = parse_s3_url(alias_uri)
        client = _s3_client_from_profile(self._storage_resolver, storage_profile)
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                raise FileNotFoundError(f"Alias not found: {alias_uri}") from exc
            raise
        data = _read_s3_body_bounded(response, max_alias_bytes, "Alias")
        return _bundle_ref_from_alias(
            _decode_json_object(data, f"Alias {alias_uri}"), alias_uri
        )

    def update(
        self,
        *,
        bundle_uri: str,
        alias_config: AliasConfig,
        bundle_ref: BundleRef,
        manifest_uri: str,
        created_at: str,
        storage_profile: str | None,
    ) -> AliasUpdate:
        """Apply S3 newer or compare-and-set semantics with bounded retries."""
        bucket, prefix_raw = parse_s3_url(bundle_uri)
        prefix = prefix_raw.rstrip("/") + "/" if prefix_raw else ""
        alias_key = f"{prefix}aliases/{alias_config.name}.json"
        alias_uri = f"s3://{bucket}/{alias_key}"
        client = _s3_client_from_profile(self._storage_resolver, storage_profile)
        status, failure_code = _update_alias_s3(
            client=client,
            bucket=bucket,
            alias_key_path=alias_key,
            alias_config=alias_config,
            manifest_uri=manifest_uri,
            manifest_sha256=bundle_ref.manifest_sha256,
            bundle_id=bundle_ref.bundle_id,
            created_at=created_at,
        )
        if failure_code is not None:
            return _failed_alias_update(alias_uri, failure_code)
        return AliasUpdate(alias_uri=alias_uri, status=status)


def _validate_staged_bundle(staged_bundle: StagedBundle, bundle_uri: str) -> None:
    if staged_bundle.bundle_id != staged_bundle.manifest.bundle_id:
        raise ValueError("Staged bundle_id does not match manifest bundle_id")
    if staged_bundle.execution_id != staged_bundle.manifest.execution.execution_id:
        raise ValueError("Staged execution_id does not match manifest execution_id")
    expected_uri = f"{bundle_uri.rstrip('/')}/{staged_bundle.bundle_id}"
    if staged_bundle.manifest.canonical_uri != expected_uri:
        raise ValueError("Manifest canonical_uri does not match the commit target")
    canonical_bytes = staged_bundle.manifest.canonical_json()
    if staged_bundle.manifest_bytes != canonical_bytes:
        raise ValueError("Staged manifest bytes are not canonical")
    actual_manifest_sha256 = hashlib.sha256(staged_bundle.manifest_bytes).hexdigest()
    if staged_bundle.manifest_sha256 != actual_manifest_sha256:
        raise ValueError("Staged manifest digest does not match manifest bytes")
    nodes_root = (staged_bundle.staging_root / "nodes").resolve()
    for artifact in staged_bundle.manifest.artifacts:
        if (
            not artifact.name
            or artifact.name in {".", ".."}
            or "/" in artifact.name
            or "\\" in artifact.name
        ):
            raise ValueError(f"Unsafe artifact name: {artifact.name!r}")
        artifact_dir = (nodes_root / artifact.name / "artifact").resolve()
        if not artifact_dir.is_relative_to(nodes_root):
            raise ValueError(f"Artifact staging path escapes root: {artifact.name!r}")
        verify_materialized_artifact(artifact, artifact_dir)


def _validate_materialization_request(
    manifest: ExportManifest,
    artifact: LogicalArtifact,
    limits: ReaderResourceLimits,
) -> None:
    if artifact not in manifest.artifacts:
        raise ValueError("Artifact is not declared by the committed manifest")
    enforce_artifact_limits(artifact, limits)


def _strip_file_scheme(uri: str) -> str:
    return uri[len("file://") :] if uri.startswith("file://") else uri


def _resolve_local_manifest_path(path: Path) -> Path:
    if path.is_file():
        return path
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        return manifest_path
    raise FileNotFoundError(f"Manifest not found: {manifest_path}")


def _resolve_s3_manifest_key(uri: str) -> tuple[str, str]:
    bucket, key = parse_s3_url(uri)
    if key.endswith("/"):
        key = f"{key}manifest.json"
    elif not key.endswith("manifest.json"):
        key = f"{key}/manifest.json"
    return bucket, key


def _parse_manifest(
    data: bytes, manifest_registry: ManifestSchemaRegistry
) -> ExportManifest:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Manifest is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Manifest must contain a JSON object")
    schema_version = raw.get("schema_version", 1)
    return manifest_registry.read(schema_version, raw, data)


def _alias_document(
    *, bundle_ref: BundleRef, manifest_uri: str, created_at: str
) -> dict[str, str]:
    return {
        "manifest_uri": manifest_uri,
        "manifest_sha256": bundle_ref.manifest_sha256,
        "bundle_id": bundle_ref.bundle_id,
        "created_at": created_at,
    }


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Serialize access to a local resource across threads and processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        _lock_file_descriptor(descriptor)
        locked = True
        yield
    finally:
        try:
            if locked:
                _unlock_file_descriptor(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def _local_alias_lock(alias_dir: Path) -> Iterator[None]:
    """Serialize local alias CAS across threads and processes."""
    with _exclusive_file_lock(alias_dir / ".tributo-alias.lock"):
        yield


def _lock_file_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        from importlib import import_module

        msvcrt = import_module("msvcrt")

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_file_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        from importlib import import_module

        msvcrt = import_module("msvcrt")

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _update_local_alias(
    *,
    alias_path: Path,
    alias_config: AliasConfig,
    bundle_ref: BundleRef,
    manifest_uri: str,
    created_at: str,
) -> AliasUpdate:
    failure_code: str | None = None
    should_write = True
    if alias_path.exists():
        existing_bytes = _read_local_bytes_bounded(
            alias_path,
            _CONTROL_DOCUMENT_MAX_BYTES,
            "Alias",
        )
        existing = _decode_json_object(existing_bytes, f"Alias {alias_path}")
        if alias_config.policy == "compare_and_swap":
            if alias_config.expected_manifest_sha256 is None:
                failure_code = "ALIAS_EXISTS"
            elif (
                existing.get("manifest_sha256") != alias_config.expected_manifest_sha256
            ):
                failure_code = "CAS_MISMATCH"
        else:
            existing_order = (
                existing.get("created_at", ""),
                existing.get("bundle_id", ""),
            )
            candidate_order = (created_at, bundle_ref.bundle_id)
            if existing_order >= candidate_order:
                should_write = False
    elif (
        alias_config.policy == "compare_and_swap"
        and alias_config.expected_manifest_sha256
    ):
        failure_code = "ALIAS_NOT_FOUND"

    if failure_code is not None:
        return _failed_alias_update(str(alias_path), failure_code)
    if not should_write:
        return AliasUpdate(alias_uri=str(alias_path), status="unchanged")

    alias_data = _alias_document(
        bundle_ref=bundle_ref,
        manifest_uri=manifest_uri,
        created_at=created_at,
    )
    temporary_path = alias_path.with_name(
        f".{alias_path.name}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        encoded = json.dumps(alias_data, indent=2).encode("utf-8")
        with temporary_path.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, alias_path)
        _fsync_dir(alias_path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return AliasUpdate(alias_uri=str(alias_path), status="updated")


def _bundle_ref_from_alias(data: Any, alias_uri: str) -> BundleRef:
    if not isinstance(data, dict):
        raise ValueError(f"Alias {alias_uri} must contain a JSON object")
    try:
        manifest_uri = str(data["manifest_uri"])
        bundle_id = str(data["bundle_id"])
        manifest_sha256 = str(data["manifest_sha256"])
    except KeyError as exc:
        raise ValueError(f"Alias {alias_uri} is missing {exc.args[0]!r}") from exc
    if manifest_uri.startswith("s3://"):
        canonical_uri = manifest_uri.removesuffix("/manifest.json")
    elif manifest_uri.startswith("file://"):
        canonical_uri = manifest_uri.removesuffix("/manifest.json")
    else:
        canonical_uri = str(Path(_strip_file_scheme(manifest_uri)).parent)
    return BundleRef(
        canonical_uri=canonical_uri,
        bundle_id=bundle_id,
        manifest_sha256=manifest_sha256,
    )


def _failed_alias_update(alias_uri: str, code: str) -> AliasUpdate:
    return AliasUpdate(
        alias_uri=alias_uri,
        status="failed",
        failure=FailureInfo(
            code=code,
            category="publish",
            message=f"Alias update failed: {code}",
        ),
    )
