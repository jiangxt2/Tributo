"""Bundle publisher — atomic local commit, S3 manifest-last, alias CAS.

Responsibilities (and only these):
- Assemble the canonical bundle directory from staging artifacts.
- Write manifest and metadata.
- Atomic publish to local filesystem or S3.
- Update aliases (newer / compare_and_swap).
- Produce ``PublishedBundle`` (which carries ``BundleResult``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from tributo._common.storage import (
    get_boto3_client,
    parse_s3_url,
)
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exporting.manifest import (
    ExportManifest,
    ManifestExecution,
    ManifestExecutionNode,
    ManifestSchemaRegistry,
    ManifestSignature,
    ManifestSourceInfo,
)
from tributo.exporting.models import (
    AliasConfig,
    BundleResult,
    ExportExecutionResult,
    FailureInfo,
    LogicalArtifact,
    PublishedBundle,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_LEASE_TTL_SECONDS = 300  # 5 minutes
_ALIAS_MAX_RETRIES = 3
# Single-PutObject hard limit — larger objects must use multipart upload.
_S3_SINGLE_PUT_LIMIT = 5 * 1024 * 1024 * 1024  # 5 GiB


# ── S3 helpers ─────────────────────────────────────────────────────────────────


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
            "metadata": {k.lower(): v for k, v in resp.get("Metadata", {}).items()},
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


def _s3_get_json(client: Any, bucket: str, key: str) -> Any:
    """Get and parse a JSON object from S3, or None if not found."""
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        raw: bytes = resp["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def _s3_get_bytes(client: Any, bucket: str, key: str) -> bytes | None:
    """Read an object's body as bytes, or None if not found."""
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body: bytes = resp["Body"].read()
        return body
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


def _copy_tree_fsync(src: Path, dst: Path) -> None:
    """Copy directory tree with per-file fsync."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            # fsync the file
            with open(target, "rb") as f:
                os.fsync(f.fileno())
        # fsync the parent directory after each file
        _fsync_dir(target.parent)


def _local_identical_manifest(final_dir: Path, manifest_bytes: bytes) -> bytes | None:
    """Return the existing manifest bytes if *final_dir* holds the same bundle.

    Compares bundle_id + artifact tree digests semantically — a subset of
    the S3 logical comparison (roles/source_info changes are not detected
    locally).  Returns None when the directory is missing, has no manifest,
    or holds different content.
    """
    existing_manifest_path = final_dir / "manifest.json"
    if not existing_manifest_path.is_file():
        return None
    try:
        existing_bytes = existing_manifest_path.read_bytes()
        existing_raw = json.loads(existing_bytes)
        candidate_raw = json.loads(manifest_bytes)
    except (json.JSONDecodeError, OSError):
        return None
    if existing_raw.get("bundle_id") != candidate_raw.get("bundle_id"):
        return None
    existing_digests = sorted(
        a.get("tree_digest", "") for a in existing_raw.get("artifacts", [])
    )
    candidate_digests = sorted(
        a.get("tree_digest", "") for a in candidate_raw.get("artifacts", [])
    )
    if existing_digests != candidate_digests:
        return None
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
) -> tuple[Path, str]:
    """Publish bundle to a local directory using atomic rename.

    Returns ``(final_dir, sha256)`` — the sha256 is from the existing manifest
    when idempotent, or from the newly written one.
    """
    final_dir = bundle_dir.resolve()
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Check for existing identical bundle (idempotency).
    existing_bytes = _local_identical_manifest(final_dir, manifest_bytes)
    if existing_bytes is not None:
        logger.info("Bundle %s already exists — idempotent", final_dir)
        return final_dir, hashlib.sha256(existing_bytes).hexdigest()
    if final_dir.exists():
        raise RuntimeError(
            f"Bundle directory {final_dir} exists but has no manifest or "
            "different content — collision"
        )

    # Create temp directory in the same filesystem as final.
    temp_dir = parent / f".tmp-{bundle_id}-{execution_id}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        temp_dir.mkdir(parents=True)

        # Copy artifacts from staging.
        artifacts_dir = temp_dir / "artifacts"
        for artifact in manifest.artifacts:
            artifact_src = staging_root / "nodes" / artifact.name / "artifact"
            artifact_dst = artifacts_dir / artifact.name
            _copy_tree_fsync(artifact_src, artifact_dst)

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
            existing_bytes = _local_identical_manifest(final_dir, manifest_bytes)
            if existing_bytes is not None:
                logger.info("Bundle %s appeared during rename — idempotent", final_dir)
                return final_dir, hashlib.sha256(existing_bytes).hexdigest()
            raise

        _fsync_dir(parent)
        return final_dir, manifest_sha256

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
    *,
    execution_id: str | None = None,
) -> tuple[str, bool]:
    """Acquire or renew a publish lease.

    Returns ``(lease_key, is_new)``.
    """
    lease_key = f"{leases_prefix}{bundle_id}.json"
    now = time.time()
    lease_data = {
        "owner": owner,
        "created_at": now,
        "expires_at": now + ttl,
        "bundle_id": bundle_id,
    }

    existing = _s3_get_json(client, bucket, lease_key)
    if existing is None:
        # First acquisition — create with If-None-Match.
        try:
            _s3_put_json(client, bucket, lease_key, lease_data, if_none_match=True)
            return lease_key, True
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "PreconditionFailed":
                # Race — another process got there first.  Re-read.
                existing = _s3_get_json(client, bucket, lease_key)
            else:
                raise

    if existing is not None:
        # Check if lease is ours or expired.  An idempotent retry of the
        # same request reuses the same execution_id but carries a fresh
        # random owner suffix — treat it as the same publisher so the
        # retry can renew the lease instead of being rejected.
        existing_owner = existing.get("owner")
        expires_at = existing.get("expires_at", 0)
        same_execution = (
            execution_id is not None
            and isinstance(existing_owner, str)
            and existing_owner.startswith(f"{execution_id}-")
        )
        if existing_owner == owner or same_execution:
            # Renew our lease.
            head = _s3_head(client, bucket, lease_key)
            if head:
                try:
                    _s3_put_json(
                        client,
                        bucket,
                        lease_key,
                        lease_data,
                        if_match=head["etag"],
                    )
                    return lease_key, False
                except client.exceptions.ClientError as exc:
                    if exc.response["Error"]["Code"] == "PreconditionFailed":
                        logger.warning("Lease %s changed under us", lease_key)
                    raise
        elif now > expires_at:
            # Expired — take over.
            head = _s3_head(client, bucket, lease_key)
            if head:
                try:
                    _s3_put_json(
                        client,
                        bucket,
                        lease_key,
                        lease_data,
                        if_match=head["etag"],
                    )
                    logger.info(
                        "Took over expired lease %s (previous owner %s)",
                        lease_key,
                        existing_owner,
                    )
                    return lease_key, False
                except client.exceptions.ClientError as exc:
                    if exc.response["Error"]["Code"] == "PreconditionFailed":
                        logger.warning(
                            "Lease %s changed during takeover attempt", lease_key
                        )
                    raise
        else:
            raise RuntimeError(
                f"Lease {lease_key} held by {existing_owner} until "
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(expires_at))}"
            )

    return lease_key, False


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

        # Upload with If-None-Match, sha256 metadata, and a server-side
        # checksum.  Objects above the single-PutObject limit (5 GiB) go
        # through boto3's multipart path.
        data = src_file.read_bytes()
        metadata = {"tributo-sha256": af.sha256}
        if af.size_bytes > _S3_SINGLE_PUT_LIMIT:
            with open(src_file, "rb") as fh:
                client.upload_fileobj(
                    fh,
                    bucket,
                    key,
                    ExtraArgs={
                        "Metadata": metadata,
                        "ChecksumAlgorithm": "SHA256",
                    },
                )
            keys.append(key)
            if created_keys is not None:
                created_keys.append(key)
            continue
        try:
            _s3_put_bytes(
                client,
                bucket,
                key,
                data,
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

    # Phase 1: Acquire lease.
    lease_key, _ = _s3_lease_acquire(
        client,
        bucket,
        leases_prefix,
        bundle_id,
        owner,
        execution_id=execution_id,
    )
    logger.info("Acquired publish lease %s", lease_key)

    try:
        # Phase 2: Upload all artifact files.  The lease is renewed at
        # half-TTL intervals so a long upload cannot be taken over by
        # another publisher or the orphan GC mid-flight.
        uploaded_keys: list[str] = []
        created_keys: list[str] = []
        manifest_committed = False
        renew_at = time.time() + _LEASE_TTL_SECONDS / 2
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
            if time.time() >= renew_at:
                _s3_lease_acquire(
                    client,
                    bucket,
                    leases_prefix,
                    bundle_id,
                    owner,
                    execution_id=execution_id,
                )
                renew_at = time.time() + _LEASE_TTL_SECONDS / 2

        # Phase 3: Write manifest-last with If-None-Match.
        manifest_key = f"{bundle_prefix}manifest.json"
        existing = _s3_head(client, bucket, manifest_key)
        if existing is not None:
            existing_meta = existing.get("metadata", {})
            existing_sha = existing_meta.get("tributo-sha256", "")
            if existing_sha == manifest_sha256:
                logger.info(
                    "Manifest s3://%s/%s already exists — idempotent",
                    bucket,
                    manifest_key,
                )
                manifest_committed = True
            else:
                # A retry of the same request re-runs the DAG and rebuilds
                # the manifest with fresh advisory fields (created_at, node
                # duration_ms, validation metrics), so byte-level shas may
                # differ even for the same logical publish.  Fall back to a
                # logical comparison (mirrors the local publish's semantic
                # check on bundle_id + artifact tree digests).
                existing_body = _s3_get_bytes(client, bucket, manifest_key)
                if existing_body is not None and _manifest_logically_equal(
                    existing_body, manifest_bytes
                ):
                    logger.info(
                        "Manifest s3://%s/%s logically identical — idempotent",
                        bucket,
                        manifest_key,
                    )
                    # The manifest already on disk is the source of truth —
                    # surface its actual sha so a retried result matches what
                    # consumers will verify against.
                    manifest_sha256 = existing_sha
                    manifest_committed = True
                else:
                    raise RuntimeError(
                        f"Manifest s3://{bucket}/{manifest_key} exists but "
                        "differs — collision"
                    )
        else:
            metadata = {"tributo-sha256": manifest_sha256}
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

        return {
            "manifest_key": manifest_key,
            "uploaded_count": len(uploaded_keys),
            # On idempotent retries this is the sha of the manifest already
            # on disk, not the freshly computed one.
            "manifest_sha256": manifest_sha256,
        }

    except Exception:
        # A failed pre-manifest publish must not leave orphaned objects.  Only
        # delete keys uploaded by this invocation; idempotently reused or
        # concurrently-created objects are never part of created_keys.
        if not manifest_committed:
            for key in created_keys:
                _s3_delete(client, bucket, key)
        raise

    finally:
        # Best-effort delete lease.
        _s3_delete(client, bucket, lease_key)


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
        existing = _s3_get_json(client, bucket, alias_key_path)
        head_info = _s3_head(client, bucket, alias_key_path)

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
            if head_info is None:
                continue  # Deleted under us — retry.
            try:
                _s3_put_json(
                    client,
                    bucket,
                    alias_key_path,
                    alias_data,
                    if_match=head_info["etag"],
                )
                return "updated", None
            except client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "PreconditionFailed":
                    continue  # Race — retry.
                return "failed", exc.response["Error"]["Code"]

    return "failed", "RETRY_EXHAUSTED"


# ── Top-level publisher ────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class Publisher:
    """Assembles and publishes a bundle from an execution result.

    The publisher is the only component that knows the final URI, manifest
    URI, and alias write outcome.  It produces the immutable ``BundleResult``.
    """

    def __init__(
        self,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
    ) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()
        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        # Register built-in v1 reader.
        from tributo.exporting.manifest import _read_manifest_v1

        try:
            self._manifest_registry.register(1, _read_manifest_v1)
        except ValueError:
            pass  # Already registered.

    def publish(
        self,
        execution: ExportExecutionResult,
        staging_root: Path,
        bundle_uri: str,
        bundle_id: str,
        execution_id: str,
        tributo_version: str,
        source_info: ManifestSourceInfo,
        input_signature: ManifestSignature | None = None,
        output_signature: ManifestSignature | None = None,
        storage_profile: str | None = None,
        alias_config: AliasConfig | None = None,
        roles: dict[str, str] | None = None,
    ) -> PublishedBundle:
        """Publish a completed execution as a bundle.

        Only ``succeeded`` or ``partial`` executions are accepted.
        ``failed`` executions raise ``ValueError``.
        """
        if execution.status == "failed":
            raise ValueError("Cannot publish a failed execution")

        effective_roles = roles if roles is not None else execution.roles

        # Roles must reference publishable explicit artifacts that succeeded
        # (plan: "Publisher 校验 role 必须引用 publish=True && status=succeeded
        # 的显式 artifact；否则整个发布失败").
        node_by_name = {nr.target_name: nr for nr in execution.node_results}
        for role_name, target_name in effective_roles.items():
            nr = node_by_name.get(target_name)
            if nr is None:
                raise ValueError(
                    f"Role {role_name!r} references unknown target {target_name!r}"
                )
            if not nr.publish:
                raise ValueError(
                    f"Role {role_name!r} references non-publishable target "
                    f"{target_name!r} (implicit nodes cannot be roles)"
                )
            if nr.status != "succeeded":
                raise ValueError(
                    f"Role {role_name!r} references target {target_name!r} "
                    f"with status {nr.status!r}; roles must reference "
                    "succeeded artifacts"
                )

        # Build manifest.
        published_artifacts: list[LogicalArtifact] = []
        has_failed_optional = False

        for nr in execution.node_results:
            if nr.status in ("failed", "blocked") and not nr.required:
                has_failed_optional = True

        manifest_nodes: list[ManifestExecutionNode] = []
        for nr in execution.node_results:
            manifest_nodes.append(
                ManifestExecutionNode(
                    node_id=nr.node_id,
                    target_name=nr.target_name,
                    exporter_id=nr.exporter_id,
                    status=nr.status,
                    required=nr.required,
                    implicit=(nr.node_id.startswith("_implicit__")),
                    artifact_ref=nr.artifact_ref,
                    failure=nr.failure,
                    duration_ms=nr.duration_ms,
                )
            )

        # Collect published artifacts (succeeded + publish=True, non-implicit).
        for nr in execution.node_results:
            if nr.status == "succeeded" and nr.publish and nr.artifact_ref is not None:
                descriptor = execution.staged_artifacts.get(nr.node_id)
                if descriptor is not None:
                    published_artifacts.append(descriptor)

        manifest = ExportManifest(
            schema_version=1,
            bundle_id=bundle_id,
            status="partial" if has_failed_optional else "succeeded",
            canonical_uri=f"{bundle_uri.rstrip('/')}/{bundle_id}",
            tributo_version=tributo_version,
            source_info=source_info,
            input_signature=input_signature or ManifestSignature(),
            output_signature=output_signature or ManifestSignature(),
            artifacts=tuple(published_artifacts),
            roles=effective_roles,
            execution=ManifestExecution(
                execution_id=execution_id,
                nodes=tuple(manifest_nodes),
            ),
        )

        manifest_bytes = manifest.canonical_json()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

        # Publish to destination.
        if bundle_uri.startswith("s3://"):
            bucket, bundle_prefix_raw = parse_s3_url(bundle_uri)
            store_prefix = (
                bundle_prefix_raw.rstrip("/") + "/" if bundle_prefix_raw else ""
            )
            bundle_prefix = f"{store_prefix}{bundle_id}/"
            client = _s3_client_from_profile(self._storage_resolver, storage_profile)
            publish_meta = _s3_publish(
                client=client,
                bucket=bucket,
                bundle_prefix=bundle_prefix,
                staging_root=staging_root,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                bundle_id=bundle_id,
                execution_id=execution_id,
            )
            # Idempotent retries report the sha of the manifest already on
            # disk — consumers verify against the object, not the draft.
            manifest_sha256 = publish_meta["manifest_sha256"]
            canonical_uri = f"s3://{bucket}/{bundle_prefix}"
            manifest_uri = f"{canonical_uri}manifest.json"
            local_bundle_dir = staging_root  # S3: no persistent local dir
            local_dir_ephemeral = True

            # Alias — written at store level (not per-bundle).
            # E.g. s3://bucket/models/aliases/latest.json
            alias_uri: str | None = None
            alias_status = "not_requested"
            alias_failure: FailureInfo | None = None

            if alias_config is not None:
                ak = f"{store_prefix}aliases/{alias_config.name}.json"
                alias_uri = f"s3://{bucket}/{ak}"
                created_at_str = manifest.created_at.isoformat()
                status, fail_code = _update_alias_s3(
                    client=client,
                    bucket=bucket,
                    alias_key_path=ak,
                    alias_config=alias_config,
                    manifest_uri=manifest_uri,
                    manifest_sha256=manifest_sha256,
                    bundle_id=bundle_id,
                    created_at=created_at_str,
                )
                alias_status = status
                if status == "failed":
                    alias_failure = FailureInfo(
                        code=fail_code or "ALIAS_FAILED",
                        category="publish",
                        message=f"Alias update failed: {fail_code}",
                    )

        else:
            # Local publish — strip file:// prefix if present.
            local_uri = (
                bundle_uri[7:] if bundle_uri.startswith("file://") else bundle_uri
            )
            bundle_dir = Path(local_uri) / bundle_id
            final_dir, actual_manifest_sha256 = _local_publish(
                bundle_dir=bundle_dir,
                staging_root=staging_root,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                manifest_sha256=manifest_sha256,
                bundle_id=bundle_id,
                execution_id=execution_id,
            )
            canonical_uri = str(final_dir)
            manifest_uri = str(final_dir / "manifest.json")
            local_bundle_dir = final_dir
            local_dir_ephemeral = False
            # Use the sha256 from the existing (idempotent) or newly written manifest.
            manifest_sha256 = actual_manifest_sha256

            # Local alias (atomic write).
            alias_uri = None
            alias_status = "not_requested"
            alias_failure = None

            if alias_config is not None:
                alias_dir = Path(local_uri) / "aliases"
                alias_dir.mkdir(parents=True, exist_ok=True)
                alias_path = alias_dir / f"{alias_config.name}.json"
                alias_uri = str(alias_path)

                created_at_str = manifest.created_at.isoformat()
                should_write = True

                # Policy checks (mirror S3 _update_alias_s3).
                if not alias_path.exists():
                    # CAS with expected digest requires the alias to already
                    # exist — nothing to compare against, so this is an error
                    # (same ALIAS_NOT_FOUND semantics as the S3 path).
                    if (
                        alias_config.policy == "compare_and_swap"
                        and alias_config.expected_manifest_sha256
                    ):
                        alias_status = "failed"
                        alias_failure = FailureInfo(
                            code="ALIAS_NOT_FOUND",
                            category="publish",
                            message=(
                                "CAS with expected digest but alias does not exist"
                            ),
                        )
                        should_write = False
                else:
                    existing = json.loads(alias_path.read_bytes())
                    if alias_config.policy == "compare_and_swap":
                        if alias_config.expected_manifest_sha256 is None:
                            alias_status = "failed"
                            alias_failure = FailureInfo(
                                code="ALIAS_EXISTS",
                                category="publish",
                                message="CAS create-only but alias already exists",
                            )
                            should_write = False
                        else:
                            current = existing.get("manifest_sha256", "")
                            if current != alias_config.expected_manifest_sha256:
                                alias_status = "failed"
                                alias_failure = FailureInfo(
                                    code="CAS_MISMATCH",
                                    category="publish",
                                    message="Expected manifest digest does not match",
                                )
                                should_write = False
                    elif alias_config.policy == "newer":
                        existing_ts = existing.get("created_at", "")
                        if existing_ts > created_at_str:
                            alias_status = "unchanged"
                            should_write = False

                if should_write:
                    alias_data = {
                        "manifest_uri": manifest_uri,
                        "manifest_sha256": manifest_sha256,
                        "bundle_id": bundle_id,
                        "created_at": created_at_str,
                    }
                    alias_bytes = json.dumps(alias_data, indent=2).encode("utf-8")
                    tmp_path = alias_path.with_suffix(".tmp")
                    tmp_path.write_bytes(alias_bytes)
                    os.replace(str(tmp_path), str(alias_path))
                    alias_status = "updated"

        bundle_result = BundleResult(
            bundle_id=bundle_id,
            canonical_uri=canonical_uri,
            manifest_uri=manifest_uri,
            manifest_sha256=manifest_sha256,
            status=manifest.status,
            artifacts=manifest.artifacts,
            node_results=execution.node_results,
            roles=effective_roles,
            alias_uri=alias_uri,
            alias_status=alias_status,
            alias_failure=alias_failure,
        )

        return PublishedBundle(
            result=bundle_result,
            local_bundle_dir=local_bundle_dir,
            local_dir_ephemeral=local_dir_ephemeral,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────
