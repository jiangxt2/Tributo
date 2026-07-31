"""S3-backed BundleRepository — atomic commit to S3 with manifest-last protocol.

Implements the ``BundleRepository`` protocol using boto3/S3.
Per-file uploads with retry, idempotency via content-addressing,
and CAS alias updates via ETag matching.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from tributo._common.storage import get_boto3_client, parse_s3_url
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exporting.models import FailureInfo
from tributo.exporting.repository import (
    AliasUpdateResult,
    BundleRef,
    CommitResult,
    UncommittedBundle,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────────

_LEASE_TTL_SECONDS = 300
_ALIAS_MAX_RETRIES = 3


@PublicAPI(stability="beta")
class S3BundleRepositoryConfig(BaseModel):
    """Configuration for ``S3BundleRepository``."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = None
    region: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    lease_ttl_seconds: int = _LEASE_TTL_SECONDS
    alias_max_retries: int = _ALIAS_MAX_RETRIES


@PublicAPI(stability="beta")
class S3BundleRepository:
    """Commit bundles to S3 with manifest-last atomicity.

    Implements ``BundleRepository`` using:
    - Per-file uploads with ``If-None-Match`` for idempotency.
    - Lease-based concurrency control across writers.
    - Manifest-last protocol (artifacts first, manifest final).
    - ETag-based CAS alias updates.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        resolver: StorageProfileResolver | None = None,
        config: S3BundleRepositoryConfig | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._resolver = resolver or StorageProfileResolver()
        self._config = config or S3BundleRepositoryConfig()

        self._client = get_boto3_client(
            endpoint=self._config.endpoint,
            access_key_id=self._config.access_key_id,
            secret_access_key=self._config.secret_access_key,
            region=self._config.region,
        )

    # ── BundleRepository.commit ───────────────────────────────────────────────

    def commit(self, bundle: UncommittedBundle) -> CommitResult:
        """Atomically persist *bundle* to S3."""
        bundle_id = bundle.bundle_id
        bundle_prefix = f"{self._prefix}{bundle_id}/"
        owner = f"{uuid.uuid4().hex[:8]}"
        leases_prefix = f"{self._prefix}.leases/"

        # Phase 1: Acquire lease.
        lease_key, _ = _s3_lease_acquire(
            self._client, self._bucket, leases_prefix, bundle_id, owner,
            self._config.lease_ttl_seconds,
        )
        logger.info("Acquired S3 publish lease s3://%s/%s", self._bucket, lease_key)

        try:
            # Phase 2: Upload artifact files.
            for artifact in bundle.artifacts:
                artifact_src = bundle.staging_root / "nodes" / artifact.name / "artifact"
                for af in artifact.files:
                    src_file = artifact_src / af.relative_path
                    key = f"{bundle_prefix}artifacts/{artifact.name}/{af.relative_path}"
                    _s3_upload_file(
                        self._client, self._bucket, key,
                        src_file.read_bytes(), af.sha256,
                    )

            # Phase 3: Write manifest-last with If-None-Match.
            manifest_key = f"{bundle_prefix}manifest.json"
            existing = _s3_head(self._client, self._bucket, manifest_key)
            if existing is not None:
                existing_meta = existing.get("metadata", {})
                existing_sha = existing_meta.get("tributo-sha256", "")
                if existing_sha == bundle.manifest_sha256:
                    return CommitResult(
                        canonical_uri=f"s3://{self._bucket}/{bundle_prefix}",
                        manifest_uri=f"s3://{self._bucket}/{manifest_key}",
                        manifest_sha256=bundle.manifest_sha256,
                        commit_status="idempotent",
                    )
                raise RuntimeError(
                    f"Manifest s3://{self._bucket}/{manifest_key} exists but differs"
                )

            _s3_put_bytes(
                self._client, self._bucket, manifest_key,
                bundle.manifest_bytes,
                if_none_match=True,
                content_type="application/json",
                metadata={"tributo-sha256": bundle.manifest_sha256},
            )

            return CommitResult(
                canonical_uri=f"s3://{self._bucket}/{bundle_prefix}",
                manifest_uri=f"s3://{self._bucket}/{manifest_key}",
                manifest_sha256=bundle.manifest_sha256,
                commit_status="committed",
            )

        finally:
            _s3_delete(self._client, self._bucket, lease_key)

    # ── BundleRepository.get ──────────────────────────────────────────────────

    def get(self, ref: BundleRef) -> dict[str, Any]:
        """Read the manifest for *ref* from S3."""
        manifest_key = f"{self._prefix}{ref.bundle_id}/manifest.json"
        raw = _s3_get_json(self._client, self._bucket, manifest_key)
        if raw is None:
            raise FileNotFoundError(
                f"Manifest not found: s3://{self._bucket}/{manifest_key}"
            )
        return raw  # type: ignore[no-any-return]

    # ── BundleRepository.update_alias ─────────────────────────────────────────

    def update_alias(
        self,
        alias: str,
        new_ref: BundleRef,
        expected_revision: str | None = None,
    ) -> AliasUpdateResult:
        """Create or update alias with CAS via ETag."""
        alias_key = f"{self._prefix}aliases/{alias}.json"
        alias_data = {
            "manifest_sha256": new_ref.manifest_sha256,
            "canonical_uri": new_ref.canonical_uri,
            "bundle_id": new_ref.bundle_id,
        }

        for _ in range(self._config.alias_max_retries):
            existing = _s3_get_json(self._client, self._bucket, alias_key)
            head = _s3_head(self._client, self._bucket, alias_key)

            if existing is None:
                try:
                    _s3_put_json(
                        self._client, self._bucket, alias_key,
                        alias_data, if_none_match=True,
                    )
                    return AliasUpdateResult(alias=alias, status="updated")
                except self._client.exceptions.ClientError as exc:
                    if exc.response["Error"]["Code"] == "PreconditionFailed":
                        continue
                    return AliasUpdateResult(
                        alias=alias, status="failed",
                        failure=FailureInfo(
                            code=exc.response["Error"]["Code"],
                            category="publish",
                            message="Alias create failed",
                        ),
                    )

            # CAS check.
            if expected_revision is not None:
                current_sha = existing.get("manifest_sha256", "")
                if current_sha != expected_revision:
                    return AliasUpdateResult(
                        alias=alias, status="failed",
                        failure=FailureInfo(
                            code="CAS_MISMATCH",
                            category="publish",
                            message="Expected manifest_sha256 does not match",
                        ),
                    )

            # Update with If-Match.
            if head is None:
                continue
            try:
                _s3_put_json(
                    self._client, self._bucket, alias_key,
                    alias_data, if_match=head["etag"],
                )
                return AliasUpdateResult(alias=alias, status="updated")
            except self._client.exceptions.ClientError as exc:
                if exc.response["Error"]["Code"] == "PreconditionFailed":
                    continue
                return AliasUpdateResult(
                    alias=alias, status="failed",
                    failure=FailureInfo(
                        code=exc.response["Error"]["Code"],
                        category="publish",
                        message="Alias update failed",
                    ),
                )

        return AliasUpdateResult(
            alias=alias, status="failed",
            failure=FailureInfo(
                code="RETRY_EXHAUSTED",
                category="publish",
                message="Alias update exhausted retries",
            ),
        )


# ── S3 helpers ─────────────────────────────────────────────────────────────────────


def _s3_head(client: Any, bucket: str, key: str) -> Any:
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        return {
            "content_length": resp.get("ContentLength", 0),
            "etag": resp.get("ETag", "").strip('"'),
            "metadata": resp.get("Metadata", {}),
        }
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return None
        raise


def _s3_put_json(
    client: Any, bucket: str, key: str, data: dict[str, Any],
    *, if_none_match: bool = False, if_match: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Any:
    extra: dict[str, Any] = {}
    if if_none_match:
        extra["IfNoneMatch"] = "*"
    if if_match is not None:
        extra["IfMatch"] = f'"{if_match}"'
    if metadata:
        extra["Metadata"] = metadata
    body = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return client.put_object(
        Bucket=bucket, Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
        **extra,
    )


def _s3_put_bytes(
    client: Any, bucket: str, key: str, data: bytes,
    *, if_none_match: bool = False, content_type: str = "application/octet-stream",
    metadata: dict[str, str] | None = None,
) -> Any:
    extra: dict[str, Any] = {}
    if if_none_match:
        extra["IfNoneMatch"] = "*"
    if metadata:
        extra["Metadata"] = metadata
    return client.put_object(
        Bucket=bucket, Key=key,
        Body=data, ContentType=content_type, **extra,
    )


def _s3_get_json(client: Any, bucket: str, key: str) -> Any:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        raw: bytes = resp["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def _s3_delete(client: Any, bucket: str, key: str) -> None:
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        logger.debug("Failed to delete s3://%s/%s", bucket, key, exc_info=True)


def _s3_upload_file(
    client: Any, bucket: str, key: str, data: bytes, sha256: str,
) -> None:
    """Upload a file to S3, idempotent via sha256 metadata."""
    existing = _s3_head(client, bucket, key)
    if existing is not None:
        existing_len = existing["content_length"]
        existing_meta = existing.get("metadata", {})
        existing_sha = existing_meta.get("tributo-sha256", "")
        if existing_len == len(data) and existing_sha == sha256:
            logger.debug("s3://%s/%s already exists — idempotent", bucket, key)
            return
        raise RuntimeError(f"Object s3://{bucket}/{key} exists but differs")

    try:
        _s3_put_bytes(
            client, bucket, key, data,
            if_none_match=True,
            metadata={"tributo-sha256": sha256},
        )
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "PreconditionFailed":
            existing = _s3_head(client, bucket, key)
            if existing:
                existing_sha = existing.get("metadata", {}).get("tributo-sha256", "")
                if existing["content_length"] == len(data) and existing_sha == sha256:
                    return
        raise


def _s3_lease_acquire(
    client: Any, bucket: str, leases_prefix: str,
    bundle_id: str, owner: str, ttl: int,
) -> tuple[str, bool]:
    lease_key = f"{leases_prefix}{bundle_id}.json"
    now = time.time()
    lease_data = {
        "owner": owner, "created_at": now,
        "expires_at": now + ttl, "bundle_id": bundle_id,
    }
    existing = _s3_get_json(client, bucket, lease_key)
    if existing is None:
        try:
            _s3_put_json(client, bucket, lease_key, lease_data, if_none_match=True)
            return lease_key, True
        except client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "PreconditionFailed":
                existing = _s3_get_json(client, bucket, lease_key)
            else:
                raise
    if existing is not None:
        existing_owner = existing.get("owner")
        expires_at = existing.get("expires_at", 0)
        if existing_owner == owner or now > expires_at:
            head = _s3_head(client, bucket, lease_key)
            if head:
                _s3_put_json(client, bucket, lease_key, lease_data, if_match=head["etag"])
                return lease_key, False
    raise RuntimeError(
        f"Lease {lease_key} held by {existing.get('owner', 'unknown')}"
        f" until {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(existing.get('expires_at', 0)))}"
        if existing else f"Lease {lease_key} unavailable"
    )
