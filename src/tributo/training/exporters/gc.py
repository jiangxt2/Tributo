"""Safe orphan bundle garbage collection for S3.

Only deletes prefixes that:
- Have no manifest object (orphan).
- Are older than the configured TTL.
- Have an expired or absent lease that can be safely acquired.

GC never touches ``.leases/``, ``aliases/``, or any reserved prefix.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from tributo._common.storage import get_boto3_client, parse_s3_url
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_GC_LEASE_TTL = 600  # 10 minutes for GC lease
_DEFAULT_ORPHAN_TTL = 3600  # 1 hour before an orphan can be collected
_RESERVED_PREFIXES: frozenset[str] = frozenset({".leases", "aliases", "trials"})


# ── Collector ──────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleGarbageCollector:
    """Scans and safely removes orphaned bundle prefixes on S3.

    Safety properties:
    - Only deletes prefixes without a ``manifest.json``.
    - Only deletes orphans older than *orphan_ttl_seconds*.
    - Acquires a short GC lease before deleting to avoid racing with a
      concurrent publish.
    - Never touches reserved prefixes (``.leases/``, ``aliases/``, ``trials/``).
    - Dry-run by default; requires explicit ``dry_run=False`` to delete.
    """

    def __init__(
        self,
        storage_resolver: StorageProfileResolver | None = None,
    ) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()

    def collect(
        self,
        bundle_uri: str,
        *,
        storage_profile: str | None = None,
        orphan_ttl_seconds: int = _DEFAULT_ORPHAN_TTL,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Scan and optionally clean up orphan bundle prefixes.

        Args:
            bundle_uri: Base S3 URI for the bundle store (e.g. ``s3://bucket/models/``).
            storage_profile: Storage profile name for credentials.
            orphan_ttl_seconds: Minimum age in seconds before an orphan is collected.
            dry_run: If ``True`` (default), only report what would be deleted.

        Returns:
            Dict with ``scanned``, ``orphans_found``, ``deleted``, ``errors``.
        """
        if not bundle_uri.startswith("s3://"):
            raise ValueError("BundleGarbageCollector only supports S3 URIs")

        bucket, prefix_raw = parse_s3_url(bundle_uri)
        prefix = prefix_raw.rstrip("/") + "/" if prefix_raw else ""
        client = _make_client(self._storage_resolver, storage_profile)

        scanned = 0
        orphans: list[str] = []
        errors: list[str] = []

        # List direct child prefixes.
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                child_prefix = cp["Prefix"]
                scanned += 1

                # Skip reserved prefixes.
                child_name = child_prefix[len(prefix) :].rstrip("/")
                if child_name in _RESERVED_PREFIXES:
                    continue

                # Check for manifest.
                manifest_key = f"{child_prefix}manifest.json"
                try:
                    client.head_object(Bucket=bucket, Key=manifest_key)
                    continue  # has manifest → not an orphan.
                except client.exceptions.ClientError as exc:
                    if exc.response["Error"]["Code"] != "404":
                        errors.append(f"HEAD {manifest_key}: {exc}")
                        continue

                # Check age of the oldest object in the prefix.
                age_ok = _check_orphan_age(
                    client, bucket, child_prefix, orphan_ttl_seconds
                )
                if not age_ok:
                    continue

                orphans.append(child_prefix)

        gc_owner = f"gc-{uuid.uuid4().hex[:8]}"
        deleted = 0
        if not dry_run:
            for orphan_prefix in orphans:
                try:
                    # Acquire GC lease to protect against concurrent publish.
                    lease_key = _acquire_gc_lease(
                        client, bucket, orphan_prefix, gc_owner
                    )
                    if lease_key is None:
                        logger.info(
                            "GC: skipping %s — lease held by another process",
                            orphan_prefix,
                        )
                        continue
                    _delete_prefix_safely(client, bucket, orphan_prefix)
                    _release_gc_lease(client, bucket, lease_key)
                    deleted += 1
                    logger.info(
                        "GC: deleted orphan prefix s3://%s/%s", bucket, orphan_prefix
                    )
                except Exception as exc:
                    errors.append(f"DELETE {orphan_prefix}: {exc}")
                    logger.warning("GC: failed to delete %s: %s", orphan_prefix, exc)

        return {
            "scanned": scanned,
            "orphans_found": len(orphans),
            "deleted": deleted,
            "errors": errors,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_client(resolver: StorageProfileResolver, storage_profile: str | None) -> Any:
    profile = resolver.resolve(storage_profile)
    return get_boto3_client(
        endpoint=profile.endpoint,
        access_key_id=profile.access_key_id,
        secret_access_key=profile.secret_access_key,
        region=profile.region,
    )


def _acquire_gc_lease(client: Any, bucket: str, prefix: str, owner: str) -> str | None:
    """Try to acquire a short-lived GC lease for *prefix*.

    Returns the lease key if acquired, or ``None`` if a concurrent process
    holds the lease.
    """
    lease_key = f"{prefix}.leases/gc.json"
    now = time.time()
    lease_data = {"owner": owner, "created_at": now, "action": "gc"}

    try:
        client.put_object(
            Bucket=bucket,
            Key=lease_key,
            Body=json.dumps(lease_data).encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        return lease_key
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "PreconditionFailed":
            logger.debug("GC lease for %s already held", prefix)
            return None
        raise


def _release_gc_lease(client: Any, bucket: str, lease_key: str) -> None:
    """Best-effort release of a GC lease."""
    try:
        client.delete_object(Bucket=bucket, Key=lease_key)
    except Exception:
        logger.debug("Failed to release GC lease %s", lease_key, exc_info=True)


def _check_orphan_age(client: Any, bucket: str, prefix: str, ttl_seconds: int) -> bool:
    """Check if all objects under *prefix* are older than *ttl_seconds*."""
    now = time.time()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=10):
        for obj in page.get("Contents", []):
            last_modified = obj["LastModified"].timestamp()
            if now - last_modified < ttl_seconds:
                return False
    return True


def _delete_prefix_safely(client: Any, bucket: str, prefix: str) -> None:
    """Delete all objects under *prefix* (best-effort, paginated)."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if objects:
            client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
            )
