"""S3 orphan-bundle garbage collection adapter.

Only deletes prefixes that:
- Have no manifest object (orphan).
- Are older than the configured TTL.
- Have an expired or absent lease that can be safely acquired.

GC never touches ``.leases/``, ``aliases/``, or any reserved prefix.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from tributo._common.storage import get_boto3_client, parse_s3_url
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exceptions import BundleCommitBusyError
from tributo.integrations.storage.bundle_repository import (
    _s3_lease_acquire,
    _s3_lease_release,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_GC_LEASE_TTL = 600  # 10 minutes for GC lease
_DEFAULT_ORPHAN_TTL = 3600  # 1 hour before an orphan can be collected
_RESERVED_PREFIXES: frozenset[str] = frozenset({".leases", "aliases", "trials"})


# ── Collector ──────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class S3BundleGarbageCollector:
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
        if not prefix_raw:
            logger.warning(
                "GC is scanning the S3 bucket root; bundle_uri must be the exact "
                "store root used by Publisher or its lease namespace will differ"
            )
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

                # Skip prefixes that don't look like bundle IDs.
                if not _looks_like_bundle_id(child_name):
                    logger.debug("GC: skipping non-bundle prefix %r", child_name)
                    continue

                # Check for manifest.
                manifest_key = f"{child_prefix}manifest.json"
                lease_key: str | None = None
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
                lease_key = None
                try:
                    # Acquire GC lease (same .leases/ prefix as Publisher)
                    # to protect against concurrent publish.
                    lease_key = _acquire_gc_lease(
                        client, bucket, prefix, orphan_prefix, gc_owner
                    )
                    if lease_key is None:
                        logger.info(
                            "GC: skipping %s — lease held by another process",
                            orphan_prefix,
                        )
                        continue
                    manifest_key = f"{orphan_prefix}manifest.json"
                    try:
                        client.head_object(Bucket=bucket, Key=manifest_key)
                        logger.info(
                            "GC: skipping %s — manifest appeared after scan",
                            orphan_prefix,
                        )
                        continue
                    except client.exceptions.ClientError as exc:
                        if exc.response["Error"]["Code"] != "404":
                            raise
                    if not _check_orphan_age(
                        client,
                        bucket,
                        orphan_prefix,
                        orphan_ttl_seconds,
                    ):
                        logger.info(
                            "GC: skipping %s — a recent object appeared after scan",
                            orphan_prefix,
                        )
                        continue
                    _delete_prefix_safely(client, bucket, orphan_prefix)
                    deleted += 1
                    logger.info(
                        "GC: deleted orphan prefix s3://%s/%s", bucket, orphan_prefix
                    )
                except Exception as exc:
                    errors.append(f"DELETE {orphan_prefix}: {exc}")
                    logger.warning("GC: failed to delete %s: %s", orphan_prefix, exc)
                finally:
                    if lease_key is not None:
                        _release_gc_lease(client, bucket, lease_key, gc_owner)

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
        use_ssl=profile.use_ssl,
        path_style=profile.path_style,
        profile_name=profile.profile_name,
    )


def _acquire_gc_lease(
    client: Any, bucket: str, store_prefix: str, bundle_prefix: str, owner: str
) -> str | None:
    """Acquire the publish lease for *bundle_id* (same key as the Publisher).

    Using the same ``{store_prefix}.leases/{bundle_id}.json`` key as the
    Publisher gives true mutual exclusion:
    - A fresh publisher lease blocks GC entirely.
    - An expired lease is taken over via CAS (ETag If-Match) — matching
      the plan's "TTL GC 先 CAS 接管过期 lease" requirement.

    Returns the lease key if acquired, or ``None`` if a concurrent
    publisher holds a live lease.
    """
    # Extract bundle_id from the bundle prefix.
    bundle_id = bundle_prefix.rstrip("/").split("/")[-1]
    leases_prefix = f"{store_prefix}.leases/"
    try:
        lease_key = _s3_lease_acquire(
            client,
            bucket,
            leases_prefix,
            bundle_id,
            owner,
            ttl=_GC_LEASE_TTL,
        )
        return lease_key
    except BundleCommitBusyError:
        # Lease held by a live publisher until its TTL — skip this bundle.
        return None


def _release_gc_lease(client: Any, bucket: str, lease_key: str, owner: str) -> None:
    """Best-effort release of a GC lease."""
    _s3_lease_release(client, bucket, lease_key, owner)


def _looks_like_bundle_id(name: str) -> bool:
    """Return True if *name* matches the bundle ID format (bundle-<32 hex>).

    Real bundle IDs are exactly 39 characters: ``"bundle-"`` + 32 hex
    digits (see ``_make_bundle_id`` in ``tributo.exporting.service``).
    Prevents GC from scanning/deleting non-bundle prefixes.
    """
    if not name.startswith("bundle-") or len(name) != 39:
        return False
    return all(c in "0123456789abcdef" for c in name[7:])


def _check_orphan_age(client: Any, bucket: str, prefix: str, ttl_seconds: int) -> bool:
    """Check if all objects under *prefix* are older than *ttl_seconds*."""
    now = time.time()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
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
            response = client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
            )
            failures = response.get("Errors", [])
            if failures:
                details = ", ".join(
                    f"{failure.get('Key', '<unknown>')} "
                    f"({failure.get('Code', 'UNKNOWN')})"
                    for failure in failures[:10]
                )
                remaining = len(failures) - 10
                if remaining > 0:
                    details = f"{details}, and {remaining} more"
                raise RuntimeError(
                    f"S3 DeleteObjects reported {len(failures)} object failure(s): "
                    f"{details}"
                )


__all__ = ["S3BundleGarbageCollector"]
