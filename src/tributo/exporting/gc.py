"""Storage-independent compatibility facade for bundle garbage collection."""

from __future__ import annotations

from typing import Any, Protocol

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.util.annotations import DeveloperAPI, PublicAPI


@DeveloperAPI
class BundleGarbageCollectorBackend(Protocol):
    """Storage adapter contract used by the public GC facade."""

    def collect(
        self,
        bundle_uri: str,
        *,
        storage_profile: str | None = None,
        orphan_ttl_seconds: int = 3600,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Collect orphan bundles owned by the adapter."""
        ...


@PublicAPI(stability="beta")
class BundleGarbageCollector:
    """Compatibility facade delegating storage work to a GC adapter."""

    def __init__(
        self,
        storage_resolver: StorageProfileResolver | None = None,
        *,
        backend: BundleGarbageCollectorBackend | None = None,
    ) -> None:
        if backend is None:
            from tributo._bootstrap import first_party_bundle_garbage_collector

            backend = first_party_bundle_garbage_collector(storage_resolver)
        self._backend = backend

    def collect(
        self,
        bundle_uri: str,
        *,
        storage_profile: str | None = None,
        orphan_ttl_seconds: int = 3600,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Collect orphan bundles through the configured adapter."""
        return self._backend.collect(
            bundle_uri,
            storage_profile=storage_profile,
            orphan_ttl_seconds=orphan_ttl_seconds,
            dry_run=dry_run,
        )


__all__ = ["BundleGarbageCollector", "BundleGarbageCollectorBackend"]
