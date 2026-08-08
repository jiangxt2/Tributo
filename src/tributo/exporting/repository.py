"""Storage-independent bundle repository contracts and routing."""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

from tributo.exporting.manifest import ExportManifest, ManifestSchemaRegistry
from tributo.exporting.models import (
    AliasConfig,
    BundleRef,
    FailureInfo,
    LogicalArtifact,
    ResolvedArtifact,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ReaderResourceLimits:
    """Resource limits enforced before repository materialization."""

    max_manifest_bytes: int = 10 * 1024 * 1024
    max_file_count: int = 256
    max_single_file_bytes: int = 5 * 1024 * 1024 * 1024
    max_total_bytes: int = 50 * 1024 * 1024 * 1024


@DeveloperAPI
@dataclass(frozen=True)
class StagedBundle:
    """Storage-neutral immutable bundle assembled before commit."""

    bundle_id: str
    execution_id: str
    manifest: ExportManifest
    manifest_bytes: bytes
    manifest_sha256: str
    staging_root: Path


@DeveloperAPI
@dataclass(frozen=True)
class RepositoryCommit:
    """Repository result after an immutable bundle commit."""

    bundle_ref: BundleRef
    manifest_uri: str
    manifest_bytes: bytes
    local_bundle_dir: Path
    local_dir_ephemeral: bool


@DeveloperAPI
@dataclass(frozen=True)
class AliasUpdate:
    """Storage-level alias update result."""

    alias_uri: str | None = None
    status: str = "not_requested"
    failure: FailureInfo | None = None


@runtime_checkable
@PublicAPI(stability="beta")
class BundleRepository(Protocol):
    """Commit, read, and materialize immutable bundles for URI schemes.

    Adapter classes discovered through entry points must accept the keyword
    constructor argument ``storage_resolver``. It may be ignored by adapters
    that do not use Tributo storage profiles.
    """

    repository_id: ClassVar[str]
    schemes: ClassVar[tuple[str, ...]]

    def commit(
        self,
        staged_bundle: StagedBundle,
        *,
        bundle_uri: str,
        storage_profile: str | None,
    ) -> RepositoryCommit:
        """Atomically commit *staged_bundle* or succeed idempotently."""
        ...

    def read_manifest(
        self,
        bundle_uri: str,
        *,
        storage_profile: str | None,
        max_manifest_bytes: int,
        manifest_registry: ManifestSchemaRegistry,
    ) -> tuple[ExportManifest, bytes]:
        """Return the parsed manifest and exact committed bytes."""
        ...

    def materialize_artifact(
        self,
        bundle_uri: str,
        manifest: ExportManifest,
        artifact: LogicalArtifact,
        *,
        storage_profile: str | None,
        limits: ReaderResourceLimits,
        cache_dir: Path,
    ) -> AbstractContextManager[ResolvedArtifact]:
        """Materialize and verify one artifact inside a bounded context.

        The adapter must validate file sizes, file digests, and the tree digest
        before yielding the resolved artifact.
        """
        ...


@runtime_checkable
@PublicAPI(stability="beta")
class BundleAliasStore(Protocol):
    """Resolve and compare-and-set aliases in a storage namespace."""

    alias_store_id: ClassVar[str]
    schemes: ClassVar[tuple[str, ...]]

    def is_alias_uri(self, uri: str) -> bool:
        """Return whether *uri* addresses a storage-level alias."""
        ...

    def resolve(
        self,
        alias_uri: str,
        *,
        storage_profile: str | None,
        max_alias_bytes: int,
    ) -> BundleRef:
        """Resolve an alias v1 document to an immutable bundle reference."""
        ...

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
        """Apply ``newer`` or compare-and-set semantics."""
        ...


@DeveloperAPI
class BundleRepositoryRouter:
    """Select repository and alias adapters without technology branches in Core."""

    def __init__(
        self,
        repositories: tuple[BundleRepository, ...],
        alias_stores: tuple[BundleAliasStore, ...],
    ) -> None:
        self._repositories: dict[str, BundleRepository] = _index_adapters(
            repositories, "repository"
        )
        self._alias_stores: dict[str, BundleAliasStore] = _index_adapters(
            alias_stores, "alias store"
        )

    def repository_for(self, uri: str) -> BundleRepository:
        """Return the repository registered for *uri*."""
        scheme = _uri_scheme(uri)
        try:
            return self._repositories[scheme]
        except KeyError as exc:
            raise ValueError(
                f"No BundleRepository registered for URI scheme {scheme!r}. "
                f"Registered schemes: {sorted(self._repositories)}"
            ) from exc

    def alias_store_for(self, uri: str) -> BundleAliasStore:
        """Return the alias store registered for *uri*."""
        scheme = _uri_scheme(uri)
        try:
            return self._alias_stores[scheme]
        except KeyError as exc:
            raise ValueError(
                f"No BundleAliasStore registered for URI scheme {scheme!r}. "
                f"Registered schemes: {sorted(self._alias_stores)}"
            ) from exc

    def resolve_alias(
        self,
        uri: str,
        *,
        storage_profile: str | None,
        max_alias_bytes: int,
    ) -> BundleRef | None:
        """Resolve *uri* when it is a storage alias, otherwise return ``None``."""
        store = self._alias_stores.get(_uri_scheme(uri))
        if store is None:
            return None
        if not store.is_alias_uri(uri):
            return None
        return store.resolve(
            uri,
            storage_profile=storage_profile,
            max_alias_bytes=max_alias_bytes,
        )


def verify_materialized_artifact(artifact: LogicalArtifact, artifact_dir: Path) -> None:
    """Verify paths, sizes, file digests, and the logical tree digest."""
    root = artifact_dir.resolve()
    for artifact_file in artifact.files:
        path = (root / artifact_file.relative_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path traversal: {artifact_file.relative_path!r}")
        if not path.is_file():
            raise FileNotFoundError(
                f"Artifact file missing: {artifact_file.relative_path!r}"
            )
        actual_size = path.stat().st_size
        if actual_size != artifact_file.size_bytes:
            raise ValueError(
                f"File {artifact_file.relative_path!r}: expected "
                f"{artifact_file.size_bytes} bytes, got {actual_size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != artifact_file.sha256:
            raise ValueError(
                f"File {artifact_file.relative_path!r}: SHA-256 mismatch "
                f"(expected {artifact_file.sha256[:16]}..., got "
                f"{actual_sha256[:16]}...)"
            )
    computed_tree_digest = LogicalArtifact.compute_tree_digest(artifact.files)
    if computed_tree_digest != artifact.tree_digest:
        raise ValueError(
            f"Tree digest mismatch: expected {artifact.tree_digest[:16]}..., "
            f"got {computed_tree_digest[:16]}..."
        )


def enforce_artifact_limits(
    artifact: LogicalArtifact, limits: ReaderResourceLimits
) -> None:
    """Enforce materialization limits at both facade and adapter boundaries."""
    if len(artifact.files) > limits.max_file_count:
        raise ValueError(
            f"Artifact file count {len(artifact.files)} exceeds "
            f"limit {limits.max_file_count}"
        )
    total_bytes = sum(file.size_bytes for file in artifact.files)
    if total_bytes > limits.max_total_bytes:
        raise ValueError(
            f"Artifact total size {total_bytes} exceeds limit {limits.max_total_bytes}"
        )
    for file in artifact.files:
        if file.size_bytes > limits.max_single_file_bytes:
            raise ValueError(
                f"File {file.relative_path!r} size {file.size_bytes} exceeds "
                f"limit {limits.max_single_file_bytes}"
            )


_AdapterT = TypeVar("_AdapterT", BundleRepository, BundleAliasStore)


def _index_adapters(
    adapters: tuple[_AdapterT, ...], label: str
) -> dict[str, _AdapterT]:
    indexed: dict[str, _AdapterT] = {}
    for adapter in adapters:
        schemes = getattr(adapter, "schemes", ())
        for scheme in schemes:
            if scheme in indexed:
                raise ValueError(f"Duplicate {label} for URI scheme {scheme!r}")
            indexed[scheme] = adapter
    return indexed


def _uri_scheme(uri: str) -> str:
    if uri.startswith("s3://"):
        return "s3"
    if uri.startswith("file://") or "://" not in uri:
        return "file"
    return uri.split("://", 1)[0].lower()


def build_default_repository_router(
    storage_resolver: object | None = None,
) -> BundleRepositoryRouter:
    """Build first-party adapters plus non-conflicting extension plugins."""
    from tributo._bootstrap import first_party_bundle_storage_adapters
    from tributo._common.storage_profiles import StorageProfileResolver
    from tributo.plugin import (
        discover_bundle_alias_store_plugins,
        discover_bundle_repository_plugins,
    )

    if storage_resolver is not None and not isinstance(
        storage_resolver, StorageProfileResolver
    ):
        raise TypeError("storage_resolver must be a StorageProfileResolver")
    builtin_repositories, builtin_alias_stores = first_party_bundle_storage_adapters(
        storage_resolver
    )
    builtin_repository_ids = {
        repository.repository_id for repository in builtin_repositories
    }
    builtin_alias_store_ids = {
        alias_store.alias_store_id for alias_store in builtin_alias_stores
    }
    plugin_repositories = tuple(
        _instantiate_repository(cls, storage_resolver)
        for cls in discover_bundle_repository_plugins()
        if cls.repository_id not in builtin_repository_ids
    )
    plugin_alias_stores = tuple(
        _instantiate_alias_store(cls, storage_resolver)
        for cls in discover_bundle_alias_store_plugins()
        if cls.alias_store_id not in builtin_alias_store_ids
    )
    repositories = builtin_repositories + plugin_repositories
    alias_stores = builtin_alias_stores + plugin_alias_stores
    return BundleRepositoryRouter(repositories, alias_stores)


def _instantiate_repository(
    adapter_class: type[Any], resolver: object | None
) -> BundleRepository:
    return cast(BundleRepository, adapter_class(storage_resolver=resolver))


def _instantiate_alias_store(
    adapter_class: type[Any], resolver: object | None
) -> BundleAliasStore:
    return cast(BundleAliasStore, adapter_class(storage_resolver=resolver))


__all__ = [
    "AliasUpdate",
    "BundleAliasStore",
    "BundleRepository",
    "BundleRepositoryRouter",
    "ReaderResourceLimits",
    "RepositoryCommit",
    "StagedBundle",
]
