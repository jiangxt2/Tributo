"""Contracts for storage-neutral post-publish hook adapters."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.events import OperationEvent
from tributo.exporting.manifest import ExportManifest, ManifestSchemaRegistry
from tributo.exporting.models import HookReceipt as HookReceipt
from tributo.exporting.models import HookStatus
from tributo.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    from tributo.exporting.models import BundleResult
    from tributo.exporting.records import OperationStore


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@PublicAPI(stability="beta")
class HookOutcome(BaseModel):
    """Adapter result before delivery metadata is attached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: HookStatus
    error_code: str | None = None
    error_summary: str | None = Field(default=None, max_length=4096)
    external_references: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
@DeveloperAPI
class ArtifactAccessor(Protocol):
    """Read and materialize the exact committed bundle named by an event."""

    def read_manifest(self) -> ExportManifest: ...

    def materialize_manifest(self) -> AbstractContextManager[Path]: ...

    def materialize_bundle(self) -> AbstractContextManager[Path]: ...


@DeveloperAPI
class BundleArtifactAccessor:
    """BundleReader-backed accessor with digest verification and cleanup."""

    def __init__(
        self,
        event: OperationEvent,
        *,
        storage_profile: str | None = None,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
        manifest: ExportManifest | None = None,
        manifest_bytes: bytes | None = None,
    ) -> None:
        self._event = event
        self._storage_profile = storage_profile
        self._reader = BundleReader(
            storage_resolver=storage_resolver,
            manifest_registry=manifest_registry,
        )
        self._manifest: ExportManifest | None = None
        self._manifest_bytes: bytes | None = None
        if (manifest is None) != (manifest_bytes is None):
            raise ValueError("manifest and manifest_bytes must be provided together")
        if manifest is not None and manifest_bytes is not None:
            self._cache_verified_manifest(manifest, manifest_bytes)

    def _cache_verified_manifest(
        self, manifest: ExportManifest, manifest_bytes: bytes
    ) -> None:
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != self._event.manifest_sha256:
            raise ValueError(
                "Committed manifest digest does not match the publication event"
            )
        if manifest.bundle_id != self._event.bundle_id:
            raise ValueError("Committed manifest bundle_id does not match the event")
        self._manifest = manifest
        self._manifest_bytes = manifest_bytes

    def read_manifest(self) -> ExportManifest:
        """Read the committed manifest and verify its event digest."""
        if self._manifest is None:
            manifest, raw = self._reader.read_manifest_with_bytes(
                self._event.canonical_uri,
                storage_profile=self._storage_profile,
            )
            self._cache_verified_manifest(manifest, raw)
        assert self._manifest is not None
        return self._manifest

    @contextmanager
    def materialize_manifest(self) -> Generator[Path, None, None]:
        """Yield the exact verified manifest bytes as a managed local file."""
        self.read_manifest()
        assert self._manifest_bytes is not None

        root = Path(tempfile.mkdtemp(prefix="tributo-committed-manifest-"))
        manifest_path = root / "manifest.json"
        try:
            manifest_path.write_bytes(self._manifest_bytes)
            yield manifest_path
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @contextmanager
    def materialize_bundle(self) -> Generator[Path, None, None]:
        """Yield a verified local bundle layout and clean remote materialization."""
        manifest = self.read_manifest()
        assert self._manifest_bytes is not None

        root = Path(tempfile.mkdtemp(prefix="tributo-committed-bundle-"))
        try:
            (root / "manifest.json").write_bytes(self._manifest_bytes)
            artifacts_root = (root / "artifacts").resolve()
            for artifact in manifest.artifacts:
                artifact_root = (artifacts_root / artifact.name).resolve()
                if artifact_root == artifacts_root or not artifact_root.is_relative_to(
                    artifacts_root
                ):
                    raise ValueError(
                        f"Artifact name escapes materialization root: {artifact.name!r}"
                    )
                with self._reader.open_artifact(
                    self._event.canonical_uri,
                    artifact_name=artifact.name,
                    storage_profile=self._storage_profile,
                    manifest=manifest,
                ) as resolved:
                    for artifact_file in artifact.files:
                        source = resolved.path_for(artifact_file.relative_path)
                        destination = artifact_root / artifact_file.relative_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
                        if destination.stat().st_size != artifact_file.size_bytes:
                            raise ValueError(
                                f"Copied artifact size changed for "
                                f"{artifact_file.relative_path!r}"
                            )
                        actual = _sha256_file(destination)
                        if actual != artifact_file.sha256:
                            raise ValueError(
                                f"Copied artifact digest changed for "
                                f"{artifact_file.relative_path!r}"
                            )
            yield root
        finally:
            shutil.rmtree(root, ignore_errors=True)


@runtime_checkable
@PublicAPI(stability="beta")
class PublicationHook(Protocol):
    """Adapter contract for one versioned external side effect."""

    api_version: ClassVar[int]
    hook_id: ClassVar[str]
    options_model: ClassVar[type[BaseModel]]

    def deliver(
        self,
        event: OperationEvent,
        artifacts: ArtifactAccessor,
        options: BaseModel,
    ) -> HookOutcome:
        """Deliver one committed publication event."""
        ...

    def idempotency_key(self, event: OperationEvent, options: BaseModel) -> str:
        """Return a deterministic key for this external side effect."""
        ...


# Descriptive alias used by the new dispatcher API.  The historical name stays
# importable for the beta compatibility window.
PostPublishHook = PublicationHook


@PublicAPI(stability="beta")
class PublicationRunner:
    """Compatibility entry point delegating execution to InlineHookDispatcher."""

    def __init__(
        self,
        hooks: list[tuple[PublicationHook, BaseModel, bool]],
        operation_store: OperationStore | None = None,
    ) -> None:
        from tributo.exporting.dispatch import InlineHookDispatcher, PreparedHook

        self._prepared = tuple(
            PreparedHook(adapter=adapter, options=options, required=required)
            for adapter, options, required in hooks
        )
        self._dispatcher = InlineHookDispatcher(operation_store)

    def run(
        self,
        *,
        event: OperationEvent,
        artifacts: ArtifactAccessor,
        bundle_result: BundleResult,
        bundle_digest: str,
    ) -> BundleResult:
        """Run prepared adapters through the shared dispatcher implementation."""
        return self._dispatcher.dispatch_prepared(
            event=event,
            bundle_result=bundle_result,
            bundle_digest=bundle_digest,
            prepared_hooks=self._prepared,
            artifacts=artifacts,
        )
