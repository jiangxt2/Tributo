"""Storage-independent bundle reader facade."""

from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exporting.manifest import ExportManifest, ManifestSchemaRegistry
from tributo.exporting.models import BundleRef, LogicalArtifact, ResolvedArtifact
from tributo.exporting.repository import (
    BundleRepositoryRouter,
    ReaderResourceLimits,
    build_default_repository_router,
    enforce_artifact_limits,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class BundleReader:
    """Read and verify bundles through storage-independent repository ports."""

    def __init__(
        self,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
        limits: ReaderResourceLimits | None = None,
        cache_dir: Path | None = None,
        repository_router: BundleRepositoryRouter | None = None,
    ) -> None:
        self._limits = limits or ReaderResourceLimits()
        self._cache_dir = (
            cache_dir or Path(tempfile.gettempdir()) / "tributo_bundle_cache"
        )
        self._repository_router = repository_router or build_default_repository_router(
            storage_resolver
        )
        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        from tributo.exporting.manifest import _read_manifest_v1, _read_manifest_v2

        try:
            self._manifest_registry.register(1, _read_manifest_v1)
        except ValueError:
            pass
        try:
            self._manifest_registry.register(2, _read_manifest_v2)
        except ValueError:
            pass

    def read_manifest_with_bytes(
        self,
        manifest_or_bundle_uri: BundleRef | str,
        *,
        storage_profile: str | None = None,
    ) -> tuple[ExportManifest, bytes]:
        """Read a manifest and return its exact committed bytes."""
        manifest_uri, expected_ref = self._resolve_bundle_location(
            manifest_or_bundle_uri,
            storage_profile=storage_profile,
        )
        return self._read_manifest_from_location(
            manifest_uri,
            expected_ref=expected_ref,
            storage_profile=storage_profile,
        )

    def _read_manifest_from_location(
        self,
        manifest_uri: str,
        *,
        expected_ref: BundleRef | None,
        storage_profile: str | None,
    ) -> tuple[ExportManifest, bytes]:
        """Read a previously resolved immutable bundle location."""
        repository = self._repository_router.repository_for(manifest_uri)
        manifest, manifest_bytes = repository.read_manifest(
            manifest_uri,
            storage_profile=storage_profile,
            max_manifest_bytes=self._limits.max_manifest_bytes,
            manifest_registry=self._manifest_registry,
        )

        if expected_ref is not None:
            actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if actual_sha256 != expected_ref.manifest_sha256:
                raise ValueError(
                    "Manifest digest mismatch: expected "
                    f"{expected_ref.manifest_sha256[:16]}..., got "
                    f"{actual_sha256[:16]}..."
                )
            if manifest.bundle_id != expected_ref.bundle_id:
                raise ValueError(
                    f"Reference bundle_id {expected_ref.bundle_id!r} does not match "
                    f"manifest bundle_id {manifest.bundle_id!r}"
                )

        return manifest, manifest_bytes

    def read_manifest(
        self,
        manifest_or_bundle_uri: BundleRef | str,
        *,
        storage_profile: str | None = None,
    ) -> ExportManifest:
        """Read and validate a bundle manifest."""
        manifest, _ = self.read_manifest_with_bytes(
            manifest_or_bundle_uri,
            storage_profile=storage_profile,
        )
        return manifest

    @contextmanager
    def open_artifact(
        self,
        manifest_or_bundle_uri: BundleRef | str,
        *,
        role: str | None = None,
        artifact_name: str | None = None,
        storage_profile: str | None = None,
        manifest: ExportManifest | None = None,
        manifest_bytes: bytes | None = None,
    ) -> Generator[ResolvedArtifact, None, None]:
        """Resolve and verify exactly one artifact within a bounded context."""
        if (role is None) == (artifact_name is None):
            raise ValueError(
                "Exactly one of 'role' or 'artifact_name' must be specified"
            )

        if (manifest is None) != (manifest_bytes is None):
            raise ValueError("manifest and manifest_bytes must be provided together")

        if manifest is None:
            bundle_uri, expected_ref = self._resolve_bundle_location(
                manifest_or_bundle_uri,
                storage_profile=storage_profile,
            )
            manifest, manifest_bytes = self._read_manifest_from_location(
                bundle_uri,
                expected_ref=expected_ref,
                storage_profile=storage_profile,
            )
        else:
            assert manifest_bytes is not None
            parsed_manifest = self._parse_manifest_bytes(manifest_bytes)
            if parsed_manifest != manifest:
                raise ValueError(
                    "Provided manifest does not match the exact manifest bytes"
                )
            if isinstance(manifest_or_bundle_uri, BundleRef):
                expected_ref = manifest_or_bundle_uri
                bundle_uri = expected_ref.canonical_uri
                self._verify_expected_ref(parsed_manifest, manifest_bytes, expected_ref)
            else:
                # A supplied snapshot has already passed any alias CAS/digest
                # checks.  Detect alias syntax locally and use the immutable
                # URI recorded by that exact snapshot without fetching the
                # alias a second time (which would create a TOCTOU window).
                alias_store = self._repository_router.alias_store_for(
                    manifest_or_bundle_uri
                )
                bundle_uri = (
                    parsed_manifest.canonical_uri
                    if alias_store.is_alias_uri(manifest_or_bundle_uri)
                    else manifest_or_bundle_uri
                )

        assert manifest is not None
        assert manifest_bytes is not None

        target_name = self._resolve_target_name(
            manifest, role=role, artifact_name=artifact_name
        )
        artifact = self._find_artifact(manifest, target_name)
        enforce_artifact_limits(artifact, self._limits)

        repository = self._repository_router.repository_for(bundle_uri)
        with repository.materialize_artifact(
            bundle_uri,
            manifest,
            artifact,
            storage_profile=storage_profile,
            limits=self._limits,
            cache_dir=self._cache_dir,
        ) as resolved:
            if resolved.descriptor != artifact:
                raise ValueError(
                    "Repository materialized an artifact descriptor that does "
                    "not match the committed manifest"
                )
            yield resolved

    def _parse_manifest_bytes(self, manifest_bytes: bytes) -> ExportManifest:
        """Parse exact manifest bytes with the same bounds as repository reads."""
        if len(manifest_bytes) > self._limits.max_manifest_bytes:
            raise ValueError(
                f"Manifest size {len(manifest_bytes)} exceeds limit "
                f"{self._limits.max_manifest_bytes}"
            )
        try:
            raw = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Manifest is not valid UTF-8 JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("Manifest must contain a JSON object")
        return self._manifest_registry.read(
            raw.get("schema_version", 1), raw, manifest_bytes
        )

    @staticmethod
    def _verify_expected_ref(
        manifest: ExportManifest, manifest_bytes: bytes, expected_ref: BundleRef
    ) -> None:
        actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_sha256 != expected_ref.manifest_sha256:
            raise ValueError(
                "Manifest digest mismatch: expected "
                f"{expected_ref.manifest_sha256[:16]}..., got "
                f"{actual_sha256[:16]}..."
            )
        if manifest.bundle_id != expected_ref.bundle_id:
            raise ValueError(
                f"Reference bundle_id {expected_ref.bundle_id!r} does not match "
                f"manifest bundle_id {manifest.bundle_id!r}"
            )

    def _resolve_bundle_location(
        self,
        manifest_or_bundle_uri: BundleRef | str,
        *,
        storage_profile: str | None,
    ) -> tuple[str, BundleRef | None]:
        """Return the effective immutable bundle URI and expected reference."""
        if isinstance(manifest_or_bundle_uri, BundleRef):
            return manifest_or_bundle_uri.canonical_uri, manifest_or_bundle_uri
        expected_ref = self._repository_router.resolve_alias(
            manifest_or_bundle_uri,
            storage_profile=storage_profile,
            max_alias_bytes=self._limits.max_manifest_bytes,
        )
        if expected_ref is not None:
            return expected_ref.canonical_uri, expected_ref
        return manifest_or_bundle_uri, None

    @staticmethod
    def _resolve_target_name(
        manifest: ExportManifest,
        *,
        role: str | None,
        artifact_name: str | None,
    ) -> str:
        if role is None:
            assert artifact_name is not None
            return artifact_name
        if role not in manifest.roles:
            raise ValueError(
                f"Role {role!r} not found in bundle. "
                f"Available roles: {list(manifest.roles)}"
            )
        return manifest.roles[role]

    @staticmethod
    def _find_artifact(manifest: ExportManifest, target_name: str) -> LogicalArtifact:
        for artifact in manifest.artifacts:
            if artifact.name == target_name:
                return artifact
        available = [artifact.name for artifact in manifest.artifacts]
        raise ValueError(
            f"Artifact {target_name!r} not found in bundle. Available: {available}"
        )


__all__ = ["BundleReader", "ReaderResourceLimits"]
