"""Bundle reader — open, verify, and resolve published bundles.

Reads a bundle from local filesystem or S3, validates manifest integrity,
and provides ``ResolvedArtifact`` instances for runtime consumption.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from tributo._common.storage import (
    get_boto3_client,
    parse_s3_url,
)
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exporting.manifest import (
    ExportManifest,
    ManifestSchemaRegistry,
)
from tributo.exporting.models import (
    LogicalArtifact,
    ResolvedArtifact,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

# ── Resource limits (sane defaults, configurable) ──────────────────────────────


@PublicAPI(stability="beta")
class ReaderResourceLimits:
    """Resource limits enforced before download to prevent disk exhaustion."""

    def __init__(
        self,
        max_manifest_bytes: int = 10 * 1024 * 1024,  # 10 MB
        max_file_count: int = 256,
        max_single_file_bytes: int = 5 * 1024 * 1024 * 1024,  # 5 GB
        max_total_bytes: int = 50 * 1024 * 1024 * 1024,  # 50 GB
    ) -> None:
        self.max_manifest_bytes = max_manifest_bytes
        self.max_file_count = max_file_count
        self.max_single_file_bytes = max_single_file_bytes
        self.max_total_bytes = max_total_bytes


# ── BundleReader ─────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleReader:
    """Reads and validates published model bundles.

    Supports local filesystem and S3 bundles.  Enforces resource limits
    before downloading to prevent disk exhaustion from malicious manifests.
    """

    def __init__(
        self,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
        limits: ReaderResourceLimits | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()
        self._limits = limits or ReaderResourceLimits()
        self._cache_dir = (
            cache_dir or Path(tempfile.gettempdir()) / "tributo_bundle_cache"
        )

        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        from tributo.exporting.manifest import _read_manifest_v1

        try:
            self._manifest_registry.register(1, _read_manifest_v1)
        except ValueError:
            pass

    def read_manifest_with_bytes(
        self,
        manifest_or_bundle_uri: str,
        *,
        storage_profile: str | None = None,
    ) -> tuple[ExportManifest, bytes]:
        """Read a bundle manifest, returning ``(manifest, raw_bytes)``.

        ``raw_bytes`` is the exact manifest file content as published.
        Digest verification MUST compare against these bytes — never a
        re-serialisation of the parsed model, which changes as soon as
        new optional fields with defaults are added to the manifest.

        *manifest_or_bundle_uri* can be:
        - ``s3://bucket/prefix/manifest.json`` (exact manifest URI)
        - ``s3://bucket/prefix/{bundle_id}/`` (bundle root — appends ``manifest.json``)
        - ``s3://bucket/prefix/aliases/{name}.json`` (alias — resolves one hop)
        - A local path to a bundle root or manifest file.

        When resolving via an alias, the alias ``manifest_sha256`` is
        verified against the raw manifest bytes.
        """
        manifest_bytes, expected_sha256 = self._fetch_manifest_bytes(
            manifest_or_bundle_uri, storage_profile
        )
        raw = json.loads(manifest_bytes.decode("utf-8"))
        schema_version = raw.get("schema_version", 1)
        manifest = self._manifest_registry.read(schema_version, raw, manifest_bytes)

        # Verify manifest digest if we have an expected value (from alias).
        if expected_sha256 is not None:
            actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Manifest digest mismatch: expected {expected_sha256[:16]}..., "
                    f"got {actual_sha256[:16]}..."
                )

        return manifest, manifest_bytes

    def read_manifest(
        self,
        manifest_or_bundle_uri: str,
        *,
        storage_profile: str | None = None,
    ) -> ExportManifest:
        """Read and validate a bundle manifest.

        *manifest_or_bundle_uri* can be:
        - ``s3://bucket/prefix/manifest.json`` (exact manifest URI)
        - ``s3://bucket/prefix/{bundle_id}/`` (bundle root — appends ``manifest.json``)
        - ``s3://bucket/prefix/aliases/{name}.json`` (alias — resolves one hop)
        - A local path to a bundle root or manifest file.

        When resolving via an alias, the alias ``manifest_sha256`` is
        verified against the raw manifest bytes.
        """
        manifest, _ = self.read_manifest_with_bytes(
            manifest_or_bundle_uri, storage_profile=storage_profile
        )
        return manifest

    @contextmanager
    def open_artifact(
        self,
        manifest_or_bundle_uri: str,
        *,
        role: str | None = None,
        artifact_name: str | None = None,
        storage_profile: str | None = None,
        manifest: ExportManifest | None = None,
    ) -> Generator[ResolvedArtifact, None, None]:
        """Open a resolved artifact from a published bundle.

        Exactly one of *role* or *artifact_name* must be specified.

        *manifest* — when provided, the already-parsed manifest is reused
        instead of re-reading it from storage.  This is the TOCTOU guard
        for callers that validated the manifest first (e.g. the bundle
        model loader): the artifact is resolved against the *same*
        manifest snapshot that the validation used, so an S3 manifest
        changed between two reads can never make validation see A and
        loading see B.

        Context manager — the artifact's local files are only valid within
        the ``with`` block.  For S3 bundles, the temp directory is cleaned
        up on exit.
        """
        if (role is None) == (artifact_name is None):
            raise ValueError(
                "Exactly one of 'role' or 'artifact_name' must be specified"
            )

        if manifest is None:
            manifest = self.read_manifest(
                manifest_or_bundle_uri, storage_profile=storage_profile
            )

        # Resolve target artifact.
        target_name: str
        if role is not None:
            if role not in manifest.roles:
                raise ValueError(
                    f"Role {role!r} not found in bundle. Available roles: {list(manifest.roles)}"
                )
            target_name = manifest.roles[role]
        else:
            assert artifact_name is not None
            target_name = artifact_name

        # Find the artifact.
        matching = [a for a in manifest.artifacts if a.name == target_name]
        if not matching:
            available = [a.name for a in manifest.artifacts]
            raise ValueError(
                f"Artifact {target_name!r} not found in bundle. Available: {available}"
            )
        artifact = matching[0]

        # Enforce resource limits.
        self._enforce_limits(artifact)

        # Resolve to local directory.
        cleanup_root: Path | None = None
        if manifest_or_bundle_uri.startswith("s3://"):
            context_root = self._download_artifact_s3(
                manifest, artifact, storage_profile
            )
            artifact_dir = context_root / "artifact"
            cleanup_root = context_root
            is_temp = True
        else:
            # Local bundle.
            bundle_dir = _resolve_local_bundle_dir(
                _strip_file_scheme(manifest_or_bundle_uri)
            )
            artifact_dir = bundle_dir / "artifacts" / artifact.name
            is_temp = False

        # Verify integrity and yield inside the same try/finally so a
        # verification failure (tampered file, digest mismatch) also
        # removes the S3 temp root — never leak it on the rejection path.
        try:
            _verify_artifact_integrity(artifact, artifact_dir)
            ra = ResolvedArtifact(descriptor=artifact, root_dir=artifact_dir)
            yield ra
        finally:
            if is_temp and cleanup_root is not None and cleanup_root.exists():
                shutil.rmtree(cleanup_root, ignore_errors=True)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _fetch_manifest_bytes(
        self, uri: str, storage_profile: str | None
    ) -> tuple[bytes, str | None]:
        """Fetch manifest bytes, resolving aliases if needed.

        Returns ``(bytes, expected_sha256_or_None)``.
        """
        if uri.startswith("s3://"):
            return self._fetch_manifest_s3(uri, storage_profile)
        return self._fetch_manifest_local(_strip_file_scheme(uri))

    def _fetch_manifest_local(self, uri: str) -> tuple[bytes, str | None]:
        path = _resolve_manifest_path(Path(uri))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        data = path.read_bytes()
        if len(data) > self._limits.max_manifest_bytes:
            raise ValueError(
                f"Manifest size {len(data)} exceeds limit {self._limits.max_manifest_bytes}"
            )
        return data, None  # No expected digest for local reads.

    def _fetch_manifest_s3(
        self, uri: str, storage_profile: str | None
    ) -> tuple[bytes, str | None]:
        bucket, key = _resolve_s3_manifest_key(uri)
        client = _s3_client(self._storage_resolver, storage_profile)
        expected_sha256: str | None = None

        # Check if this is an alias (resolves one hop).
        if "aliases/" in key:
            alias_data = _s3_get_json(client, bucket, key)
            if alias_data is None:
                raise FileNotFoundError(f"Alias not found: s3://{bucket}/{key}")
            manifest_uri = alias_data.get("manifest_uri")
            if manifest_uri is None:
                raise ValueError(f"Alias s3://{bucket}/{key} has no manifest_uri")
            expected_sha256 = alias_data.get("manifest_sha256")
            if manifest_uri.startswith("s3://"):
                bucket, key = parse_s3_url(manifest_uri)
            else:
                raise ValueError(f"Unsupported alias target: {manifest_uri}")

        # Fetch manifest.
        resp = client.get_object(Bucket=bucket, Key=key)
        content_length = resp.get("ContentLength", 0)
        if content_length > self._limits.max_manifest_bytes:
            raise ValueError(
                f"Manifest size {content_length} exceeds limit {self._limits.max_manifest_bytes}"
            )
        data = resp["Body"].read()
        return data, expected_sha256

    def _download_artifact_s3(
        self,
        manifest: ExportManifest,
        artifact: LogicalArtifact,
        storage_profile: str | None,
    ) -> Path:
        """Download artifact files from S3 to a local cache directory.

        Downloads fill the shared cache (keyed by tree_digest) atomically —
        write-to-temp-then-rename — and the returned directory is a fresh
        per-context copy.  Consumers may delete their copy on exit without
        invalidating the shared cache for other contexts.

        Resource limits are enforced pre-download: the manifest carries
        declared sizes, and each object is HEAD-checked so a lying or
        mutated remote object is rejected before any bytes are fetched.
        """
        bucket, _key = parse_s3_url(manifest.canonical_uri)
        artifact_prefix = f"{_key.rstrip('/')}/artifacts/{artifact.name}/"

        # Cache keyed by tree_digest.
        cache_root = self._cache_dir / artifact.tree_digest
        cache_root.mkdir(parents=True, exist_ok=True)

        client = _s3_client(self._storage_resolver, storage_profile)
        expected_sizes = {af.relative_path: af.size_bytes for af in artifact.files}

        for af in artifact.files:
            s3_key = f"{artifact_prefix}{af.relative_path}"
            local_path = cache_root / af.relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if not local_path.exists():
                # HEAD pre-check: verify the remote size matches the manifest
                # before downloading.
                head = client.head_object(Bucket=bucket, Key=s3_key)
                remote_size = head.get("ContentLength", 0)
                expected = expected_sizes.get(af.relative_path)
                if expected is not None and remote_size != expected:
                    raise ValueError(
                        f"S3 object {s3_key} size {remote_size} does not "
                        f"match manifest size {expected}"
                    )
                if remote_size > self._limits.max_single_file_bytes:
                    raise ValueError(
                        f"S3 object {s3_key} size {remote_size} exceeds "
                        f"limit {self._limits.max_single_file_bytes}"
                    )

                # Atomic cache fill: download to temp, then rename.
                tmp_path = local_path.with_name(
                    f"{local_path.name}.tmp-{uuid.uuid4().hex[:8]}"
                )
                try:
                    client.download_file(bucket, s3_key, str(tmp_path))
                    os.replace(str(tmp_path), str(local_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)

        # Fresh per-context copy of the shared cache.  The caller
        # receives the full temp root so cleanup removes the parent
        # directory too, not just the artifact subtree.
        context_root = Path(tempfile.mkdtemp(prefix="tributo-bundle-artifact-"))
        context_dir = context_root / "artifact"
        shutil.copytree(cache_root, context_dir)
        return context_root

    def _enforce_limits(self, artifact: LogicalArtifact) -> None:
        """Check artifact against configured resource limits."""
        if len(artifact.files) > self._limits.max_file_count:
            raise ValueError(
                f"Artifact file count {len(artifact.files)} exceeds "
                f"limit {self._limits.max_file_count}"
            )
        total_bytes = sum(af.size_bytes for af in artifact.files)
        if total_bytes > self._limits.max_total_bytes:
            raise ValueError(
                f"Artifact total size {total_bytes} exceeds limit "
                f"{self._limits.max_total_bytes}"
            )
        for af in artifact.files:
            if af.size_bytes > self._limits.max_single_file_bytes:
                raise ValueError(
                    f"File {af.relative_path!r} size {af.size_bytes} exceeds "
                    f"limit {self._limits.max_single_file_bytes}"
                )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _s3_client(resolver: StorageProfileResolver, storage_profile: str | None) -> Any:
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


def _s3_get_json(client: Any, bucket: str, key: str) -> Any:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        raw: bytes = resp["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except client.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def _strip_file_scheme(uri: str) -> str:
    """Strip a ``file://`` scheme so local URIs work as plain paths.

    ``BundleOutputConfig`` accepts ``file://`` URIs; the reader treats
    everything that is not ``s3://`` as a local path.
    """
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def _resolve_manifest_path(path: Path) -> Path:
    """Resolve a local path to the manifest.json file."""
    p = path.resolve()
    if p.is_file():
        return p
    if p.is_dir():
        candidate = p / "manifest.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find manifest at {path}")


def _resolve_s3_manifest_key(uri: str) -> tuple[str, str]:
    """Resolve an S3 URI to (bucket, manifest_key).

    Handles:
    - Exact manifest path: ``s3://b/pre/manifest.json``
    - Bundle root: ``s3://b/pre/`` → ``s3://b/pre/manifest.json``
    - Alias: ``s3://b/pre/aliases/name.json`` (caller handles one-hop)
    """
    bucket, key = parse_s3_url(uri)
    if key.endswith("/"):
        key = f"{key}manifest.json"
    elif not key.endswith("manifest.json") and "aliases/" not in key:
        key = f"{key}/manifest.json"
    return bucket, key


def _resolve_local_bundle_dir(uri: str) -> Path:
    """Resolve a local URI to the bundle root directory."""
    p = _resolve_manifest_path(Path(uri))
    return p.parent


def _sha256_file_streaming(path: Path) -> str:
    """Compute SHA-256 of a file in chunks (safe for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_artifact_integrity(artifact: LogicalArtifact, artifact_dir: Path) -> None:
    """Verify all files exist, have correct sizes and SHA-256 digests."""
    for af in artifact.files:
        fp = (artifact_dir / af.relative_path).resolve()
        root = artifact_dir.resolve()
        if not fp.is_relative_to(root):
            raise ValueError(f"Path traversal: {af.relative_path!r}")
        if not fp.is_file():
            raise FileNotFoundError(f"Artifact file missing: {af.relative_path!r}")

        actual_size = fp.stat().st_size
        if actual_size != af.size_bytes:
            raise ValueError(
                f"File {af.relative_path!r}: expected {af.size_bytes} bytes, "
                f"got {actual_size}"
            )

        actual_sha = _sha256_file_streaming(fp)
        if actual_sha != af.sha256:
            raise ValueError(
                f"File {af.relative_path!r}: SHA-256 mismatch "
                f"(expected {af.sha256[:16]}..., got {actual_sha[:16]}...)"
            )

    # Verify tree digest.
    computed = LogicalArtifact.compute_tree_digest(artifact.files)
    if computed != artifact.tree_digest:
        raise ValueError(
            f"Tree digest mismatch: expected {artifact.tree_digest[:16]}..., "
            f"got {computed[:16]}..."
        )
