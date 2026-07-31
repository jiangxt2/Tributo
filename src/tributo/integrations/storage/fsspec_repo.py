"""fsspec-backed BundleRepository — works with any fsspec-compatible filesystem.

Supporting S3, GCS, Azure Blob, HDFS, and 40+ other filesystems through
the fsspec abstraction layer.  Uses atomic ``write-to-tmp-then-rename``
where the backend supports it (local, GCS, Azure); falls back to direct
write for backends without atomic rename.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from tributo.exporting.models import FailureInfo
from tributo.exporting.repository import (
    AliasUpdateResult,
    BundleRef,
    BundleRepository,
    CommitResult,
    UncommittedBundle,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class FsspecBundleRepository:
    """Commit bundles to any fsspec-compatible filesystem.

    Built on ``fsspec`` — supports S3, GCS, Azure, HDFS, SFTP, and more
    through a single implementation.

    Args:
        root_uri: Root URI for the bundle store (e.g. ``gs://my-bucket/bundles``,
            ``abfs://container/bundles``, ``hdfs://namenode:8020/bundles``).
        fs_kwargs: Extra keyword arguments passed to ``fsspec.filesystem()``.
    """

    def __init__(
        self,
        root_uri: str,
        **fs_kwargs: Any,
    ) -> None:
        import fsspec

        self._root_uri = root_uri.rstrip("/")
        self._fs, self._root_path = fsspec.core.url_to_fs(
            root_uri, **fs_kwargs
        )
        self._fs_kwargs = fs_kwargs
        self._supports_atomic = _check_atomic_support(self._fs)

    def commit(self, bundle: UncommittedBundle) -> CommitResult:
        """Atomically persist *bundle*."""
        bundle_id = bundle.bundle_id
        bundle_prefix = f"{self._root_path}/{bundle_id}"

        # Check idempotency.
        manifest_path = f"{bundle_prefix}/manifest.json"
        if self._fs.exists(manifest_path):
            existing_bytes = self._fs.cat(manifest_path)
            existing_raw = json.loads(existing_bytes)
            existing_digests = sorted(
                a.get("tree_digest", "")
                for a in existing_raw.get("artifacts", [])
            )
            candidate_digests = sorted(
                a.tree_digest for a in bundle.artifacts
            )
            if (
                existing_raw.get("bundle_id") == bundle_id
                and existing_digests == candidate_digests
            ):
                existing_sha = hashlib.sha256(existing_bytes).hexdigest()
                return CommitResult(
                    canonical_uri=f"{self._root_uri}/{bundle_id}",
                    manifest_uri=f"{self._root_uri}/{bundle_id}/manifest.json",
                    manifest_sha256=existing_sha,
                    commit_status="idempotent",
                )
            raise RuntimeError(
                f"Bundle {bundle_prefix} exists with different content"
            )

        # Upload artifacts.
        for artifact in bundle.artifacts:
            artifact_src = bundle.staging_root / "nodes" / artifact.name / "artifact"
            for af in artifact.files:
                src_file = artifact_src / af.relative_path
                dst = f"{bundle_prefix}/artifacts/{artifact.name}/{af.relative_path}"
                self._fs.makedirs(Path(dst).parent.as_posix(), exist_ok=True)
                _atomic_put(self._fs, src_file, dst, self._supports_atomic)

        # Write manifest.
        self._fs.makedirs(bundle_prefix, exist_ok=True)
        _atomic_put_bytes(
            self._fs, bundle.manifest_bytes, manifest_path, self._supports_atomic
        )

        return CommitResult(
            canonical_uri=f"{self._root_uri}/{bundle_id}",
            manifest_uri=f"{self._root_uri}/{bundle_id}/manifest.json",
            manifest_sha256=bundle.manifest_sha256,
            commit_status="committed",
        )

    def get(self, ref: BundleRef) -> dict[str, Any]:
        """Read the manifest for *ref*."""
        manifest_path = f"{self._root_path}/{ref.bundle_id}/manifest.json"
        if not self._fs.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        raw = json.loads(self._fs.cat(manifest_path))
        return raw  # type: ignore[no-any-return]

    def update_alias(
        self,
        alias: str,
        new_ref: BundleRef,
        expected_revision: str | None = None,
    ) -> AliasUpdateResult:
        """Write an alias file.

        Note: fsspec backends vary in atomicity guarantees.  This method
        uses write-to-tmp-then-rename where supported.
        """
        alias_path = f"{self._root_path}/aliases/{alias}.json"

        alias_data = {
            "manifest_sha256": new_ref.manifest_sha256,
            "canonical_uri": new_ref.canonical_uri,
            "bundle_id": new_ref.bundle_id,
        }
        alias_bytes = json.dumps(alias_data, indent=2).encode("utf-8")

        # CAS check.
        if expected_revision is not None:
            if self._fs.exists(alias_path):
                current = json.loads(self._fs.cat(alias_path))
                current_sha = current.get("manifest_sha256", "")
                if current_sha != expected_revision:
                    return AliasUpdateResult(
                        alias=alias, status="failed",
                        failure=FailureInfo(
                            code="CAS_MISMATCH",
                            category="publish",
                            message="Expected manifest_sha256 does not match",
                        ),
                    )

        self._fs.makedirs(f"{self._root_path}/aliases", exist_ok=True)
        _atomic_put_bytes(self._fs, alias_bytes, alias_path, self._supports_atomic)

        return AliasUpdateResult(alias=alias, status="updated")


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _check_atomic_support(fs: Any) -> bool:
    """Check if the filesystem supports atomic rename (tmp → target)."""
    # Standard filesystem-like backends support rename.
    # HTTP-based ones (like some WebHDFS configs) do not.
    protocol = getattr(fs, "protocol", "")
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0] if protocol else ""
    # Local, GCS, Azure (abfs), HDFS, SFTP support rename.
    non_atomic = {"http", "https", "webhdfs"}
    return protocol not in non_atomic


def _atomic_put(fs: Any, src_file: Path, dst: str, supports_atomic: bool) -> None:
    """Put a local file to fsspec, atomically if supported."""
    if supports_atomic:
        tmp_dst = f"{dst}.tmp"
        fs.put(str(src_file), tmp_dst)
        fs.rm(dst, recursive=False) if fs.exists(dst) else None
        fs.mv(tmp_dst, dst)
    else:
        fs.put(str(src_file), dst)


def _atomic_put_bytes(
    fs: Any, data: bytes, dst: str, supports_atomic: bool,
) -> None:
    """Put bytes to fsspec, atomically if supported."""
    if supports_atomic:
        tmp_dst = f"{dst}.tmp"
        with fs.open(tmp_dst, "wb") as f:
            f.write(data)
        fs.rm(dst, recursive=False) if fs.exists(dst) else None
        fs.mv(tmp_dst, dst)
    else:
        with fs.open(dst, "wb") as f:
            f.write(data)
