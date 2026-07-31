"""Local filesystem bundle repository.

Atomic commit via per-file fsync + atomic directory rename.
stdlib only — no external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from tributo.exporting.models import FailureInfo
from tributo.exporting.repository import (
    AliasUpdateResult,
    BundleRef,
    CommitResult,
    UncommittedBundle,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


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
            with open(target, "rb") as f:
                os.fsync(f.fileno())
        _fsync_dir(target.parent)


@PublicAPI(stability="beta")
class LocalBundleRepository:
    """Commit bundles to a local directory with atomic rename.

    Implements the ``BundleRepository`` protocol using only stdlib:
    per-file fsync for durability and ``os.rename`` for atomicity.
    """

    def commit(self, bundle: UncommittedBundle) -> CommitResult:
        """Commit *bundle* to a local directory."""
        bundle_uri = bundle.manifest.get("canonical_uri", "")
        bundle_id = bundle.bundle_id

        # Parse bundle directory from canonical_uri.
        # canonical_uri is typically <bundle_uri>/<bundle_id>.
        if bundle_uri:
            final_dir = Path(bundle_uri).resolve()
        else:
            final_dir = bundle.staging_root / bundle_id

        parent = final_dir.parent
        parent.mkdir(parents=True, exist_ok=True)

        # Idempotency check.
        if final_dir.exists():
            existing_manifest_path = final_dir / "manifest.json"
            if not existing_manifest_path.is_file():
                raise RuntimeError(
                    f"Bundle directory {final_dir} exists but has no manifest"
                )
            existing_bytes = existing_manifest_path.read_bytes()
            existing_raw = json.loads(existing_bytes)
            existing_bundle_id = existing_raw.get("bundle_id")
            existing_artifacts = existing_raw.get("artifacts", [])
            candidate_artifacts = json.loads(bundle.manifest_bytes).get(
                "artifacts", []
            )
            existing_digests = sorted(
                a.get("tree_digest", "") for a in existing_artifacts
            )
            candidate_digests = sorted(
                a.get("tree_digest", "") for a in candidate_artifacts
            )

            if (
                existing_bundle_id == bundle_id
                and existing_digests == candidate_digests
            ):
                logger.info("Bundle %s already exists — idempotent", final_dir)
                manifest_sha256 = hashlib.sha256(existing_bytes).hexdigest()
                return CommitResult(
                    canonical_uri=str(final_dir),
                    manifest_uri=str(final_dir / "manifest.json"),
                    manifest_sha256=manifest_sha256,
                    commit_status="idempotent",
                )
            raise RuntimeError(
                f"Bundle directory {final_dir} exists with different content"
            )

        # Create temp directory in the same filesystem for atomic rename.
        temp_dir = parent / f".tmp-{bundle_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            temp_dir.mkdir(parents=True)

            # Copy artifacts from staging.
            artifacts_src = bundle.staging_root / "nodes"
            if artifacts_src.exists():
                artifacts_dst = temp_dir / "artifacts"
                for af in bundle.artifacts:
                    artifact_src = artifacts_src / af.name / "artifact"
                    artifact_dst = artifacts_dst / af.name
                    _copy_tree_fsync(artifact_src, artifact_dst)

            # Write manifest.
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_bytes(bundle.manifest_bytes)
            with open(manifest_path, "rb") as f:
                os.fsync(f.fileno())

            # fsync temp dir.
            _fsync_dir(temp_dir)

            # Atomic rename.
            try:
                os.rename(str(temp_dir), str(final_dir))
            except OSError:
                if not final_dir.exists():
                    shutil.move(str(temp_dir), str(final_dir))

            _fsync_dir(parent)
            logger.info("Bundle committed to %s", final_dir)

            return CommitResult(
                canonical_uri=str(final_dir),
                manifest_uri=str(final_dir / "manifest.json"),
                manifest_sha256=bundle.manifest_sha256,
                commit_status="committed",
            )

        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def get(self, ref: BundleRef) -> dict[str, Any]:
        """Read the manifest for *ref*."""
        manifest_path = Path(ref.canonical_uri) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_bytes())

    def update_alias(
        self,
        alias: str,
        new_ref: BundleRef,
        expected_revision: str | None = None,
    ) -> AliasUpdateResult:
        """Write an alias file with atomic rename."""
        alias_dir = Path(new_ref.canonical_uri).parent / "aliases"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_path = alias_dir / f"{alias}.json"

        alias_data = {
            "manifest_uri": new_ref.manifest_sha256,
            "canonical_uri": new_ref.canonical_uri,
            "bundle_id": new_ref.bundle_id,
        }

        # CAS: if expected_revision is set, check current.
        if expected_revision is not None:
            if alias_path.exists():
                current = json.loads(alias_path.read_bytes())
                current_sha = current.get("manifest_uri", "")
                if current_sha != expected_revision:
                    return AliasUpdateResult(
                        alias=alias,
                        status="failed",
                        failure=FailureInfo(
                            code="CAS_MISMATCH",
                            category="publish",
                            message="Expected manifest_sha256 does not match current alias",
                        ),
                    )
            else:
                return AliasUpdateResult(
                    alias=alias,
                    status="failed",
                    failure=FailureInfo(
                        code="ALIAS_NOT_FOUND",
                        category="publish",
                        message="Alias does not exist for CAS update",
                    ),
                )

        tmp_path = alias_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(alias_data, indent=2))
        os.replace(str(tmp_path), str(alias_path))
        return AliasUpdateResult(alias=alias, status="updated")
