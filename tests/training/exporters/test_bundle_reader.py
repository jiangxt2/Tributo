"""Tests for BundleReader — manifest reading, artifact opening, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tributo.exporting.bundle_reader import BundleReader, ReaderResourceLimits
from tributo.exporting.manifest import ManifestSourceInfo
from tributo.exporting.models import (
    ArtifactFile,
    LogicalArtifact,
    ProducerInfo,
)
from tributo.exporting.publisher import Publisher

# ── Helpers ───────────────────────────────────────────────────────────────────


def _create_test_bundle(tmp_path: Path) -> tuple[Path, str, LogicalArtifact]:
    """Publish a real local bundle and return (bundle_dir, manifest_sha256, artifact)."""
    staging = tmp_path / "staging"
    dest = tmp_path / "bundles"

    # Create a real artifact file with known content.
    content = b"onnx-model-content-123"
    artifact_dir = staging / "nodes" / "fp32" / "artifact"
    artifact_dir.mkdir(parents=True)
    artifact_dir.joinpath("model.onnx").write_bytes(content)

    import hashlib

    sha = hashlib.sha256(content).hexdigest()
    files = (
        ArtifactFile(
            relative_path="model.onnx",
            sha256=sha,
            size_bytes=len(content),
            role="model",
        ),
    )
    tree_digest = LogicalArtifact.compute_tree_digest(files)
    artifact = LogicalArtifact(
        name="fp32",
        format="onnx",
        flavor_id="onnx-runtime-v1",
        files=files,
        entrypoint="model.onnx",
        tree_digest=tree_digest,
        producer=ProducerInfo(exporter_id="test-v1"),
    )

    from tributo.exporting.models import (
        ArtifactRef,
        ExportExecutionResult,
        NodeResult,
    )

    ref = ArtifactRef(
        node_id="fp32",
        artifact_name="fp32",
        tree_digest=artifact.tree_digest,
    )
    execution = ExportExecutionResult(
        execution_id="exec-1",
        status="succeeded",
        node_results=(
            NodeResult(
                node_id="fp32",
                target_name="fp32",
                status="succeeded",
                required=True,
                publish=True,
                exporter_id="test-v1",
                output_format="onnx",
                flavor_id="onnx-runtime-v1",
                artifact_ref=ref,
            ),
        ),
        staged_artifacts={"fp32": artifact},
        roles={"inference": "fp32"},
    )

    publisher = Publisher()
    published = publisher.publish(
        execution=execution,
        staging_root=staging,
        bundle_uri=str(dest),
        bundle_id="reader-test-1",
        execution_id="exec-1",
        tributo_version="0.1.0",
        source_info=ManifestSourceInfo(source_kind="pytorch_result"),
    )

    return published.local_bundle_dir, published.result.manifest_sha256, artifact


# ── Tests ────────────────────────────────────────────────────────────────────


class TestReadManifest:
    def test_read_manifest_from_root(self, tmp_path: Path) -> None:
        bundle_dir, manifest_sha256, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        manifest = reader.read_manifest(str(bundle_dir))
        assert manifest.bundle_id == "reader-test-1"
        assert manifest.status == "succeeded"
        assert len(manifest.artifacts) == 1

    def test_read_manifest_from_exact_path(self, tmp_path: Path) -> None:
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        manifest = reader.read_manifest(str(bundle_dir / "manifest.json"))
        assert manifest.bundle_id == "reader-test-1"

    def test_read_manifest_nonexistent_file(self, tmp_path: Path) -> None:
        reader = BundleReader()
        with pytest.raises(FileNotFoundError):
            reader.read_manifest(str(tmp_path / "does-not-exist"))


class TestOpenArtifact:
    def test_open_by_role(self, tmp_path: Path) -> None:
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        with reader.open_artifact(str(bundle_dir), role="inference") as ra:
            assert ra.descriptor.name == "fp32"
            assert ra.entrypoint_path.is_file()
            assert ra.entrypoint_path.read_bytes() == b"onnx-model-content-123"

    def test_open_by_artifact_name(self, tmp_path: Path) -> None:
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        with reader.open_artifact(str(bundle_dir), artifact_name="fp32") as ra:
            assert ra.descriptor.name == "fp32"

    def test_open_missing_role_raises(self, tmp_path: Path) -> None:
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        with pytest.raises(ValueError, match="not found in bundle"):
            with reader.open_artifact(str(bundle_dir), role="nonexistent"):
                pass

    def test_open_missing_artifact_raises(self, tmp_path: Path) -> None:
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        with pytest.raises(ValueError, match="not found in bundle"):
            with reader.open_artifact(str(bundle_dir), artifact_name="nonexistent"):
                pass

    def test_open_neither_role_nor_name_raises(self, tmp_path: Path) -> None:
        reader = BundleReader()
        with pytest.raises(ValueError, match="Exactly one"):
            with reader.open_artifact("/tmp"):
                pass

    def test_open_both_role_and_name_raises(self, tmp_path: Path) -> None:
        reader = BundleReader()
        with pytest.raises(ValueError, match="Exactly one"):
            with reader.open_artifact("/tmp", role="x", artifact_name="y"):
                pass

    def test_integrity_check_passes_with_valid_files(self, tmp_path: Path) -> None:
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()
        # Should pass without error.
        with reader.open_artifact(str(bundle_dir), role="inference") as ra:
            assert ra.descriptor.entrypoint == "model.onnx"


class TestResourceLimits:
    def test_default_limits(self) -> None:
        limits = ReaderResourceLimits()
        assert limits.max_manifest_bytes == 10 * 1024 * 1024
        assert limits.max_file_count == 256
        assert limits.max_total_bytes == 50 * 1024 * 1024 * 1024

    def test_custom_limits(self) -> None:
        limits = ReaderResourceLimits(
            max_manifest_bytes=1024,
            max_file_count=10,
            max_single_file_bytes=100,
            max_total_bytes=1000,
        )
        assert limits.max_manifest_bytes == 1024
        assert limits.max_file_count == 10

    def test_custom_limits_applied(self, tmp_path: Path) -> None:
        """Custom limits are passed to the reader and enforced."""
        bundle_dir, _, _ = _create_test_bundle(tmp_path)
        reader = BundleReader(
            limits=ReaderResourceLimits(
                max_file_count=1,  # Artifact has 1 file — OK.
                max_total_bytes=1,  # But file is 21 bytes → should fail.
            ),
        )
        with pytest.raises(ValueError, match="exceeds limit"):
            with reader.open_artifact(str(bundle_dir), role="inference"):
                pass


class TestRoundTrip:
    """Full round-trip: publish → read → verify."""

    def test_publish_then_read_role(self, tmp_path: Path) -> None:
        bundle_dir, manifest_sha256, _ = _create_test_bundle(tmp_path)
        reader = BundleReader()

        reader.read_manifest(str(bundle_dir))
        with reader.open_artifact(str(bundle_dir), role="inference") as ra:
            assert ra.descriptor.name == "fp32"
            assert ra.descriptor.format == "onnx"
            assert ra.entrypoint_path.read_bytes() == b"onnx-model-content-123"
