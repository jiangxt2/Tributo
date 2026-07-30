"""Tests for Publisher — local atomic commit, S3 publish, alias CAS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tributo.training.exporters.manifest import ManifestSourceInfo
from tributo.training.exporters.models import (
    AliasConfig,
    ArtifactFile,
    ArtifactRef,
    ExportExecutionResult,
    LogicalArtifact,
    NodeResult,
    ProducerInfo,
)
from tributo.training.exporters.publisher import Publisher

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_producer() -> ProducerInfo:
    return ProducerInfo(exporter_id="test-v1")


def _make_artifact_files() -> tuple[ArtifactFile, ...]:
    return (
        ArtifactFile(
            relative_path="model.onnx", sha256="a" * 64, size_bytes=100, role="model"
        ),
        ArtifactFile(
            relative_path="config.json", sha256="b" * 64, size_bytes=50, role="config"
        ),
    )


def _make_logical_artifact(
    name: str = "fp32",
    fmt: str = "onnx",
    files: tuple[ArtifactFile, ...] | None = None,
) -> LogicalArtifact:
    f = files or _make_artifact_files()
    tree_digest = LogicalArtifact.compute_tree_digest(f)
    return LogicalArtifact(
        name=name,
        format=fmt,
        flavor_id="onnx-runtime-v1",
        files=f,
        entrypoint="model.onnx",
        tree_digest=tree_digest,
        producer=_make_producer(),
    )


def _make_node_result(
    node_id: str = "fp32",
    status: str = "succeeded",
    required: bool = True,
    publish: bool = True,
    artifact_ref: ArtifactRef | None = None,
    exporter_id: str = "test-v1",
) -> NodeResult:
    return NodeResult(
        node_id=node_id,
        target_name=node_id,
        status=status,
        required=required,
        publish=publish,
        exporter_id=exporter_id,
        artifact_ref=artifact_ref,
    )


def _make_execution(
    nodes: tuple[NodeResult, ...] | None = None,
    status: str = "succeeded",
    roles: dict[str, str] | None = None,
    staged_artifacts: dict[str, LogicalArtifact] | None = None,
) -> ExportExecutionResult:
    if nodes is None:
        ref = ArtifactRef(node_id="fp32", artifact_name="fp32", tree_digest="a" * 64)
        nodes = (_make_node_result(artifact_ref=ref),)
    return ExportExecutionResult(
        execution_id="exec-1",
        status=status,
        node_results=nodes,
        roles=roles or {},
        staged_artifacts=staged_artifacts or {},
    )


def _setup_staging(
    staging_root: Path,
    artifacts: list[LogicalArtifact],
) -> dict[str, LogicalArtifact]:
    """Create staging directory with fake artifact files.

    Returns a dict suitable for ``ExportExecutionResult.staged_artifacts``.
    """
    for artifact in artifacts:
        artifact_dir = staging_root / "nodes" / artifact.name / "artifact"
        artifact_dir.mkdir(parents=True)
        for af in artifact.files:
            fp = artifact_dir / af.relative_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"fake-content" + af.relative_path.encode())
    return {a.name: a for a in artifacts}


def _make_source_info() -> ManifestSourceInfo:
    return ManifestSourceInfo(
        source_kind="pytorch_result",
        framework="pytorch",
        framework_version="2.5.0",
        task_type="classification",
    )


# ── Local publish tests ───────────────────────────────────────────────────────


class TestLocalPublish:
    def test_basic_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert published.result.status == "succeeded"
        assert published.result.bundle_id == "test-bundle-1"
        assert published.result.manifest_sha256
        assert len(published.result.manifest_sha256) == 64
        assert published.local_bundle_dir.exists()
        assert (published.local_bundle_dir / "manifest.json").is_file()
        assert (
            published.local_bundle_dir / "artifacts" / "fp32" / "model.onnx"
        ).is_file()

    def test_manifest_json_is_valid(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        manifest_path = published.local_bundle_dir / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        assert manifest_data["schema_version"] == 1
        assert manifest_data["bundle_id"] == "test-bundle-1"
        assert len(manifest_data["artifacts"]) == 1

    def test_idempotent_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        result1 = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        # Second publish with same content + same execution_id should be idempotent.
        result2 = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",  # Same execution_id → identical manifest.
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert result1.result.manifest_sha256 == result2.result.manifest_sha256

    def test_partial_bundle_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        _setup_staging(staging, [artifact])

        ref_ok = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(
                _make_node_result(node_id="fp32", artifact_ref=ref_ok),
                _make_node_result(
                    node_id="opt", status="failed", required=False, artifact_ref=None
                ),
            ),
            status="partial",
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert published.result.status == "partial"
        assert published.local_bundle_dir.exists()

    def test_failed_execution_rejected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"

        execution = _make_execution(
            nodes=(_make_node_result(status="failed", artifact_ref=None),),
            status="failed",
        )

        publisher = Publisher()
        with pytest.raises(ValueError, match="Cannot publish a failed execution"):
            publisher.publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=str(dest),
                bundle_id="test-bundle-1",
                execution_id="exec-1",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

    def test_roles_in_result(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            roles={"inference": "fp32"},
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert published.result.roles == {"inference": "fp32"}

    def test_alias_not_requested_for_local(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        # alias with local URI should not attempt update.
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
            alias_config=AliasConfig(name="latest", policy="newer"),
        )

        assert published.result.alias_status == "not_requested"


# ── S3 publish tests (botocore Stubber) ──────────────────────────────────────


class TestS3PublishLogic:
    """Tests for S3-specific logic using unit-level verification.

    Integration tests with a real S3-compatible endpoint (MinIO) are in
    the CI workflow.
    """

    def test_s3_bundle_uri_produces_s3_manifest_uri(self, tmp_path: Path) -> None:
        """Skip: actual S3 calls require botocore stubber or MinIO."""
        pytest.skip("S3 integration test — requires MinIO or botocore Stubber setup")

    def test_alias_newer_policy_comparison(self) -> None:
        """Alias with policy=newer should compare created_at timestamps."""
        pytest.skip("Alias unit test — requires S3 client mock")

    def test_alias_cas_policy_rejects_mismatch(self) -> None:
        """Alias with compare_and_swap should reject mismatched digests."""
        pytest.skip("Alias unit test — requires S3 client mock")
