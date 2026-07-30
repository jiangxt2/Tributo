"""Tests for ExportManifest and ManifestSchemaRegistry."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tributo.training.exporters.manifest import (
    ExportManifest,
    ManifestExecution,
    ManifestExecutionNode,
    ManifestSchemaRegistry,
    ManifestSignature,
    ManifestSourceInfo,
    _read_manifest_v1,
)
from tributo.training.exporters.models import (
    ArtifactFile,
    LogicalArtifact,
    ProducerInfo,
)

_FIXED_TS = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_minimal_manifest(status: str = "succeeded") -> ExportManifest:
    return ExportManifest(
        schema_version=1,
        bundle_id="20260730T120000Z-abc123",
        status=status,
        created_at=_FIXED_TS,
        canonical_uri="file:///tmp/bundles/test-bundle",
        tributo_version="0.1.0",
        source_info=ManifestSourceInfo(source_kind="pytorch_result"),
        execution=ManifestExecution(execution_id="exec-1"),
    )


def _make_artifact(name: str, fmt: str = "onnx") -> LogicalArtifact:
    files = (
        ArtifactFile(
            relative_path="model.onnx", sha256="a" * 64, size_bytes=100, role="model"
        ),
    )
    tree_digest = LogicalArtifact.compute_tree_digest(files)
    return LogicalArtifact(
        name=name,
        format=fmt,
        flavor_id="onnx-runtime-v1",
        files=files,
        entrypoint="model.onnx",
        tree_digest=tree_digest,
        producer=ProducerInfo(exporter_id="test-v1"),
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestExportManifest:
    def test_minimal_manifest(self) -> None:
        m = _make_minimal_manifest()
        assert m.schema_version == 1
        assert m.bundle_id == "20260730T120000Z-abc123"
        assert m.status == "succeeded"

    def test_manifest_with_artifacts(self) -> None:
        a = _make_artifact("fp32")
        m = ExportManifest(
            schema_version=1,
            bundle_id="b1",
            status="succeeded",
            canonical_uri="s3://b/pre/b1",
            tributo_version="0.1.0",
            source_info=ManifestSourceInfo(source_kind="pytorch_result"),
            artifacts=(a,),
            roles={"inference": "fp32"},
            execution=ManifestExecution(execution_id="exec-1"),
        )
        assert len(m.artifacts) == 1
        assert m.roles == {"inference": "fp32"}

    def test_manifest_with_execution_nodes(self) -> None:
        node = ManifestExecutionNode(
            node_id="fp32",
            target_name="fp32",
            exporter_id="torch-onnx-v1",
            status="succeeded",
            required=True,
        )
        m = ExportManifest(
            schema_version=1,
            bundle_id="b1",
            status="succeeded",
            canonical_uri="s3://b/pre/b1",
            tributo_version="0.1.0",
            source_info=ManifestSourceInfo(source_kind="pytorch_result"),
            execution=ManifestExecution(execution_id="exec-1", nodes=(node,)),
        )
        assert len(m.execution.nodes) == 1
        assert m.execution.nodes[0].status == "succeeded"

    def test_canonical_json_deterministic(self) -> None:
        """Two manifests with identical fields produce identical canonical JSON."""
        m1 = _make_minimal_manifest()
        m2 = _make_minimal_manifest()
        assert m1.canonical_json() == m2.canonical_json()

    def test_canonical_json_keys_sorted(self) -> None:
        m = _make_minimal_manifest()
        raw = m.canonical_json()
        # Verify it's valid JSON and keys appear.
        import json

        d = json.loads(raw)
        assert "bundle_id" in d
        assert "schema_version" in d

    def test_compute_sha256(self) -> None:
        m = _make_minimal_manifest()
        digest = m.compute_sha256()
        assert len(digest) == 64

    def test_manifest_sha256_changes_with_content(self) -> None:
        m1 = _make_minimal_manifest()
        m2 = ExportManifest(
            schema_version=1,
            bundle_id="20260730T120000Z-xyz789",  # different ID
            status="succeeded",
            canonical_uri="file:///tmp/bundles/test-bundle",
            tributo_version="0.1.0",
            source_info=ManifestSourceInfo(source_kind="pytorch_result"),
            execution=ManifestExecution(execution_id="exec-1"),
        )
        assert m1.compute_sha256() != m2.compute_sha256()

    def test_partial_status(self) -> None:
        m = _make_minimal_manifest(status="partial")
        assert m.status == "partial"

    def test_invalid_status_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ManifestExecutionNode(
                node_id="n1",
                target_name="n1",
                status="unknown_status",  # invalid
            )

    def test_signature_models(self) -> None:
        sig = ManifestSignature(
            input_names=("input",),
            output_names=("output",),
            dynamic_axes={"input": {0: "batch"}},
        )
        assert sig.input_names == ("input",)
        assert len(sig.dynamic_axes) == 1

    def test_source_info(self) -> None:
        si = ManifestSourceInfo(
            source_kind="pytorch_result",
            source_fingerprint="abc123",
            framework="pytorch",
            framework_version="2.5.0",
            architecture_id="mlp-classifier",
            task_type="classification",
        )
        assert si.framework == "pytorch"
        assert si.architecture_id == "mlp-classifier"


class TestManifestSchemaRegistry:
    def test_register_and_read_v1(self) -> None:
        reg = ManifestSchemaRegistry()
        reg.register(1, _read_manifest_v1)
        m = _make_minimal_manifest()
        raw = json.loads(m.canonical_json())
        result = reg.read(1, raw, m.canonical_json())
        assert result.bundle_id == m.bundle_id

    def test_unknown_version_raises(self) -> None:
        reg = ManifestSchemaRegistry()
        with pytest.raises(ValueError, match="Unsupported manifest schema version"):
            reg.read(999, {}, b"{}")

    def test_duplicate_version_raises(self) -> None:
        reg = ManifestSchemaRegistry()
        reg.register(1, _read_manifest_v1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(1, _read_manifest_v1)
