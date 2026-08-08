"""Tests for bundle export data models — pure memory, no real exporters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from tributo.exporting.events import OperationEvent
from tributo.exporting.models import (
    AliasConfig,
    ArtifactDraft,
    ArtifactFile,
    ArtifactRef,
    BundleOutputConfig,
    BundleResult,
    DraftFile,
    ExportExecutionResult,
    ExportTarget,
    FailureInfo,
    HookBinding,
    LogicalArtifact,
    NodeResult,
    PlannedTarget,
    ProducerInfo,
    SupportResult,
    ValidationResult,
)

# ── ExportTarget ─────────────────────────────────────────────────────────────


class TestExportTarget:
    @pytest.mark.parametrize("format_id", ("onnx", "ubj", "xgboost-json", "pt2"))
    def test_formats_use_one_configuration_field(self, format_id: str) -> None:
        t = ExportTarget(name="native", format=format_id)
        assert t.name == "native"
        assert t.format == format_id
        assert t.required is True
        assert t.depends_on == ()

    def test_legacy_xgboost_default_normalises_to_ubj(self) -> None:
        with pytest.warns(DeprecationWarning, match="format='xgboost'"):
            target = ExportTarget(name="native", format="xgboost")

        assert target.format == "ubj"
        assert target.exporter_id == "xgboost-ubj-v1"
        assert target.options == {}

    def test_legacy_xgboost_json_normalises_before_planning(self) -> None:
        with pytest.warns(DeprecationWarning, match="xgboost-json"):
            target = ExportTarget(
                name="native",
                format="xgboost",
                exporter_id="xgboost-native-v1",
                options={"fmt": "json"},
            )

        assert target.format == "xgboost-json"
        assert target.exporter_id == "xgboost-json-v1"
        assert target.options == {}

    @pytest.mark.parametrize("legacy_format", ("binary", 1, ["ubj"]))
    def test_legacy_xgboost_rejects_unknown_secondary_format(
        self,
        legacy_format: Any,
    ) -> None:
        with pytest.raises(ValidationError, match="must be 'ubj' or 'json'"):
            ExportTarget(
                name="native",
                format="xgboost",
                options={"fmt": legacy_format},
            )

    @pytest.mark.parametrize("format_id", ("ONNX", "xgboost_json", "-onnx"))
    def test_rejects_noncanonical_format_id(self, format_id: str) -> None:
        with pytest.raises(ValidationError, match="lowercase kebab-case"):
            ExportTarget(name="native", format=format_id)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ExportTarget(name="", format="ubj")

    def test_rejects_invalid_chars_in_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ExportTarget(name="bad/name", format="ubj")

    def test_with_options(self) -> None:
        t = ExportTarget(name="fp32", format="onnx", options={"opset": 18})
        assert t.options == {"opset": 18}

    def test_depends_on(self) -> None:
        t = ExportTarget(name="int8", format="onnx", depends_on=("fp32",))
        assert t.depends_on == ("fp32",)


def test_operation_event_rejects_unstable_time_and_correlation_fields() -> None:
    event = OperationEvent.bundle_published(
        occurred_at=datetime(2025, 1, 1, 8, tzinfo=timezone.utc),
        bundle_id="bundle-1",
        canonical_uri="file:///bundle-1",
        manifest_sha256="a" * 64,
        correlation_ids={"run_id": "run-1"},
    )
    assert event.occurred_at.tzinfo is timezone.utc

    with pytest.raises(ValidationError, match="timezone-aware"):
        OperationEvent.bundle_published(
            occurred_at=datetime(2025, 1, 1),
            bundle_id="bundle-1",
            canonical_uri="file:///bundle-1",
            manifest_sha256="a" * 64,
        )
    with pytest.raises(ValidationError, match="unsupported correlation ID"):
        OperationEvent.bundle_published(
            occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            bundle_id="bundle-1",
            canonical_uri="file:///bundle-1",
            manifest_sha256="a" * 64,
            correlation_ids={"attempt_id": "attempt-1"},
        )


# ── BundleOutputConfig ───────────────────────────────────────────────────────


class TestBundleOutputConfig:
    def test_legacy_mode_none_targets(self) -> None:
        cfg = BundleOutputConfig()
        assert cfg.targets is None
        assert cfg.bundle_uri is None

    def test_bundle_mode_requires_uri(self) -> None:
        with pytest.raises(ValidationError, match="bundle_uri is required"):
            BundleOutputConfig(targets=[ExportTarget(name="a", format="onnx")])

    def test_hook_bindings_are_explicit_and_unique(self) -> None:
        cfg = BundleOutputConfig(
            hooks=(HookBinding(hook_id="mlflow-v1", required=True),)
        )
        assert cfg.hooks[0].required is True
        with pytest.raises(ValidationError, match="hook_id values must be unique"):
            BundleOutputConfig(
                hooks=(
                    HookBinding(hook_id="mlflow-v1"),
                    HookBinding(hook_id="mlflow-v1"),
                )
            )

    def test_unique_target_names(self) -> None:
        with pytest.raises(ValidationError, match="target names must be unique"):
            BundleOutputConfig(
                bundle_uri="s3://bucket/model",
                targets=[
                    ExportTarget(name="a", format="onnx"),
                    ExportTarget(name="a", format="ubj"),
                ],
            )

    def test_depends_on_unknown_accepted_at_config_level(self) -> None:
        """Unknown depends_on names are now accepted at config validation.

        They may refer to upstream_requirements that the planner resolves
        into implicit nodes.  The planner (not the config model) raises
        an error if a name cannot be resolved.
        """
        cfg = BundleOutputConfig(
            bundle_uri="s3://bucket/model",
            targets=[
                ExportTarget(name="a", format="onnx", depends_on=("b",)),
            ],
        )
        assert cfg.targets is not None
        assert cfg.targets[0].depends_on == ("b",)

    def test_self_dependency(self) -> None:
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            BundleOutputConfig(
                bundle_uri="s3://bucket/model",
                targets=[
                    ExportTarget(name="a", format="onnx", depends_on=("a",)),
                ],
            )

    def test_roles_implicit_allowed(self) -> None:
        """Roles may reference names resolved by upstream_requirements."""
        cfg = BundleOutputConfig(
            bundle_uri="s3://bucket/model",
            targets=[ExportTarget(name="a", format="onnx")],
            roles={"inference": "b"},
        )
        assert cfg.roles == {"inference": "b"}

    def test_alias_newer_rejects_expected_sha256(self) -> None:
        with pytest.raises(ValidationError, match="policy='newer'"):
            BundleOutputConfig(
                bundle_uri="s3://bucket/model",
                targets=[ExportTarget(name="a", format="onnx")],
                alias=AliasConfig(
                    name="latest",
                    policy="newer",
                    expected_manifest_sha256="a" * 64,
                ),
            )

    def test_s3_uri_accepted(self) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="s3://my-bucket/models/v1",
            targets=[ExportTarget(name="a", format="onnx")],
        )
        assert cfg.bundle_uri == "s3://my-bucket/models/v1"

    def test_file_uri_accepted(self) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="file:///tmp/models",
            targets=[ExportTarget(name="a", format="onnx")],
        )
        assert cfg.bundle_uri == "file:///tmp/models"

    def test_bare_path_accepted(self) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/models",
            targets=[ExportTarget(name="a", format="onnx")],
        )
        assert cfg.bundle_uri == "/tmp/models"

    def test_rejects_root_file_uri(self) -> None:
        with pytest.raises(ValidationError, match="filesystem root"):
            BundleOutputConfig(
                bundle_uri="file:///",
                targets=[ExportTarget(name="a", format="onnx")],
            )


# ── ArtifactFile ─────────────────────────────────────────────────────────────


class TestArtifactFile:
    def test_minimal(self) -> None:
        f = ArtifactFile(
            relative_path="model.onnx",
            sha256="a" * 64,
            size_bytes=1024,
        )
        assert f.role == "model"

    def test_rejects_abs_path(self) -> None:
        with pytest.raises(ValidationError, match="relative"):
            ArtifactFile(relative_path="/etc/passwd", sha256="a" * 64, size_bytes=0)

    def test_rejects_parent_traversal(self) -> None:
        with pytest.raises(ValidationError, match="'.' or '..'"):
            ArtifactFile(relative_path="../model.onnx", sha256="a" * 64, size_bytes=0)

    def test_rejects_backslash(self) -> None:
        with pytest.raises(ValidationError, match="relative"):
            ArtifactFile(relative_path="sub\\model.onnx", sha256="a" * 64, size_bytes=0)

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            ArtifactFile(
                relative_path="model.onnx",
                sha256="a" * 64,
                size_bytes=0,
                role="executable",
            )

    def test_multi_file_dir(self) -> None:
        f = ArtifactFile(
            relative_path="subdir/model.onnx", sha256="b" * 64, size_bytes=2048
        )
        assert f.relative_path == "subdir/model.onnx"


# ── LogicalArtifact ──────────────────────────────────────────────────────────


class TestLogicalArtifactTreeDigest:
    @pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", "a\\b"])
    def test_rejects_unsafe_artifact_name(self, name: str) -> None:
        with pytest.raises(ValidationError, match="artifact name"):
            LogicalArtifact(
                name=name,
                format="onnx",
                flavor_id="onnx-runtime-v1",
                files=(
                    ArtifactFile(
                        relative_path="model.onnx",
                        sha256="a" * 64,
                        size_bytes=1,
                    ),
                ),
                entrypoint="model.onnx",
                tree_digest="b" * 64,
                producer=ProducerInfo(exporter_id="test-v1"),
            )

    def test_deterministic(self) -> None:
        files1 = (
            ArtifactFile(relative_path="a.onnx", sha256="a" * 64, size_bytes=1),
            ArtifactFile(relative_path="b.onnx", sha256="b" * 64, size_bytes=2),
        )
        files2 = (
            ArtifactFile(relative_path="b.onnx", sha256="b" * 64, size_bytes=2),
            ArtifactFile(relative_path="a.onnx", sha256="a" * 64, size_bytes=1),
        )
        assert LogicalArtifact.compute_tree_digest(
            files1
        ) == LogicalArtifact.compute_tree_digest(files2)

    def test_changes_on_content(self) -> None:
        f1 = (ArtifactFile(relative_path="a.onnx", sha256="a" * 64, size_bytes=1),)
        f2 = (ArtifactFile(relative_path="a.onnx", sha256="b" * 64, size_bytes=1),)
        assert LogicalArtifact.compute_tree_digest(
            f1
        ) != LogicalArtifact.compute_tree_digest(f2)

    def test_changes_on_size(self) -> None:
        f1 = (ArtifactFile(relative_path="a.onnx", sha256="a" * 64, size_bytes=1),)
        f2 = (ArtifactFile(relative_path="a.onnx", sha256="a" * 64, size_bytes=2),)
        assert LogicalArtifact.compute_tree_digest(
            f1
        ) != LogicalArtifact.compute_tree_digest(f2)

    def test_changes_on_role(self) -> None:
        f1 = (
            ArtifactFile(
                relative_path="a.onnx", sha256="a" * 64, size_bytes=1, role="model"
            ),
        )
        f2 = (
            ArtifactFile(
                relative_path="a.onnx", sha256="a" * 64, size_bytes=1, role="aux"
            ),
        )
        assert LogicalArtifact.compute_tree_digest(
            f1
        ) != LogicalArtifact.compute_tree_digest(f2)


# ── ArtifactDraft ────────────────────────────────────────────────────────────


class TestArtifactDraft:
    @pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b"])
    def test_rejects_unsafe_name_before_publish(self, name: str) -> None:
        with pytest.raises(ValidationError, match="artifact name"):
            ArtifactDraft(
                name=name,
                format="onnx",
                flavor_id="onnx-runtime-v1",
                files=(DraftFile(relative_path="model.onnx", role="model"),),
                entrypoint="model.onnx",
                producer=ProducerInfo(exporter_id="test-v1"),
            )

    def test_valid(self) -> None:
        draft = ArtifactDraft(
            name="fp32",
            format="onnx",
            flavor_id="onnx-runtime-v1",
            files=(DraftFile(relative_path="model.onnx", role="model"),),
            entrypoint="model.onnx",
            producer=ProducerInfo(exporter_id="torch-onnx-v1"),
        )
        assert draft.entrypoint == "model.onnx"

    def test_entrypoint_must_be_in_files(self) -> None:
        with pytest.raises(ValidationError, match="entrypoint"):
            ArtifactDraft(
                name="fp32",
                format="onnx",
                flavor_id="onnx-runtime-v1",
                files=(DraftFile(relative_path="model.onnx", role="model"),),
                entrypoint="other.onnx",
                producer=ProducerInfo(exporter_id="test-v1"),
            )


# ── BundleResult ─────────────────────────────────────────────────────────────


class TestBundleResult:
    def test_minimal(self) -> None:
        result = BundleResult(
            bundle_id="20260730-abc123",
            canonical_uri="s3://bucket/20260730-abc123/",
            manifest_uri="s3://bucket/20260730-abc123/manifest.json",
            manifest_sha256="c" * 64,
            status="succeeded",
        )
        assert result.alias_status == "not_requested"

    def test_partial_with_alias_failure(self) -> None:
        result = BundleResult(
            bundle_id="20260730-abc123",
            canonical_uri="s3://bucket/20260730-abc123/",
            manifest_uri="s3://bucket/20260730-abc123/manifest.json",
            manifest_sha256="c" * 64,
            status="partial",
            alias_status="failed",
            alias_failure=FailureInfo(
                code="CAS_CONFLICT",
                category="publish",
                message="Alias already updated by another process",
            ),
        )
        assert result.alias_status == "failed"


# ── ExportExecutionResult ────────────────────────────────────────────────────


class TestExportExecutionResult:
    def test_succeeded_artifacts(self) -> None:
        ref = ArtifactRef(
            node_id="node-fp32", artifact_name="fp32", tree_digest="d" * 64
        )
        nr = NodeResult(
            node_id="node-fp32",
            target_name="fp32",
            status="succeeded",
            required=True,
            publish=True,
            exporter_id="torch-onnx-v1",
            artifact_ref=ref,
        )
        result = ExportExecutionResult(
            execution_id="exec-1",
            status="succeeded",
            node_results=(nr,),
        )
        assert result.succeeded_artifacts == {"node-fp32": ref}


# ── PlannedTarget ────────────────────────────────────────────────────────────


class TestPlannedTarget:
    def test_explicit(self) -> None:
        target = ExportTarget(name="fp32", format="onnx")
        pt = PlannedTarget(
            target=target,
            exporter_id="torch-onnx-v1",
            typed_options={"opset": 18},
            implicit=False,
            publish=True,
        )
        assert pt.exporter_id == "torch-onnx-v1"
        assert not pt.implicit


# ── Support request/result ───────────────────────────────────────────────────


class TestSupportModels:
    def test_supported(self) -> None:
        r = SupportResult(supported=True, code="OK")
        assert r.supported

    def test_unsupported_with_details(self) -> None:
        r = SupportResult(
            supported=False,
            code="UNSUPPORTED_TASK",
            reason="Only classification is supported",
            missing_dependencies=(),
        )
        assert not r.supported
        assert r.code == "UNSUPPORTED_TASK"


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidationResult:
    def test_passed(self) -> None:
        vr = ValidationResult(validator_id="structure-v1", status="passed")
        assert vr.status == "passed"

    def test_failed_with_failure(self) -> None:
        vr = ValidationResult(
            validator_id="parity-v1",
            status="failed",
            failure=FailureInfo(
                code="PARITY_THRESHOLD",
                category="validation",
                message="Cosine similarity 0.8 below threshold 0.999",
            ),
        )
        assert vr.status == "failed"
        assert vr.failure is not None
        assert vr.failure.code == "PARITY_THRESHOLD"


# ── Export context ───────────────────────────────────────────────────────────


class TestExportContext:
    def test_create(self, tmp_path: Any) -> None:
        from pathlib import Path

        ctx = __import__(
            "tributo.exporting.models", fromlist=["ExportContext"]
        ).ExportContext(
            execution_id="exec-1",
            node_id="node-1",
            artifact_dir=Path(tmp_path) / "artifacts" / "node-1",
        )
        assert ctx.execution_id == "exec-1"
