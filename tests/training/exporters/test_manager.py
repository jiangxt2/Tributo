"""Tests for ExportManager — state machine, re-hash, validation chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel

from tributo.exporting.executor import ExportManager
from tributo.exporting.models import (
    ArtifactDraft,
    BundleOutputConfig,
    DraftFile,
    ExportContext,
    ExportSource,
    ExportTarget,
    FailureInfo,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidationResult,
    ValidatorBinding,
)
from tributo.exporting.planner import ExportPlanner
from tributo.exporting.registries import (
    ExportRegistry,
    ValidatorRegistry,
)

# ── Fake exporter ────────────────────────────────────────────────────────────


class _MinOpts(BaseModel):
    model_config = {"extra": "forbid"}


class _WritingExporter:
    """Writes a real model.onnx file to artifact_dir, returns a valid draft."""

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "fake-writer-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinOpts
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: Any,
    ) -> ArtifactDraft:
        # Write a real file.
        onnx_path = context.artifact_dir / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx-content")
        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="onnx-runtime-v1",
            files=(DraftFile(relative_path="model.onnx", role="model"),),
            entrypoint="model.onnx",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


class _FailingExporter:
    """Always raises."""

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "failing-v1"
    priority: ClassVar[int] = 10
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinOpts
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("export failed")


class _MultiFileExporter:
    """Writes multiple files — tests re-hash logic."""

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "multi-file-v1"
    priority: ClassVar[int] = 10
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinOpts
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: Any,
    ) -> ArtifactDraft:
        (context.artifact_dir / "model.onnx").write_bytes(b"onnx-data")
        (context.artifact_dir / "tokenizer.json").write_bytes(b'{"vocab": 100}')
        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="hf-onnx-v1",
            files=(
                DraftFile(relative_path="model.onnx", role="model"),
                DraftFile(relative_path="tokenizer.json", role="tokenizer"),
            ),
            entrypoint="model.onnx",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


# ── Fake validator ───────────────────────────────────────────────────────────


class _PassingValidator:
    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "pass-v1"
    options_model: ClassVar[type[BaseModel]] = _MinOpts

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        return ValidationResult(validator_id=self.validator_id, status="passed")


class _FailingValidator:
    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "fail-v1"
    options_model: ClassVar[type[BaseModel]] = _MinOpts

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        return ValidationResult(
            validator_id=self.validator_id,
            status="failed",
            failure=FailureInfo(
                code="FAIL", category="validation", message="validation failed"
            ),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_source() -> ExportSource:
    return ExportSource(source_kind="pytorch_result")


def _make_plan(cfg: BundleOutputConfig) -> tuple[Any, Any, Any]:
    er = ExportRegistry()
    _reg_defaults(er)
    vr = ValidatorRegistry()
    vr.register(_PassingValidator)
    vr.register(_FailingValidator)
    planner = ExportPlanner(er, vr)
    return planner.plan(cfg, _make_source()), er, vr


def _reg_defaults(er: ExportRegistry) -> None:
    """Register exporters with distinct priorities to avoid ambiguity."""
    er.register(_WritingExporter)  # priority 100
    er.register(_FailingExporter)  # priority 90
    er.register(_MultiFileExporter)  # priority 80


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSuccessfulExport:
    def test_single_success(self, tmp_path: Path) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="fp32", format="onnx")],
        )
        plan, er, vr = _make_plan(cfg)
        mgr = ExportManager(er, vr)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")

        assert result.status == "succeeded"
        assert len(result.node_results) == 1
        nr = result.node_results[0]
        assert nr.status == "succeeded"
        assert nr.artifact_ref is not None
        assert nr.artifact_ref.tree_digest  # non-empty hash

    def test_artifact_ref_is_valid(self, tmp_path: Path) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="fp32", format="onnx")],
        )
        plan, er, vr = _make_plan(cfg)
        mgr = ExportManager(er, vr)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")
        nr = result.node_results[0]
        assert nr.artifact_ref is not None
        assert len(nr.artifact_ref.tree_digest) == 64

    def test_multi_file_artifact(self, tmp_path: Path) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="hf", format="onnx", exporter_id="multi-file-v1")
            ],
        )
        plan, er, vr = _make_plan(cfg)
        mgr = ExportManager(er, vr)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")
        assert result.status == "succeeded"


class TestFailurePropagation:
    def test_required_failure_cancels_downstream(self, tmp_path: Path) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(
                    name="fail",
                    format="onnx",
                    required=True,
                    exporter_id="failing-v1",
                ),
                ExportTarget(
                    name="after",
                    format="onnx",
                    depends_on=("fail",),
                ),
            ],
        )
        plan, er, vr = _make_plan(cfg)
        mgr = ExportManager(er, vr)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")

        assert result.status == "failed"
        fail_nr = [n for n in result.node_results if n.node_id == "fail"][0]
        after_nr = [n for n in result.node_results if n.node_id == "after"][0]
        assert fail_nr.status == "failed"
        assert after_nr.status == "blocked"

    def test_optional_failure_allows_others(self, tmp_path: Path) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(
                    name="opt_fail",
                    format="onnx",
                    required=False,
                    exporter_id="failing-v1",
                ),
                ExportTarget(name="ok", format="onnx"),
            ],
        )
        plan, er, vr = _make_plan(cfg)
        mgr = ExportManager(er, vr)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")

        assert result.status == "partial"
        fail_nr = [n for n in result.node_results if n.node_id == "opt_fail"][0]
        ok_nr = [n for n in result.node_results if n.node_id == "ok"][0]
        assert fail_nr.status == "failed"
        assert ok_nr.status == "succeeded"

    def test_blocked_by_failed_dependency(self, tmp_path: Path) -> None:
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="base", format="onnx", exporter_id="failing-v1"),
                ExportTarget(name="dependent", format="onnx", depends_on=("base",)),
            ],
        )
        plan, er, vr = _make_plan(cfg)
        mgr = ExportManager(er, vr)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")

        dep_nr = [n for n in result.node_results if n.node_id == "dependent"][0]
        assert dep_nr.status == "blocked"

    def test_session_fatal_optional_cancels_downstream(self, tmp_path: Path) -> None:
        """Integrity violation on an optional node is session-fatal.

        Staging/path integrity violations cancel all remaining nodes and
        force the overall result to ``failed`` even when the failing node
        is optional (plan: session-fatal semantics).
        """

        class _IntegrityViolator:
            api_version: ClassVar[int] = 1
            exporter_id: ClassVar[str] = "integrity-v1"
            priority: ClassVar[int] = 10
            output_format: ClassVar[str] = "onnx"
            options_model: ClassVar[type[BaseModel]] = _MinOpts
            validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
            mutates_source: ClassVar[bool] = False

            @classmethod
            def supports(cls, r: SupportRequest) -> SupportResult:
                return SupportResult(supported=True, code="OK")

            def export(
                self,
                context: ExportContext,
                source: ExportSource,
                upstream: Mapping[str, ResolvedArtifact],
                target: Any,
            ) -> ArtifactDraft:
                (context.artifact_dir / "model.onnx").write_bytes(b"data")
                (context.artifact_dir / "undeclared.log").write_bytes(b"leak")
                return ArtifactDraft(
                    name=target.target.name,
                    format="onnx",
                    flavor_id="onnx-runtime-v1",
                    files=(DraftFile(relative_path="model.onnx", role="model"),),
                    entrypoint="model.onnx",
                    producer=ProducerInfo(exporter_id=self.exporter_id),
                )

        er2 = ExportRegistry()
        er2.register(_IntegrityViolator)
        vr2 = ValidatorRegistry()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(
                    name="bad",
                    format="onnx",
                    required=False,
                    exporter_id="integrity-v1",
                ),
                ExportTarget(name="dependent", format="onnx", depends_on=("bad",)),
                ExportTarget(name="independent", format="onnx"),
            ],
        )
        planner = ExportPlanner(er2, vr2)
        plan = planner.plan(cfg, _make_source())
        mgr = ExportManager(er2, vr2)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")

        # Session-fatal overrides the "optional failure → partial" rule.
        assert result.status == "failed"
        bad_nr = [n for n in result.node_results if n.node_id == "bad"][0]
        dep_nr = [n for n in result.node_results if n.node_id == "dependent"][0]
        ind_nr = [n for n in result.node_results if n.node_id == "independent"][0]
        assert bad_nr.status == "failed"
        assert dep_nr.status == "blocked"  # blocked by its failed dependency
        assert ind_nr.status == "cancelled"  # remaining DAG cancelled


class TestRehashVerification:
    def test_undeclared_file_causes_failure(self, tmp_path: Path) -> None:
        """Exporter writes extra file not in draft — should fail."""

        class _UndeclaredWriter:
            api_version: ClassVar[int] = 1
            exporter_id: ClassVar[str] = "undeclared-v1"
            priority: ClassVar[int] = 10
            output_format: ClassVar[str] = "onnx"
            options_model: ClassVar[type[BaseModel]] = _MinOpts
            validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
            mutates_source: ClassVar[bool] = False

            @classmethod
            def supports(cls, r: SupportRequest) -> SupportResult:
                return SupportResult(supported=True, code="OK")

            def export(
                self,
                context: ExportContext,
                source: ExportSource,
                upstream: Mapping[str, ResolvedArtifact],
                target: Any,
            ) -> ArtifactDraft:
                (context.artifact_dir / "model.onnx").write_bytes(b"data")
                (context.artifact_dir / "extra.log").write_bytes(b"leak")
                return ArtifactDraft(
                    name=target.target.name,
                    format="onnx",
                    flavor_id="onnx-runtime-v1",
                    files=(DraftFile(relative_path="model.onnx", role="model"),),
                    entrypoint="model.onnx",
                    producer=ProducerInfo(exporter_id=self.exporter_id),
                )

        er2 = ExportRegistry()
        er2.register(_UndeclaredWriter)
        vr2 = ValidatorRegistry()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="a", format="onnx", exporter_id="undeclared-v1")
            ],
        )
        planner = ExportPlanner(er2, vr2)
        plan = planner.plan(cfg, _make_source())
        mgr = ExportManager(er2, vr2)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")
        assert result.status == "failed"

    def test_missing_file_in_draft_causes_failure(self, tmp_path: Path) -> None:
        """Exporter declares file but doesn't write it."""

        class _MissingWriter:
            api_version: ClassVar[int] = 1
            exporter_id: ClassVar[str] = "missing-v1"
            priority: ClassVar[int] = 10
            output_format: ClassVar[str] = "onnx"
            options_model: ClassVar[type[BaseModel]] = _MinOpts
            validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
            mutates_source: ClassVar[bool] = False

            @classmethod
            def supports(cls, r: SupportRequest) -> SupportResult:
                return SupportResult(supported=True, code="OK")

            def export(
                self,
                context: ExportContext,
                source: ExportSource,
                upstream: Mapping[str, ResolvedArtifact],
                target: Any,
            ) -> ArtifactDraft:
                return ArtifactDraft(
                    name=target.target.name,
                    format="onnx",
                    flavor_id="onnx-runtime-v1",
                    files=(DraftFile(relative_path="ghost.onnx", role="model"),),
                    entrypoint="ghost.onnx",
                    producer=ProducerInfo(exporter_id=self.exporter_id),
                )

        er2 = ExportRegistry()
        er2.register(_MissingWriter)
        vr2 = ValidatorRegistry()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="a", format="onnx", exporter_id="missing-v1")],
        )
        planner = ExportPlanner(er2, vr2)
        plan = planner.plan(cfg, _make_source())
        mgr = ExportManager(er2, vr2)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")
        assert result.status == "failed"


class TestValidatorChain:
    def test_required_validator_failure(self, tmp_path: Path) -> None:
        """Exporter with required failing validator should fail."""

        class _ValidatedExporter:
            api_version: ClassVar[int] = 1
            exporter_id: ClassVar[str] = "validated-v1"
            priority: ClassVar[int] = 10
            output_format: ClassVar[str] = "onnx"
            options_model: ClassVar[type[BaseModel]] = _MinOpts
            validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
                ValidatorBinding(validator_id="fail-v1", required=True),
            )
            mutates_source: ClassVar[bool] = False

            @classmethod
            def supports(cls, r: SupportRequest) -> SupportResult:
                return SupportResult(supported=True, code="OK")

            def export(
                self,
                context: ExportContext,
                source: ExportSource,
                upstream: Mapping[str, ResolvedArtifact],
                target: Any,
            ) -> ArtifactDraft:
                (context.artifact_dir / "model.onnx").write_bytes(b"x")
                return ArtifactDraft(
                    name=target.target.name,
                    format="onnx",
                    flavor_id="onnx-runtime-v1",
                    files=(DraftFile(relative_path="model.onnx", role="model"),),
                    entrypoint="model.onnx",
                    producer=ProducerInfo(exporter_id=self.exporter_id),
                )

        er2 = ExportRegistry()
        er2.register(_ValidatedExporter)
        vr2 = ValidatorRegistry()
        vr2.register(_FailingValidator)
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="a", format="onnx")],
        )
        planner = ExportPlanner(er2, vr2)
        plan = planner.plan(cfg, _make_source())
        mgr = ExportManager(er2, vr2)
        result = mgr.execute(plan, _make_source(), tmp_path / "staging", "exec-1")
        assert result.status == "failed"
