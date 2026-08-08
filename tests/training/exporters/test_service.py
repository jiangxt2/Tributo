"""Tests for BundleExportService — full lifecycle orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from tributo.exceptions import (
    BundleExportError,
    JobConfigurationError,
    UnsupportedArtifactFormat,
)
from tributo.exporting.models import (
    ArtifactDraft,
    BundleOutputConfig,
    CheckpointField,
    DraftFile,
    ExportCheckpointV1,
    ExportContext,
    ExportSource,
    ExportTarget,
    ProducerInfo,
    PublishedBundle,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.exporting.registries import (
    ExportRegistry,
    FlavorRegistry,
    ValidatorRegistry,
)
from tributo.exporting.runtime import BundleModel, BundleModelLoader
from tributo.exporting.service import BundleExportService, bundle_id_for_request
from tributo.exporting.validators import StructureValidator

# ── Fake components ───────────────────────────────────────────────────────────


class _MinOpts(BaseModel):
    """No-op options model."""

    model_config = {"extra": "forbid"}


class _TestExporter:
    """Exporter that writes a simple model file."""

    api_version: int = 2
    exporter_id: str = "test-exporter-v1"
    priority: int = 100
    output_format: str = "onnx"
    output_flavor_id: str = "onnx-runtime-v1"
    options_model: type[BaseModel] = _MinOpts  # type: ignore[assignment]
    validator_bindings: tuple[ValidatorBinding, ...] = ()
    mutates_source: bool = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: dict,
        target: object,
    ) -> ArtifactDraft:
        (context.artifact_dir / "model.onnx").write_bytes(b"service-test-model")
        return ArtifactDraft(
            name=target.target.name if hasattr(target, "target") else "test",
            format="onnx",
            flavor_id="onnx-runtime-v1",
            files=(DraftFile(relative_path="model.onnx", role="model"),),
            entrypoint="model.onnx",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


class _FailingExporter:
    """Required exporter whose export() always fails."""

    api_version: int = 2
    exporter_id: str = "failing-exporter-v1"
    priority: int = 100
    output_format: str = "onnx"
    output_flavor_id: str = "onnx-runtime-v1"
    options_model: type[BaseModel] = _MinOpts
    validator_bindings: tuple[ValidatorBinding, ...] = ()
    mutates_source: bool = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: dict,
        target: object,
    ) -> ArtifactDraft:
        raise RuntimeError("simulated required export failure")


class _SignatureGateFlavor:
    """Test loader that must not run before the signature gate."""

    api_version = 1
    flavor_id = "onnx-runtime-v1"
    security_mode = "safe"
    signature_required = True
    required_dependencies: tuple[str, ...] = ()

    def load(
        self,
        artifact: object,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        raise AssertionError("signature gate must reject before loading")


def _make_source() -> ExportSource:
    return ExportSource(
        source_kind="pytorch_result",
        metadata={
            "task_type": "classification",
            "framework": "pytorch",
            "framework_version": "2.5.0",
        },
    )


def _make_config(tmp_path: Path) -> BundleOutputConfig:
    return BundleOutputConfig(
        bundle_uri=str(tmp_path / "bundles"),
        targets=[ExportTarget(name="fp32", format="onnx")],
        roles={"inference": "fp32"},
    )


def _make_registries() -> tuple[ExportRegistry, ValidatorRegistry]:
    er = ExportRegistry()
    er.register(_TestExporter)
    vr = ValidatorRegistry()
    return er, vr


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBundleExportService:
    def test_preserves_pre_registered_builtin_validator(self) -> None:
        exporters = ExportRegistry()
        validators = ValidatorRegistry()
        validators.register(StructureValidator)

        BundleExportService(
            export_registry=exporters,
            validator_registry=validators,
        )

        assert validators.get(StructureValidator.validator_id) is StructureValidator

    def test_run_id_is_stable_bundle_identity(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path).model_copy(update={"run_id": "run-42"})
        assert bundle_id_for_request("run-42") == bundle_id_for_request("run-42")
        result = BundleExportService(
            export_registry=_make_registries()[0],
            validator_registry=_make_registries()[1],
        ).export_bundle(source=_make_source(), config=config)
        assert result.bundle_id == bundle_id_for_request("run-42")

    def test_request_id_and_run_id_must_match(self) -> None:
        with pytest.raises(ValueError, match="same run"):
            BundleOutputConfig(request_id="request-1", run_id="run-1")

    def test_full_lifecycle_success(self, tmp_path: Path) -> None:
        er, vr = _make_registries()
        service = BundleExportService(
            export_registry=er,
            validator_registry=vr,
        )
        source = _make_source()
        config = _make_config(tmp_path)

        result = service.export_bundle(
            source=source,
            config=config,
            tributo_version="0.1.0",
        )

        assert result.status == "succeeded"
        assert result.bundle_id
        assert result.manifest_uri
        assert len(result.manifest_sha256) == 64
        assert result.roles == {"inference": "fp32"}

    def test_result_has_manifest_uri(self, tmp_path: Path) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source()
        config = _make_config(tmp_path)

        result = service.export_bundle(source=source, config=config)

        assert result.manifest_uri.endswith("manifest.json")

    def test_commit_derives_operation_event_without_persisting_it(
        self, tmp_path: Path
    ) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)

        result = service.export_bundle(
            source=_make_source(),
            config=_make_config(tmp_path).model_copy(
                update={"request_id": "event-run-1"}
            ),
        )

        event = service.last_operation_event
        assert event is not None
        assert event.event_kind == "bundle.published"
        assert event.bundle_id == result.bundle_id
        assert event.canonical_uri.endswith(result.bundle_id)
        assert event.manifest_sha256 == result.manifest_sha256
        assert event.correlation_ids["request_id"] == "event-run-1"

    def test_post_commit_uses_repository_manifest_bytes_without_reread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Event derivation consumes the bytes that won the repository commit."""
        from tributo.exporting.bundle_reader import BundleReader

        def reject_reread(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("post-commit manifest must not be fetched again")

        monkeypatch.setattr(BundleReader, "read_manifest", reject_reread)
        monkeypatch.setattr(BundleReader, "read_manifest_with_bytes", reject_reread)
        captured: list[PublishedBundle] = []
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)

        result = service.export_bundle(
            source=_make_source(),
            config=_make_config(tmp_path),
            callback=captured.append,
        )

        assert len(captured) == 1
        committed_bytes = Path(result.manifest_uri).read_bytes()
        assert captured[0].manifest_bytes == committed_bytes
        assert service.last_operation_event is not None

    def test_pre_e2_export_is_rejected_by_default_serving_gate(
        self, tmp_path: Path
    ) -> None:
        """The current export path's empty signature has an actionable error."""
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        result = service.export_bundle(
            source=_make_source(),
            config=_make_config(tmp_path),
        )

        flavors = FlavorRegistry()
        flavors.register(_SignatureGateFlavor)
        loader = BundleModelLoader(flavor_registry=flavors)
        with pytest.raises(
            UnsupportedArtifactFormat,
            match="published without the typed input/output contract",
        ):
            loader.open(result.manifest_uri)

    def test_checkpoint_contract_becomes_typed_manifest_signature(
        self, tmp_path: Path
    ) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source().model_copy(
            update={
                "checkpoint_contract": ExportCheckpointV1(
                    trainer_type="xgboost",
                    architecture_id="xgboost",
                    input_schema=(
                        CheckpointField(
                            name="float_input",
                            dtype="float32",
                            shape=("batch", 2),
                        ),
                    ),
                    output_schema=(
                        CheckpointField(
                            name="prediction",
                            dtype="float32",
                            shape=("batch",),
                        ),
                    ),
                    preprocessing={"type": "none"},
                    task_type="regression",
                    framework="xgboost",
                    framework_version="2.1.0",
                    checkpoint_format_version=1,
                )
            }
        )
        result = service.export_bundle(source=source, config=_make_config(tmp_path))

        from tributo.exporting.bundle_reader import BundleReader

        manifest = BundleReader().read_manifest(result.canonical_uri)
        assert manifest.input_signature.input_fields[0].model_dump() == {
            "name": "float_input",
            "dtype": "float32",
            "shape": ("batch", 2),
        }
        assert manifest.output_signature.output_fields[0].model_dump() == {
            "name": "prediction",
            "dtype": "float32",
            "shape": ("batch",),
        }

    def test_legacy_mode_rejected(self) -> None:
        service = BundleExportService()
        source = _make_source()
        config = BundleOutputConfig()

        with pytest.raises(JobConfigurationError, match="targets"):
            service.export_bundle(source=source, config=config)

    def test_callback_called(self, tmp_path: Path) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source()
        config = _make_config(tmp_path)

        callback_called = []

        def on_bundle(pb: PublishedBundle) -> None:
            callback_called.append(pb)

        result = service.export_bundle(
            source=source,
            config=config,
            callback=on_bundle,
        )

        assert len(callback_called) == 1
        assert callback_called[0].result.bundle_id == result.bundle_id

    def test_callback_staging_accessible(self, tmp_path: Path) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source()
        config = _make_config(tmp_path)

        def on_bundle(pb: PublishedBundle) -> None:
            assert pb.local_bundle_dir.exists()

        service.export_bundle(
            source=source,
            config=config,
            callback=on_bundle,
        )

    def test_callback_error_does_not_raise_by_default(self, tmp_path: Path) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source()
        config = _make_config(tmp_path)

        def failing_callback(pb: PublishedBundle) -> None:
            raise RuntimeError("callback failed")

        result = service.export_bundle(
            source=source,
            config=config,
            callback=failing_callback,
        )
        assert result.status == "succeeded"

    def test_callback_error_raises_when_configured(self, tmp_path: Path) -> None:
        from tributo.exceptions import PostPublishCallbackError

        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source()
        config = _make_config(tmp_path)

        def failing_callback(pb: PublishedBundle) -> None:
            raise RuntimeError("callback failed")

        with pytest.raises(PostPublishCallbackError) as exc_info:
            service.export_bundle(
                source=source,
                config=config,
                callback=failing_callback,
                raise_on_callback_error=True,
            )
        assert exc_info.value.bundle_result is not None
        assert exc_info.value.bundle_result.status == "succeeded"

    def test_required_failure_raises_and_publishes_nothing(
        self, tmp_path: Path
    ) -> None:
        er, vr = _make_registries()
        er.register(_FailingExporter)
        service = BundleExportService(export_registry=er, validator_registry=vr)
        config = BundleOutputConfig(
            bundle_uri=str(tmp_path / "bundles"),
            targets=[
                ExportTarget(
                    name="fp32",
                    format="onnx",
                    exporter_id="failing-exporter-v1",
                )
            ],
            roles={"inference": "fp32"},
        )

        with pytest.raises(BundleExportError, match="Bundle export failed"):
            service.export_bundle(source=_make_source(), config=config)

        # A failed bundle is never published — no manifest on disk.
        assert not (tmp_path / "bundles").exists()

    def test_request_id_determines_bundle_id(self, tmp_path: Path) -> None:
        er, vr = _make_registries()
        service = BundleExportService(export_registry=er, validator_registry=vr)
        source = _make_source()

        # Use separate directories to avoid collision from same-second timestamps.
        config1 = BundleOutputConfig(
            bundle_uri=str(tmp_path / "bundles1"),
            targets=[ExportTarget(name="fp32", format="onnx")],
            request_id="my-stable-id",
        )
        config2 = BundleOutputConfig(
            bundle_uri=str(tmp_path / "bundles2"),
            targets=[ExportTarget(name="fp32", format="onnx")],
            request_id="my-stable-id",
        )

        r1 = service.export_bundle(source=source, config=config1)
        r2 = service.export_bundle(source=source, config=config2)

        parts1 = r1.bundle_id.split("-")
        parts2 = r2.bundle_id.split("-")
        assert parts1[-1] == parts2[-1]  # Same sha256 suffix from same request_id.
