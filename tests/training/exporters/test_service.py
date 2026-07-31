"""Tests for BundleExportService — full lifecycle orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from tributo.exporting.models import (
    ArtifactDraft,
    BundleOutputConfig,
    DraftFile,
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
    ValidatorRegistry,
)
from tributo.exporting.service import BundleExportService

# ── Fake components ───────────────────────────────────────────────────────────


class _MinOpts(BaseModel):
    """No-op options model."""

    model_config = {"extra": "forbid"}


class _TestExporter:
    """Exporter that writes a simple model file."""

    api_version: int = 1
    exporter_id: str = "test-exporter-v1"
    priority: int = 100
    output_format: str = "onnx"
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

    def test_legacy_mode_rejected(self) -> None:
        service = BundleExportService()
        source = _make_source()
        config = BundleOutputConfig()

        with pytest.raises(ValueError, match="targets"):
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
