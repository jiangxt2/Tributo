"""End-to-end test for the P0 vertical slice: XGBoost → ONNX → Local bundle.

Requires ``xgboost`` and ``onnxmltools``.  Skipped when not installed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.integration.test_export_helpers import (
    _assert_onnx_signature_matches_manifest,
)

# ── Skip if optional deps are missing ──────────────────────────────────────────

xgboost = pytest.importorskip("xgboost", reason="xgboost not installed")
onnxmltools = pytest.importorskip("onnxmltools", reason="onnxmltools not installed")
numpy = pytest.importorskip("numpy", reason="numpy not installed")

# Needs optional deps (xgboost/onnxmltools) — owned by the future
# export-xgboost job, so the unit job skips this module entirely.
pytestmark = pytest.mark.integration


# ── Helpers ────────────────────────────────────────────────────────────────────


def _train_tiny_booster() -> Any:
    """Train a minimal XGBoost booster on synthetic data."""
    X = numpy.random.randn(100, 5).astype(numpy.float32)
    y = numpy.random.randint(0, 2, 100).astype(numpy.float32)
    dtrain = xgboost.DMatrix(X, label=y)
    params = {"max_depth": 2, "eta": 0.1, "objective": "binary:logistic"}
    booster = xgboost.train(params, dtrain, num_boost_round=3)
    booster.feature_names = [f"f{i}" for i in range(5)]
    return booster


def _train_tiny_regressor() -> Any:
    """Train a minimal squared-error XGBoost regressor."""
    X = numpy.random.randn(100, 3).astype(numpy.float32)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.25).astype(numpy.float32)
    dtrain = xgboost.DMatrix(X, label=y)
    params = {"max_depth": 2, "eta": 0.1, "objective": "reg:squarederror"}
    booster = xgboost.train(params, dtrain, num_boost_round=3)
    booster.feature_names = [f"f{i}" for i in range(3)]
    return booster


def _assert_typed_manifest(manifest: Any) -> None:
    """Require non-empty typed signatures in the published Manifest."""
    assert manifest.input_signature is not None
    assert manifest.output_signature is not None

    fields = (
        *manifest.input_signature.input_fields,
        *manifest.output_signature.output_fields,
    )
    assert fields
    assert all(field.name and field.dtype and field.shape for field in fields)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestExporterSupports:
    """Verify exporter protocol conformance."""

    def test_supports_xgboost_result(self) -> None:
        """XGBoostONNXExporter.supports() accepts numeric-feature classification."""
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        cls = type(XGBoostONNXExporter())
        result = cls.supports(
            SupportRequest(
                source_kind="xgboost_result",
                source_metadata={"objective": "binary:logistic"},
            )
        )
        assert result.supported is True

    def test_supports_squared_error_regression_objective(self) -> None:
        """Standard squared-error regression is supported at plan time."""
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        cls = type(XGBoostONNXExporter())
        result = cls.supports(
            SupportRequest(
                source_kind="xgboost_result",
                source_metadata={"objective": "reg:squarederror"},
            )
        )
        assert result.supported is True
        assert result.code == "OK"

    def test_rejects_unknown_objective(self) -> None:
        """A missing objective cannot be verified — reject at plan time."""
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        cls = type(XGBoostONNXExporter())
        result = cls.supports(SupportRequest(source_kind="xgboost_result"))
        assert result.supported is False
        assert result.code == "UNKNOWN_OBJECTIVE"

    def test_rejects_binary_hinge_objective(self) -> None:
        """Hinge output does not satisfy the classifier probability contract."""
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        result = XGBoostONNXExporter.supports(
            SupportRequest(
                source_kind="xgboost_result",
                source_metadata={"objective": "binary:hinge"},
            )
        )
        assert result.supported is False
        assert result.code == "UNSUPPORTED_OBJECTIVE"

    def test_rejects_categorical_features(self) -> None:
        """Categorical feature types are rejected at plan time."""
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        cls = type(XGBoostONNXExporter())
        result = cls.supports(
            SupportRequest(
                source_kind="xgboost_result",
                source_metadata={
                    "objective": "binary:logistic",
                    "has_categorical_features": True,
                },
            )
        )
        assert result.supported is False
        assert result.code == "CATEGORICAL_FEATURES"

    def test_rejects_unknown_source_kind(self) -> None:
        """XGBoostONNXExporter.supports() rejects unknown kinds."""
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        cls = type(XGBoostONNXExporter())
        result = cls.supports(SupportRequest(source_kind="pytorch_result"))
        assert result.supported is False
        assert result.code == "UNSUPPORTED_SOURCE_KIND"

    def test_classvars_are_correct(self) -> None:
        """Verify XGBoostONNXExporter declares correct metadata."""
        from tributo.integrations.exporters.xgboost_onnx import (
            XGBoostONNXExporter,
        )

        cls = XGBoostONNXExporter
        assert cls.api_version == 2
        assert cls.exporter_id == "xgboost-onnx-v1"
        assert cls.output_format == "onnx"
        assert cls.priority == 100
        assert cls.mutates_source is True
        assert cls.upstream_requirements == ()


class TestExportPipelineWithRealExporter:
    """End-to-end pipeline using the real XGBoostONNXExporter."""

    @pytest.mark.slow
    def test_xgboost_to_onnx_local_bundle(self) -> None:
        """Train tiny XGBoost → export to ONNX → verify bundle exists."""
        from tributo.exporting.models import (
            BundleOutputConfig,
            ExportTarget,
        )
        from tributo.exporting.service import BundleExportService
        from tributo.integrations.sources.ray_xgboost import (
            RayXGBoostSourceProvider,
        )

        booster = _train_tiny_booster()
        tmpdir = Path(tempfile.mkdtemp(prefix="tributo-e2e-"))
        try:
            service = BundleExportService()

            checkpoint_dir = tmpdir / "checkpoint"
            checkpoint_dir.mkdir()
            booster.save_model(str(checkpoint_dir / "model.json"))
            config = BundleOutputConfig(
                bundle_uri=str(tmpdir / "bundles"),
                targets=[
                    ExportTarget(name="model", format="onnx", options={"opset": 12}),
                ],
            )

            provider = RayXGBoostSourceProvider()
            with provider.open_source(str(checkpoint_dir)) as source:
                result = service.export_bundle(
                    source=source,
                    config=config,
                )

            assert result.status in ("succeeded", "partial")
            assert result.bundle_id
            assert result.canonical_uri

            # Assert ONNX file exists in the bundle.
            bundle_dir = Path(result.canonical_uri)
            onnx_path = bundle_dir / "artifacts" / "model" / "model.onnx"
            assert onnx_path.is_file(), f"ONNX file not found at {onnx_path}"

            from tributo.exporting.bundle_reader import BundleReader

            manifest = BundleReader().read_manifest(result.canonical_uri)
            _assert_typed_manifest(manifest)
            assert manifest.input_signature.input_fields[0].dtype == "float32"
            assert manifest.output_signature.output_fields[0].name == "label"
            assert manifest.output_signature.output_fields[1].name == "probabilities"
            _assert_onnx_signature_matches_manifest(onnx_path, manifest)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.slow
    def test_xgboost_regression_bundle_has_typed_signature(self) -> None:
        """Train regression → provider → Bundle → ONNX with typed outputs."""
        from tributo.exporting.bundle_reader import BundleReader
        from tributo.exporting.models import BundleOutputConfig, ExportTarget
        from tributo.exporting.service import BundleExportService
        from tributo.integrations.sources.ray_xgboost import (
            RayXGBoostSourceProvider,
        )

        booster = _train_tiny_regressor()
        tmpdir = Path(tempfile.mkdtemp(prefix="tributo-e2e-regression-"))
        try:
            checkpoint_dir = tmpdir / "checkpoint"
            checkpoint_dir.mkdir()
            booster.save_model(str(checkpoint_dir / "model.json"))

            config = BundleOutputConfig(
                bundle_uri=str(tmpdir / "bundles"),
                targets=[
                    ExportTarget(
                        name="regression", format="onnx", options={"opset": 12}
                    )
                ],
                roles={"inference": "regression"},
            )
            provider = RayXGBoostSourceProvider()
            with provider.open_source(str(checkpoint_dir)) as source:
                result = BundleExportService().export_bundle(
                    source=source,
                    config=config,
                )

            assert result.status == "succeeded"
            manifest = BundleReader().read_manifest(result.canonical_uri)
            bundle_dir = Path(result.canonical_uri)
            onnx_path = bundle_dir / "artifacts" / "regression" / "model.onnx"
            assert onnx_path.is_file(), f"ONNX file not found at {onnx_path}"
            _assert_typed_manifest(manifest)
            assert manifest.input_signature.input_fields[0].dtype == "float32"
            assert manifest.output_signature.output_fields[0].name == "prediction"
            assert manifest.output_signature.output_fields[0].dtype == "float32"
            assert manifest.output_signature.output_fields[0].shape == ("batch", 1)
            _assert_onnx_signature_matches_manifest(onnx_path, manifest)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
