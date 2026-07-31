"""End-to-end test for the P0 vertical slice: XGBoost → ONNX → Local bundle.

Requires ``xgboost`` and ``onnxmltools``.  Skipped when not installed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

# ── Skip if optional deps are missing ──────────────────────────────────────────

xgboost = pytest.importorskip("xgboost", reason="xgboost not installed")
onnxmltools = pytest.importorskip("onnxmltools", reason="onnxmltools not installed")
numpy = pytest.importorskip("numpy", reason="numpy not installed")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _train_tiny_booster() -> xgboost.Booster:  # type: ignore[no-any-unimported]
    """Train a minimal XGBoost booster on synthetic data."""
    X = numpy.random.randn(100, 5).astype(numpy.float32)
    y = numpy.random.randint(0, 2, 100).astype(numpy.float32)
    dtrain = xgboost.DMatrix(X, label=y)
    params = {"max_depth": 2, "eta": 0.1, "objective": "binary:logistic"}
    booster = xgboost.train(params, dtrain, num_boost_round=3)
    booster.feature_names = [f"f{i}" for i in range(5)]
    return booster


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

    def test_rejects_regression_objective(self) -> None:
        """Regression / ranking / count objectives are rejected at plan time."""
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
        assert result.supported is False
        assert result.code == "UNSUPPORTED_OBJECTIVE"

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
        assert cls.api_version == 1
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
            ExportSource,
            ExportTarget,
        )
        from tributo.exporting.service import BundleExportService

        booster = _train_tiny_booster()
        tmpdir = Path(tempfile.mkdtemp(prefix="tributo-e2e-"))
        try:
            service = BundleExportService()

            source = ExportSource(
                source_kind="xgboost_result",
                model_object=booster,
                feature_schema={"feature_names": list(booster.feature_names)}
                if booster.feature_names
                else {},
                metadata={
                    "framework": "xgboost",
                    "framework_version": xgboost.__version__,
                    "n_features": 5,
                },
            )
            config = BundleOutputConfig(
                bundle_uri=str(tmpdir / "bundles"),
                targets=[
                    ExportTarget(name="model", format="onnx", options={"opset": 12}),
                ],
            )

            result = service.export_bundle(source=source, config=config)

            assert result.status in ("succeeded", "partial")
            assert result.bundle_id
            assert result.canonical_uri

            # Assert ONNX file exists in the bundle.
            bundle_dir = Path(result.canonical_uri)
            onnx_path = bundle_dir / "artifacts" / "model" / "model.onnx"
            assert onnx_path.is_file(), f"ONNX file not found at {onnx_path}"

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
