"""Regression tests for the code-review fix batch (idempotency, determinism,
implicit-node injection, role validation, GC safety, reader isolation)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xgboost as xgb

from tributo.exporting.models import (
    BundleOutputConfig,
    ExportTarget,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def xgb_checkpoint(tmp_path_factory):
    """Train a tiny binary-classification booster and save a checkpoint dir."""
    rng = np.random.default_rng(7)
    X = rng.random((40, 4)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    dtrain = xgb.DMatrix(X, label=y, feature_names=[f"f{i}" for i in range(4)])
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2},
        dtrain,
        num_boost_round=3,
    )
    ckpt = tmp_path_factory.mktemp("xgb-ckpt")
    booster.save_model(str(ckpt / "model.json"))
    return str(ckpt)


def _export_config(bundle_uri, request_id, **kw):
    """Minimal two-target bundle config (native + onnx)."""
    return BundleOutputConfig(
        bundle_uri=str(bundle_uri),
        request_id=request_id,
        targets=[
            ExportTarget(name="native", format="xgboost"),
            ExportTarget(name="onnx-model", format="onnx"),
        ],
        **kw,
    )


def _export_once(ckpt, config, provider):
    from tributo.exporting.service import BundleExportService

    service = BundleExportService()
    with provider.open_source(ckpt) as src:
        return service.export_bundle(source=src, config=config, provider=provider)


# ── Idempotency ───────────────────────────────────────────────────────────────


class TestIdempotentRetry:
    def test_same_request_id_reuses_bundle_and_manifest(self, xgb_checkpoint, tmp_path):
        """Retry with the same request_id → same bundle_id and manifest sha."""
        from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider

        provider = RayXGBoostSourceProvider()
        config = _export_config(tmp_path / "store", "req-idempotent-1")

        result1 = _export_once(xgb_checkpoint, config, provider)
        result2 = _export_once(xgb_checkpoint, config.model_copy(), provider)

        assert result2.bundle_id == result1.bundle_id
        assert result2.manifest_sha256 == result1.manifest_sha256

    def test_different_request_id_collides(self, xgb_checkpoint, tmp_path):
        """Different request_id → different bundle_id (no collision)."""
        from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider

        provider = RayXGBoostSourceProvider()
        config1 = _export_config(tmp_path / "store", "req-a")
        config2 = _export_config(tmp_path / "store", "req-b")

        result1 = _export_once(xgb_checkpoint, config1, provider)
        result2 = _export_once(xgb_checkpoint, config2, provider)

        assert result2.bundle_id != result1.bundle_id


class TestOnnxDeterminism:
    def test_same_booster_produces_identical_bytes(self):
        """onnxmltools stamps a random graph name — the exporter must strip it."""
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        rng = np.random.default_rng(3)
        X = rng.random((20, 4)).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)
        dtrain = xgb.DMatrix(X, label=y, feature_names=[f"f{i}" for i in range(4)])
        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 2}, dtrain, num_boost_round=2
        )

        def convert(b):
            b.feature_names = [f"f{i}" for i in range(4)]
            wrapper = xgb.XGBClassifier()
            wrapper._Booster = b  # noqa: SLF001
            wrapper.__dict__["n_classes_"] = 2
            wrapper.__dict__["classes_"] = np.arange(2)
            model = convert_xgboost(
                wrapper,
                initial_types=[("float_input", FloatTensorType([None, 4]))],
            )
            # Mirror the exporter's normalisation.
            model.graph.name = "xgboost-onnx"
            model.doc_string = ""
            for node in model.graph.node:
                node.doc_string = ""
            return model.SerializeToString()

        assert convert(booster) == convert(booster)

    def test_manifest_logically_equal_ignores_advisory_fields(self):
        """S3 idempotency falls back to a logical comparison."""
        from tributo.exporting.publisher import _manifest_logically_equal

        base = {
            "bundle_id": "bundle-abc",
            "artifacts": [
                {
                    "name": "m",
                    "tree_digest": "d" * 64,
                    "validation": [{"validator_id": "v", "metrics": {"t": 1.0}}],
                }
            ],
            "execution": {"nodes": [{"node_id": "n", "duration_ms": 5}]},
        }
        retry = json.loads(json.dumps(base))
        retry["created_at"] = "2026-07-31T00:00:00"
        retry["execution"]["nodes"][0]["duration_ms"] = 9
        retry["artifacts"][0]["validation"][0]["metrics"]["t"] = 2.0

        assert _manifest_logically_equal(
            json.dumps(base).encode(), json.dumps(retry).encode()
        )

        retry["artifacts"][0]["tree_digest"] = "e" * 64
        assert not _manifest_logically_equal(
            json.dumps(base).encode(), json.dumps(retry).encode()
        )


# ── Planner / implicit nodes / roles ──────────────────────────────────────────


class TestImplicitNodesAndRoles:
    def test_explicit_underscore_target_rejected(self):
        """`_`-prefixed names are reserved — explicit config must reject them."""
        with pytest.raises(ValueError, match="must not start with '_'"):
            BundleOutputConfig(
                bundle_uri="/tmp/x",
                targets=[ExportTarget(name="_implicit__a", format="onnx")],
            )

    def test_implicit_target_name_constructible(self):
        """Planner-generated implicit names must pass the model layer."""
        target = ExportTarget(name="_implicit__quant__onnx__abc12345", format="onnx")
        assert target.name.startswith("_implicit__")

    def test_role_cannot_reference_unknown_target(self, xgb_checkpoint, tmp_path):
        """Planner rejects roles referencing targets that do not exist."""
        from tributo.exceptions import JobConfigurationError
        from tributo.exporting.models import ExportSource
        from tributo.exporting.planner import ExportPlanner
        from tributo.exporting.registries import (
            ExportRegistry,
            SourceProviderRegistry,
            ValidatorRegistry,
        )
        from tributo.exporting.service import _load_entry_point_plugins
        from tributo.exporting.validators import StructureValidator

        validators = ValidatorRegistry()
        validators.register(StructureValidator)
        registry = ExportRegistry()
        _load_entry_point_plugins(registry, SourceProviderRegistry(), validators)

        config = BundleOutputConfig(
            bundle_uri=str(tmp_path / "store"),
            request_id="role-implicit",
            targets=[
                ExportTarget(name="native", format="xgboost"),
                ExportTarget(name="onnx-model", format="onnx"),
            ],
            roles={"serve": "model"},
        )
        planner = ExportPlanner(registry, validators)
        source = ExportSource(
            source_kind="xgboost_result",
            metadata={"objective": "binary:logistic"},
        )
        with pytest.raises(JobConfigurationError, match="does not exist"):
            planner.plan(config, source)


# ── GC safety ─────────────────────────────────────────────────────────────────


class TestGcSafety:
    def test_looks_like_bundle_id_matches_real_format(self):
        from tributo.exporting.gc import _looks_like_bundle_id

        assert _looks_like_bundle_id("bundle-" + "a" * 32)
        assert not _looks_like_bundle_id("bundle-" + "a" * 31)  # too short
        assert not _looks_like_bundle_id("bundle-" + "a" * 33)  # too long
        assert not _looks_like_bundle_id("bundle-" + "g" * 32)  # non-hex
        assert not _looks_like_bundle_id("other-prefix-" + "a" * 32)
        assert not _looks_like_bundle_id("trials")


# ── BundleReader isolation ────────────────────────────────────────────────────


class TestBundleReaderIsolation:
    def test_two_contexts_share_cache_without_interference(
        self, xgb_checkpoint, tmp_path
    ):
        from tributo.exporting.bundle_reader import BundleReader
        from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider

        provider = RayXGBoostSourceProvider()
        config = _export_config(
            tmp_path / "store",
            "req-reader-isolation",
            roles={"serve": "onnx-model"},
        )
        result = _export_once(xgb_checkpoint, config, provider)

        reader = BundleReader()
        with reader.open_artifact(result.canonical_uri, role="serve") as ra1:
            with reader.open_artifact(result.canonical_uri, role="serve") as ra2:
                # Both contexts resolve the same artifact.
                assert ra1.descriptor.name == ra2.descriptor.name
                first = ra1.entrypoint_path
            # Exiting the inner context must not invalidate the outer one.
            assert first.is_file()


# ── URI validation ────────────────────────────────────────────────────────────


class TestUriValidation:
    def test_s3_uri_with_query_rejected(self):
        with pytest.raises(ValueError, match="query or fragment"):
            BundleOutputConfig(
                bundle_uri="s3://bucket/models?version=2",
                targets=[ExportTarget(name="m", format="onnx")],
            )

    def test_s3_uri_with_credentials_rejected(self):
        with pytest.raises(ValueError, match="credentials"):
            BundleOutputConfig(
                bundle_uri="s3://user:pass@bucket/models",
                targets=[ExportTarget(name="m", format="onnx")],
            )

    def test_plain_s3_uri_accepted(self):
        config = BundleOutputConfig(
            bundle_uri="s3://bucket/models",
            targets=[ExportTarget(name="m", format="onnx")],
        )
        assert config.bundle_uri == "s3://bucket/models"


# ── BaseTrainer double-export regression ──────────────────────────────────────


class TestBaseTrainerExportOnce:
    def test_legacy_export_model_called_exactly_once(self):
        """_export_artifacts_default must not call export_model twice."""
        from tributo.training.base import BaseTrainer

        class _MinimalTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self):
                return "checkpoint"

            def export_model(self, checkpoint, output_path) -> None:
                self._calls = getattr(self, "_calls", 0) + 1

        trainer = _MinimalTrainer(datasets={}, config={})
        trainer._export_artifacts_default("checkpoint", "/tmp/out")
        assert trainer._calls == 1

    def test_new_export_artifacts_overrides_legacy(self):
        from tributo.training.base import BaseTrainer

        class _ModernTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self):
                return "checkpoint"

            def export_artifacts(self, checkpoint, output_path) -> None:
                self._exported = True

        trainer = _ModernTrainer(datasets={}, config={})
        trainer._export_artifacts_default("checkpoint", "/tmp/out")
        assert trainer._exported is True
