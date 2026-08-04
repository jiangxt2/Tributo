"""Tests for inference/batch_predictor.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo._common.dependencies import MissingOptionalDependency
from tributo.exceptions import DataSourceError, JobConfigurationError
from tributo.inference.batch_predictor import XGBoostONNXPredictor


class TestXGBoostONNXPredictorExceptions:
    """Tests for exception types raised by XGBoostONNXPredictor."""

    def test_model_not_found_raises_data_source_error(self):
        """本地模型路径不存在时应抛出 DataSourceError。"""
        with pytest.raises(DataSourceError, match="ONNX model not found"):
            XGBoostONNXPredictor("/nonexistent/model.onnx")

    def test_boto3_missing_raises_missing_optional_dependency(self):
        """boto3 未安装时应抛出 MissingOptionalDependency（带安装提示）。"""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(
                MissingOptionalDependency, match=r"pip install tributo\[s3\]"
            ):
                predictor = XGBoostONNXPredictor.__new__(XGBoostONNXPredictor)
                predictor._s3_config = {}
                predictor._download_from_s3("s3://bucket/model.onnx")

    def test_onnxruntime_import_error(self):
        """onnxruntime 未安装时应抛出 ImportError。"""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime is required"):
                predictor = XGBoostONNXPredictor.__new__(XGBoostONNXPredictor)
                predictor._init_session("/fake/model.onnx")

    def test_call_returns_predictions(self):
        """正常推理应返回带预测列的 batch。"""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="features")]
        mock_session.run.return_value = [
            np.array([0, 1]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
        ]

        predictor = XGBoostONNXPredictor.__new__(XGBoostONNXPredictor)
        predictor.session = mock_session
        predictor.input_name = "features"
        predictor.return_probs = True
        predictor._runtime = None
        predictor._output_names = ()
        predictor.prediction_column = "prediction"
        predictor.feature_names = None  # 回退到旧行为（无 feature_names 元数据）

        batch = {
            "feat_a": np.array([1.0, 2.0]),
            "feat_b": np.array([3.0, 4.0]),
        }
        result = predictor(batch)

        assert "prediction" in result
        assert result["prediction"].shape == (2, 2)


class TestXGBoostONNXPredictorConfig:
    """XGBoostONNXPredictor predictor_config 解析测试。"""

    def test_default_config(self):
        """默认 predictor_config 应使用默认值。"""
        with patch.object(XGBoostONNXPredictor, "_load_model"):
            predictor = XGBoostONNXPredictor("dummy.onnx", predictor_config=None)

        assert predictor.return_probs is True
        assert predictor.prediction_column == "prediction"
        assert predictor._s3_config == {}

    def test_custom_config(self):
        """自定义 predictor_config 应正确解析。"""
        with patch.object(XGBoostONNXPredictor, "_load_model"):
            predictor = XGBoostONNXPredictor(
                "dummy.onnx",
                predictor_config={
                    "return_probs": False,
                    "prediction_column": "score",
                    "s3_config": {"endpoint": "http://minio:9000"},
                },
            )

        assert predictor.return_probs is False
        assert predictor.prediction_column == "score"
        assert predictor._s3_config["endpoint"] == "http://minio:9000"

    def test_get_feature_names_accepts_predictor_config(self):
        """get_feature_names 应接受 predictor_config 参数。"""
        with pytest.raises(DataSourceError, match="ONNX model not found"):
            XGBoostONNXPredictor.get_feature_names(
                "/nonexistent/model.onnx", {"s3_config": {}}
            )

    def test_legacy_s3_client_uses_shared_storage_resolver(self, monkeypatch):
        """Raw-model S3 downloads share BundleReader's client resolution."""
        captured = {}

        def fake_client(config):
            captured.update(config)
            return object()

        monkeypatch.setattr(
            "tributo._common.storage.get_boto3_client_from_config", fake_client
        )
        XGBoostONNXPredictor._build_s3_client(
            {
                "endpoint": "http://minio:9000",
                "access_key_id": "key",
                "secret_access_key": "secret",
                "region": "test-region",
            }
        )
        assert captured == {
            "endpoint": "http://minio:9000",
            "access_key_id": "key",
            "secret_access_key": "secret",
            "region": "test-region",
        }


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))


class TestXGBoostONNXPredictorBundle:
    """E3: XGBoostONNXPredictor 经 bundle_uri 加载。"""

    def _make_bundle(self, tmp_path) -> str:
        """构造带 typed signature 的本地 ONNX bundle。"""
        from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)
        return str(bundle)

    def test_bundle_and_model_uri_mutually_exclusive(self):
        """model_uri 与 bundle_uri 同时提供应报错。"""
        with pytest.raises(ValueError, match="exactly one"):
            XGBoostONNXPredictor("/x.onnx", bundle_uri="/y")

    def test_bundle_loads_and_predicts(self, tmp_path):
        """bundle 加载后批量推理返回预测列。"""
        bundle = self._make_bundle(tmp_path)
        predictor = XGBoostONNXPredictor(
            bundle_uri=bundle,
            role="inference",
            predictor_config={
                "return_probs": True,
                "feature_names": ["float_input"],
            },
        )
        assert predictor.feature_names == ["float_input"]
        batch = {"float_input": np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)}
        result = predictor(batch)
        assert "prediction" in result
        # return_probs=True: probabilities matrix (one column per class),
        # consistent with the legacy raw-ONNX path.
        assert result["prediction"].shape == (2, 2)

    def test_bundle_explicit_feature_names_wins(self, tmp_path):
        """显式 feature_names 优先于 manifest signature。"""
        bundle = self._make_bundle(tmp_path)
        predictor = XGBoostONNXPredictor(
            bundle_uri=bundle,
            predictor_config={"feature_names": ["float_input"]},
        )
        assert predictor.feature_names == ["float_input"]

    def test_close_idempotent_and_predict_after_close(self, tmp_path):
        """close() 幂等；close 后 predict 仍可用（内存模型契约）。"""
        bundle = self._make_bundle(tmp_path)
        predictor = XGBoostONNXPredictor(
            bundle_uri=bundle,
            role="inference",
            predictor_config={"feature_names": ["float_input"]},
        )

        predictor.close()
        predictor.close()  # 第二次 close 是 no-op

        batch = {"float_input": np.array([[0.5, 0.5]], dtype=np.float32)}
        result = predictor(batch)
        assert "prediction" in result

    def test_bundle_empty_signature_refused_by_default(self, tmp_path):
        """空签名 bundle 默认拒绝；unsafe 放行。"""
        from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path, with_signature=False)
        with pytest.raises(Exception, match="no typed"):
            XGBoostONNXPredictor(bundle_uri=str(bundle))
        # unsafe=True 允许 compat 加载
        predictor = XGBoostONNXPredictor(
            bundle_uri=str(bundle),
            unsafe=True,
            predictor_config={"feature_names": ["float_input"]},
        )
        assert predictor.feature_names == ["float_input"]

    def test_bundle_without_metadata_requires_explicit_feature_columns(
        self, tmp_path, monkeypatch
    ):
        """Tensor names must not be guessed as Dataset column names."""
        from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx
        from tributo.exporting.runtime import BundleModelLoader

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)

        captured = []
        original_open = BundleModelLoader.open

        def capturing_open(self, *args, **kwargs):
            runtime = original_open(self, *args, **kwargs)
            captured.append(runtime)
            return runtime

        monkeypatch.setattr(BundleModelLoader, "open", capturing_open)

        with pytest.raises(JobConfigurationError, match="Dataset columns"):
            XGBoostONNXPredictor(bundle_uri=str(bundle))

        assert len(captured) == 1
        assert captured[0].closed is True


class TestXGBoostONNXPredictorRealFeatureNames:
    """E3 fix: 真实 XGBoost 特征名场景——ONNX metadata 记录表格特征名，
    manifest signature 记录 tensor 名（float_input），两者不再混淆。"""

    @staticmethod
    def _onnx_with_feature_metadata(tmp_path, feature_names: list[str]) -> str:
        """生成 ONNX 模型并注入 feature_names metadata（XGBoost exporter 行为）。"""
        import json

        import onnx

        from tests.serving.bundle_fixtures import make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        model = onnx.load(onnx_path)
        prop = model.metadata_props.add()
        prop.key = "feature_names"
        prop.value = json.dumps(feature_names)
        onnx.save(model, onnx_path)
        return onnx_path

    def test_feature_names_from_metadata_not_signature(self, tmp_path):
        """特征列名（user_id/age/income）从 ONNX metadata 读取，tensor 名从
        manifest signature 读取——两者解耦，均可正常工作。"""
        from tests.serving.bundle_fixtures import build_test_bundle

        feature_names = ["user_id", "age"]  # matches the model's 2-dim input
        onnx_path = self._onnx_with_feature_metadata(tmp_path, feature_names)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)

        predictor = XGBoostONNXPredictor(bundle_uri=str(bundle))
        # 特征列名来自 ONNX metadata（表格特征），而非 manifest 的 tensor 名
        assert predictor.feature_names == feature_names
        assert predictor.input_name == "float_input"

        batch = {
            "user_id": np.array([1, 2], dtype=np.float32),
            "age": np.array([30.0, 40.0], dtype=np.float32),
        }
        result = predictor(batch)
        assert "prediction" in result
        assert result["prediction"].shape[0] == 2

    def test_get_feature_names_bundle_reads_metadata(self, tmp_path):
        """driver 侧 get_feature_names 的 bundle 分支同样从 metadata 读。"""
        from tests.serving.bundle_fixtures import build_test_bundle

        feature_names = ["f0", "f1"]
        onnx_path = self._onnx_with_feature_metadata(tmp_path, feature_names)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)

        names = XGBoostONNXPredictor.get_feature_names(None, {}, bundle_uri=str(bundle))
        assert names == feature_names

    def test_corrupt_feature_names_metadata_fails_fast(self, tmp_path):
        """feature_names metadata 存在但 JSON 损坏 → fail-fast，不静默降级。"""
        import onnx

        from tests.serving.bundle_fixtures import make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        model = onnx.load(onnx_path)
        prop = model.metadata_props.add()
        prop.key = "feature_names"
        prop.value = "{not-json"
        onnx.save(model, onnx_path)

        with pytest.raises(ValueError, match="feature_names.*corrupt"):
            XGBoostONNXPredictor._load_feature_names(onnx_path)

    @pytest.mark.parametrize(
        "value",
        ['{"a": 1}', '["float_input", 42]', "42"],
    )
    def test_non_list_feature_names_metadata_fails_fast(self, tmp_path, value):
        """feature_names 非 list[str]（dict/混合/标量）→ fail-fast。"""
        import onnx

        from tests.serving.bundle_fixtures import make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        model = onnx.load(onnx_path)
        prop = model.metadata_props.add()
        prop.key = "feature_names"
        prop.value = value
        onnx.save(model, onnx_path)

        with pytest.raises(ValueError, match="JSON list of strings"):
            XGBoostONNXPredictor._load_feature_names(onnx_path)

    def test_missing_feature_names_metadata_tolerated(self, tmp_path):
        """无 feature_names metadata → 空列表（合法降级）。"""
        from tests.serving.bundle_fixtures import make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        # The runtime dependency is onnxruntime; the optional authoring
        # package ``onnx`` must not be required by the batch read path.
        with patch.dict(sys.modules, {"onnx": None}):
            assert XGBoostONNXPredictor._load_feature_names(onnx_path) == []
