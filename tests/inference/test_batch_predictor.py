"""Tests for inference/batch_predictor.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo.exceptions import DataSourceError, JobConfigurationError
from tributo.inference.batch_predictor import XGBoostONNXPredictor


class TestXGBoostONNXPredictorExceptions:
    """Tests for exception types raised by XGBoostONNXPredictor."""

    def test_model_not_found_raises_data_source_error(self):
        """本地模型路径不存在时应抛出 DataSourceError。"""
        with pytest.raises(DataSourceError, match="ONNX model not found"):
            XGBoostONNXPredictor("/nonexistent/model.onnx")

    def test_boto3_import_error_raises_job_configuration_error(self):
        """boto3 未安装时应抛出 JobConfigurationError。"""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(JobConfigurationError, match="boto3 is required"):
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


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
