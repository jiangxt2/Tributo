"""Unit tests for training/xgboost_trainer.py (no Ray/S3 dependency)."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from tributo._common.storage import parse_s3_url
from tributo.training.xgboost_trainer import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RayConfig,
    S3Config,
    TrainingParams,
    XGBoostTrainingConfig,
)


class TestParseS3Url:
    """parse_s3_url 单元测试。"""

    def test_valid_url(self):
        bucket, key = parse_s3_url("s3://my-bucket/path/to/file.parquet")
        assert bucket == "my-bucket"
        assert key == "path/to/file.parquet"

    def test_bucket_only(self):
        bucket, key = parse_s3_url("s3://my-bucket")
        assert bucket == "my-bucket"
        assert key == ""

    def test_bucket_with_trailing_slash(self):
        bucket, key = parse_s3_url("s3://my-bucket/")
        assert bucket == "my-bucket"
        assert key == ""

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid S3 URL"):
            parse_s3_url("http://bucket/key")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid S3 URL"):
            parse_s3_url("")


class TestS3Config:
    """S3Config Pydantic 模型测试。"""

    def test_defaults_all_none(self):
        cfg = S3Config()
        assert cfg.region is None
        assert cfg.access_key_id is None
        assert cfg.secret_access_key is None
        assert cfg.endpoint is None


class TestDataConfig:
    """DataConfig Pydantic 模型测试。"""

    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.type == "csv"
        assert cfg.format == "parquet"
        assert cfg.label_col == "label"
        assert cfg.path is None
        assert cfg.uri is None

    def test_s3_type_with_uri(self):
        cfg = DataConfig(
            type="s3",
            uri="s3://bucket/data.parquet",
            s3=S3Config(endpoint="http://minio:9000"),
        )
        assert cfg.type == "s3"
        assert cfg.uri == "s3://bucket/data.parquet"
        assert cfg.s3.endpoint == "http://minio:9000"

    def test_feature_columns(self):
        cfg = DataConfig(
            type="csv",
            path="data/train.csv",
            label_col="label",
            feature_columns=["a", "b", "c"],
        )
        assert cfg.feature_columns == ["a", "b", "c"]

    def test_feature_columns_default_empty(self):
        cfg = DataConfig()
        assert cfg.feature_columns == []


class TestSetupFeatureSelection:
    """XGBoostTrainerImpl.setup 严格特征选列测试。"""

    def test_setup_selects_feature_columns(self):
        from unittest.mock import MagicMock

        ds = MagicMock()
        ds.schema.return_value.names = ["label", "a", "b", "c", "d"]
        from tributo.training import xgboost_evaluator
        from tributo.training.xgboost_trainer import XGBoostTrainerImpl

        original_filter = xgboost_evaluator.filter_invalid_labels
        original_split = xgboost_evaluator.split_dataset
        xgboost_evaluator.filter_invalid_labels = lambda ds, label_col: ds
        xgboost_evaluator.split_dataset = lambda ds, val_size, test_size, seed: (
            ds,
            None,
            None,
        )
        try:
            trainer = XGBoostTrainerImpl(
                datasets={"train": ds},
                config={
                    "data": {
                        "type": "csv",
                        "path": "data/train.csv",
                        "label_col": "label",
                        "feature_columns": ["a", "b"],
                    },
                    "training": {"val_size": 0, "test_size": 0, "seed": 42},
                },
            )
            trainer.setup()
            ds.select_columns.assert_called_once_with(["a", "b", "label"])
        finally:
            xgboost_evaluator.filter_invalid_labels = original_filter
            xgboost_evaluator.split_dataset = original_split

    def test_setup_skips_select_when_no_feature_columns(self):
        from unittest.mock import MagicMock

        ds = MagicMock()
        ds.schema.return_value.names = ["label", "a", "b"]
        from tributo.training import xgboost_evaluator
        from tributo.training.xgboost_trainer import XGBoostTrainerImpl

        original_filter = xgboost_evaluator.filter_invalid_labels
        original_split = xgboost_evaluator.split_dataset
        xgboost_evaluator.filter_invalid_labels = lambda ds, label_col: ds
        xgboost_evaluator.split_dataset = lambda ds, val_size, test_size, seed: (
            ds,
            None,
            None,
        )
        try:
            trainer = XGBoostTrainerImpl(
                datasets={"train": ds},
                config={
                    "data": {
                        "type": "csv",
                        "path": "data/train.csv",
                        "label_col": "label",
                    },
                    "training": {"val_size": 0, "test_size": 0, "seed": 42},
                },
            )
            trainer.setup()
            ds.select_columns.assert_not_called()
        finally:
            xgboost_evaluator.filter_invalid_labels = original_filter
            xgboost_evaluator.split_dataset = original_split


class TestModelConfig:
    """ModelConfig Pydantic 模型测试。"""

    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.objective == "binary:logistic"
        assert cfg.num_class is None

    def test_multi_class_with_num_class(self):
        cfg = ModelConfig(objective="multi:softprob", num_class=3)
        assert cfg.objective == "multi:softprob"
        assert cfg.num_class == 3

    def test_num_class_lt_2_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(num_class=1)

    def test_extra_fields_allowed(self):
        cfg = ModelConfig(
            objective="binary:logistic",
            max_depth=6,
            eta=0.3,
        )
        assert cfg.max_depth == 6
        assert cfg.eta == 0.3

    def test_num_class_extra_fields_preserved(self):
        """model_dump(exclude={'objective'}) 应保留 num_class。"""
        cfg = ModelConfig(objective="multi:softprob", num_class=3, max_depth=6)
        d = cfg.model_dump(exclude={"objective"})
        assert d["num_class"] == 3
        assert d["max_depth"] == 6


class TestTrainingParams:
    """TrainingParams Pydantic 模型测试。"""

    def test_defaults(self):
        cfg = TrainingParams()
        assert cfg.num_rounds == 100
        assert cfg.early_stopping_rounds is None
        assert cfg.val_size == 0.2
        assert cfg.seed == 42

    def test_num_rounds_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainingParams(num_rounds=0)

    def test_val_size_must_be_lt_1(self):
        with pytest.raises(ValidationError):
            TrainingParams(val_size=1.0)

    def test_val_size_can_be_zero(self):
        cfg = TrainingParams(val_size=0.0)
        assert cfg.val_size == 0.0


class TestXGBoostTrainingConfig:
    """XGBoostTrainingConfig 完整配置测试。"""

    def test_nested_defaults(self):
        cfg = XGBoostTrainingConfig()
        assert cfg.data.type == "csv"
        assert cfg.model.objective == "binary:logistic"
        assert cfg.training.num_rounds == 100
        assert cfg.ray.num_workers == 4
        assert cfg.output.onnx_opset == 12

    def test_full_config(self):
        cfg = XGBoostTrainingConfig(
            data=DataConfig(
                type="s3",
                uri="s3://bucket/data.parquet",
                label_col="target",
            ),
            model=ModelConfig(max_depth=8, eta=0.1),
            training=TrainingParams(num_rounds=200, early_stopping_rounds=10),
            ray=RayConfig(num_workers=2, use_gpu=True),
            output=OutputConfig(onnx_path="model.onnx", onnx_opset=15),
        )
        assert cfg.data.label_col == "target"
        assert cfg.model.max_depth == 8
        assert cfg.training.early_stopping_rounds == 10
        assert cfg.ray.use_gpu is True
        assert cfg.output.onnx_opset == 15

    def test_model_validate_from_dict(self):
        """model_validate 应接受原始字典（模拟 YAML 解析结果）。"""
        raw = {
            "data": {"type": "csv", "path": "/data/train.csv"},
            "model": {"objective": "reg:squarederror"},
            "training": {"num_rounds": 50},
        }
        cfg = XGBoostTrainingConfig.model_validate(raw)
        assert cfg.data.path == "/data/train.csv"
        assert cfg.model.objective == "reg:squarederror"
        assert cfg.training.num_rounds == 50

    def test_data_config_feature_columns(self):
        """XGBoostTrainingConfig 应接受 data.feature_columns 并正确传递。"""
        cfg = XGBoostTrainingConfig(
            data={
                "type": "csv",
                "path": "data/train.csv",
                "label_col": "label",
                "feature_columns": ["a", "b", "c"],
            }
        )
        assert cfg.data.feature_columns == ["a", "b", "c"]


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
