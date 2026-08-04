"""Unit tests for training/xgboost_trainer.py (no Ray/S3 dependency)."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from tributo._common.storage import parse_s3_url
from tributo.training.xgboost_trainer import (
    ModelConfig,
    OutputConfig,
    RayConfig,
    S3Config,
    TrainingParams,
    XGBoostDataConfig,
    XGBoostTrainingConfig,
    _merge_xgb_eval_results,
    _populate_xgb_eval_metrics,
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


class TestXGBoostDataConfig:
    """XGBoostDataConfig Pydantic 模型测试。"""

    def test_defaults(self):
        cfg = XGBoostDataConfig()
        assert cfg.type == "csv"
        assert cfg.format == "parquet"
        assert cfg.label_col == "label"
        assert cfg.path is None
        assert cfg.uri is None

    def test_s3_type_with_uri(self):
        cfg = XGBoostDataConfig(
            type="s3",
            uri="s3://bucket/data.parquet",
            s3=S3Config(endpoint="http://minio:9000"),
        )
        assert cfg.type == "s3"
        assert cfg.uri == "s3://bucket/data.parquet"
        assert cfg.s3.endpoint == "http://minio:9000"

    def test_feature_columns(self):
        cfg = XGBoostDataConfig(
            type="csv",
            path="data/train.csv",
            label_col="label",
            feature_columns=["a", "b", "c"],
        )
        assert cfg.feature_columns == ["a", "b", "c"]

    def test_feature_columns_default_empty(self):
        cfg = XGBoostDataConfig()
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

    def test_reserved_external_memory_rejected(self):
        """external_memory 作为保留参数被结构化拒绝。"""
        with pytest.raises(ValidationError, match="external_memory"):
            XGBoostTrainingConfig.model_validate({"model": {"external_memory": True}})

    def test_reserved_data_iter_rejected(self):
        """data_iter（DataIter 回调）同样被拒绝。"""
        with pytest.raises(ValidationError, match="data_iter"):
            XGBoostTrainingConfig.model_validate({"model": {"data_iter": lambda: None}})

    def test_build_trainer_raw_config_rejects_reserved(self):
        """build_trainer（raw dict 入口）同样拒绝保留参数。"""
        from unittest.mock import MagicMock

        from tributo.exceptions import JobConfigurationError
        from tributo.training.xgboost_trainer import build_trainer

        with pytest.raises(JobConfigurationError, match="external_memory"):
            build_trainer(
                ray_dataset=MagicMock(),
                train_config={
                    "xgb_params": {
                        "objective": "binary:logistic",
                        "external_memory": True,
                    },
                    "num_rounds": 3,
                },
            )

    def test_build_trainer_raw_config_allows_normal_params(self):
        """非保留参数在 raw 入口继续透传（不误拒）。"""
        pytest.importorskip("xgboost")  # build_trainer 内部 import ray.train.xgboost
        from unittest.mock import MagicMock

        from tributo.training.xgboost_trainer import build_trainer

        # 校验发生在 ray import 之前——MagicMock dataset 足够到达校验点
        result = build_trainer(
            ray_dataset=MagicMock(),
            train_config={
                "xgb_params": {"objective": "binary:logistic", "max_depth": 8},
                "num_rounds": 3,
            },
        )
        assert result is not None

    def test_normal_extra_params_still_allowed(self):
        """非保留的 native 参数继续透传。"""
        cfg = XGBoostTrainingConfig.model_validate(
            {"model": {"max_depth": 8, "eta": 0.1}}
        )
        assert cfg.model.max_depth == 8

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

    def test_max_rows_per_worker_legacy_name(self):
        cfg = TrainingParams(max_rows_per_worker=500)
        assert cfg.max_rows_per_worker == 500

    def test_max_input_rows_per_worker_alias(self):
        """max_input_rows_per_worker 是 max_rows_per_worker 的别名。"""
        cfg = TrainingParams(max_input_rows_per_worker=500)
        assert cfg.max_rows_per_worker == 500
        # model_dump 保留旧字段名（train_loop_config 兼容）
        assert cfg.model_dump()["max_rows_per_worker"] == 500


class TestXGBoostTrainingConfig:
    """XGBoostTrainingConfig 完整配置测试。"""

    def test_nested_defaults(self):
        cfg = XGBoostTrainingConfig()
        assert cfg.data.type == "csv"
        assert cfg.model.objective == "binary:logistic"
        assert cfg.training.num_rounds == 100
        assert cfg.ray.num_workers == 4
        assert cfg.output.onnx_opset == 12

    def test_default_resource_budget_is_active(self):
        """resource 预算默认启用（无条件安全基线）。"""
        from tributo.training.resource import MIB

        cfg = XGBoostTrainingConfig()
        assert cfg.resource.max_batch_bytes == 64 * MIB
        assert cfg.resource.max_worker_materialization_bytes == 1024 * MIB
        assert cfg.resource.max_input_rows_per_worker is None

    def test_custom_resource_budget(self):
        cfg = XGBoostTrainingConfig(
            resource={"max_batch_bytes": 1024, "max_input_rows_per_worker": 100}
        )
        assert cfg.resource.max_batch_bytes == 1024
        assert cfg.resource.max_input_rows_per_worker == 100

    def test_full_config(self):
        cfg = XGBoostTrainingConfig(
            data=XGBoostDataConfig(
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

    def test_resume_eval_metrics_include_current_and_history_keys(self):
        metrics = {}
        _populate_xgb_eval_metrics(
            metrics,
            {"train": {"logloss": [0.8, 0.4]}},
        )
        assert metrics["train-logloss"] == 0.4
        assert metrics["train-logloss_history"] == [0.8, 0.4]

    def test_resume_eval_history_is_continuous(self):
        merged = _merge_xgb_eval_results(
            {"train": {"logloss": [0.8, 0.4]}},
            {"train": {"logloss": [0.3, 0.2]}},
        )
        assert merged == {"train": {"logloss": [0.8, 0.4, 0.3, 0.2]}}


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
