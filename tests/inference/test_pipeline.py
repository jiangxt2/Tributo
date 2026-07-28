"""Tests for inference/pipeline.py (InferenceConfig + JSON parsing)."""

from __future__ import annotations

import json
import sys
import tempfile
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tributo.exceptions import JobConfigurationError
from tributo.inference.pipeline import InferenceConfig


class TestInferenceConfig:
    """InferenceConfig Pydantic 验证测试。"""

    def test_defaults(self):
        """默认值应符合预期。"""
        cfg = InferenceConfig(
            input_uri="s3://bucket/input.parquet",
            output_uri="s3://bucket/output/",
            model_uri="s3://bucket/model.onnx",
        )
        assert cfg.batch_size == 4096
        assert cfg.concurrency == 4
        assert cfg.num_cpus_per_actor == 1.0
        assert cfg.num_gpus_per_actor == 0.0
        assert cfg.output_compression == "zstd"
        assert cfg.predictor_config == {}
        assert cfg.feature_columns == []
        # 兼容属性默认值
        assert cfg.return_probs is True
        assert cfg.prediction_column == "prediction"

    def test_frozen(self):
        """InferenceConfig 应不可变。"""
        cfg = InferenceConfig(
            input_uri="s3://bucket/in.parquet",
            output_uri="s3://bucket/out/",
            model_uri="s3://bucket/model.onnx",
        )
        with pytest.raises(ValidationError):
            cfg.batch_size = 100  # type: ignore[misc]

    def test_empty_input_path_raises(self):
        """空 input_uri 应校验失败。"""
        with pytest.raises(ValidationError):
            InferenceConfig(
                input_uri="",
                output_uri="s3://bucket/output/",
                model_uri="s3://bucket/model.onnx",
            )

    def test_empty_model_uri_raises(self):
        """空 model_uri 应校验失败。"""
        with pytest.raises(ValidationError):
            InferenceConfig(
                input_uri="s3://bucket/input.parquet",
                output_uri="s3://bucket/output/",
                model_uri="",
            )


class TestRunInferenceFromJson:
    """run_inference_from_json 解析测试。"""

    def _write_json(self, data: dict) -> str:
        """写入临时 JSON 文件并返回路径。"""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_invalid_config_root_raises(self):
        """JSON 根节点非 dict 应抛出 JobConfigurationError。"""
        from tributo.inference.pipeline import run_inference_from_json

        path = self._write_json([1, 2, 3])
        with pytest.raises(JobConfigurationError, match="must be a mapping"):
            run_inference_from_json(path)

    def test_validation_error_wrapped(self):
        """Pydantic 校验失败应包装为 JobConfigurationError。"""
        from tributo.inference.pipeline import run_inference_from_json

        # 缺少必填字段
        path = self._write_json({"data": {}, "model": {}, "output": {}})
        with pytest.raises(JobConfigurationError, match="Invalid inference config"):
            run_inference_from_json(path)

    @patch("tributo.inference.pipeline.run_batch_inference")
    def test_deprecated_field_compat(self, mock_run):
        """旧字段 input / path 应兼容解析。"""
        from tributo.inference.pipeline import run_inference_from_json

        mock_run.return_value = {"status": "completed"}
        path = self._write_json(
            {
                "data": {
                    "input": "s3://bucket/input.parquet",
                    "feature_columns": ["f0", "f1"],
                },
                "model": {"path": "s3://bucket/model.onnx"},
                "output": {"path": "s3://bucket/output/"},
            }
        )
        run_inference_from_json(path)

        call_cfg = mock_run.call_args[0][0]
        assert call_cfg.input_uri == "s3://bucket/input.parquet"
        assert call_cfg.model_uri == "s3://bucket/model.onnx"
        assert call_cfg.output_uri == "s3://bucket/output/"

    @patch("tributo.inference.pipeline.run_batch_inference")
    def test_new_uri_fields(self, mock_run):
        """新字段 uri 应正确解析。"""
        from tributo.inference.pipeline import run_inference_from_json

        mock_run.return_value = {"status": "completed"}
        path = self._write_json(
            {
                "data": {
                    "uri": "s3://bucket/input.parquet",
                    "feature_columns": ["f0"],
                },
                "model": {"uri": "s3://bucket/model.onnx", "return_probs": False},
                "output": {
                    "uri": "s3://bucket/output/",
                    "prediction_column": "score",
                    "compression": "snappy",
                },
                "ray": {"concurrency": 8, "batch_size": 2048},
            }
        )
        run_inference_from_json(path)

        call_cfg = mock_run.call_args[0][0]
        assert call_cfg.input_uri == "s3://bucket/input.parquet"
        assert call_cfg.model_uri == "s3://bucket/model.onnx"
        assert call_cfg.predictor_config["return_probs"] is False
        assert call_cfg.predictor_config["prediction_column"] == "score"
        assert call_cfg.output_compression == "snappy"
        assert call_cfg.concurrency == 8
        assert call_cfg.batch_size == 2048


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
