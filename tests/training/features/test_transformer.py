"""特征预处理器测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tributo.training.features.column_types import (
    DenseFeat,
    NormMethod,
    SparseFeat,
)
from tributo.training.features.transformer import FeatureTransformer


class TestFeatureTransformer:
    """FeatureTransformer 测试。"""

    def test_fit_transform_sparse(self):
        """测试 Sparse 特征预处理。"""
        features = [SparseFeat(name="category", vocab_size=10)]
        data = {"category": np.array(["a", "b", "c", "a", "b"])}

        transformer = FeatureTransformer(features)
        result = transformer.fit_transform(data)

        # 应该有 3 个唯一值
        assert "category" in result
        assert result["category"].dtype == np.int64
        assert len(np.unique(result["category"])) == 3

    def test_fit_transform_dense_minmax(self):
        """测试 Dense 特征 MinMax 归一化。"""
        features = [DenseFeat(name="value", norm=NormMethod.MINMAX)]
        data = {"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        transformer = FeatureTransformer(features)
        result = transformer.fit_transform(data)

        assert "value" in result
        assert result["value"].dtype == np.float32
        assert result["value"].min() == pytest.approx(0.0)
        assert result["value"].max() == pytest.approx(1.0)

    def test_fit_transform_dense_standard(self):
        """测试 Dense 特征 Standard 归一化。"""
        features = [DenseFeat(name="value", norm=NormMethod.STANDARD)]
        data = {"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        transformer = FeatureTransformer(features)
        result = transformer.fit_transform(data)

        assert "value" in result
        # Standard 归一化后均值应接近 0
        assert np.mean(result["value"]) == pytest.approx(0.0, abs=1e-6)

    def test_fit_transform_dense_log(self):
        """测试 Dense 特征 Log 变换。"""
        features = [DenseFeat(name="value", norm=NormMethod.LOG)]
        data = {"value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        transformer = FeatureTransformer(features)
        result = transformer.fit_transform(data)

        assert "value" in result
        assert np.all(result["value"] >= 0)  # Log 变换后应该非负

    def test_hash_encoding(self):
        """测试 Hash Encoding。"""
        features = [
            SparseFeat(
                name="user_id", vocab_size=1000, use_hash=True, hash_bucket_size=100
            )
        ]
        data = {"user_id": np.array(["user_1", "user_2", "user_3"])}

        transformer = FeatureTransformer(features)
        result = transformer.fit_transform(data)

        assert "user_id" in result
        assert result["user_id"].dtype == np.int64
        assert np.all(result["user_id"] >= 0)
        assert np.all(result["user_id"] < 100)

    def test_not_fitted_error(self):
        """测试未拟合时转换报错。"""
        features = [DenseFeat(name="value")]
        transformer = FeatureTransformer(features)

        with pytest.raises(RuntimeError, match="not fitted"):
            transformer.transform({"value": np.array([1.0])})

    def test_save_load(self):
        """测试保存和加载。"""
        features = [
            SparseFeat(name="category", vocab_size=10),
            DenseFeat(name="value", norm=NormMethod.MINMAX),
        ]
        data = {
            "category": np.array(["a", "b", "c"]),
            "value": np.array([1.0, 2.0, 3.0]),
        }

        transformer = FeatureTransformer(features)
        transformer.fit(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "transformer.json"
            transformer.save(save_path)

            # 加载
            loaded = FeatureTransformer.load(save_path)

            # 验证
            assert loaded.fitted is True
            assert len(loaded.features) == 2
            assert loaded.label_encoders == transformer.label_encoders
            assert loaded.norm_params == transformer.norm_params

    def test_transform_consistency(self):
        """测试转换一致性（fit_transform 和 transform 结果相同）。"""
        features = [
            SparseFeat(name="cat", vocab_size=10),
            DenseFeat(name="num", norm=NormMethod.MINMAX),
        ]
        data = {
            "cat": np.array(["a", "b", "c"]),
            "num": np.array([1.0, 2.0, 3.0]),
        }

        transformer = FeatureTransformer(features)
        result1 = transformer.fit_transform(data)
        result2 = transformer.transform(data)

        np.testing.assert_array_equal(result1["cat"], result2["cat"])
        np.testing.assert_array_almost_equal(result1["num"], result2["num"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
