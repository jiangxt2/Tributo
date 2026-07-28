"""特征列类型定义测试。"""

from __future__ import annotations

import pytest

from tributo.training.features.column_types import (
    DenseFeat,
    NormMethod,
    SparseFeat,
    get_dense_features,
    get_feature_names,
    get_sparse_features,
)


class TestSparseFeat:
    """SparseFeat 测试。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        feat = SparseFeat(name="category", vocab_size=100, embedding_dim=8)
        assert feat.name == "category"
        assert feat.vocab_size == 100
        assert feat.embedding_dim == 8
        assert feat.use_hash is False
        assert feat.hash_bucket_size == 100000

    def test_hash_encoding(self):
        """测试 Hash Encoding 配置。"""
        feat = SparseFeat(
            name="user_id",
            vocab_size=1000000,
            embedding_dim=16,
            use_hash=True,
            hash_bucket_size=100000,
        )
        assert feat.use_hash is True
        assert feat.hash_bucket_size == 100000

    def test_invalid_vocab_size(self):
        """测试无效 vocab_size。"""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            SparseFeat(name="test", vocab_size=0)

    def test_invalid_embedding_dim(self):
        """测试无效 embedding_dim。"""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            SparseFeat(name="test", vocab_size=100, embedding_dim=-1)

    def test_invalid_hash_bucket_size(self):
        """测试无效 hash_bucket_size。"""
        with pytest.raises(ValueError, match="hash_bucket_size must be positive"):
            SparseFeat(name="test", vocab_size=100, use_hash=True, hash_bucket_size=0)


class TestDenseFeat:
    """DenseFeat 测试。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        feat = DenseFeat(name="age", dimension=1, norm=NormMethod.NONE)
        assert feat.name == "age"
        assert feat.dimension == 1
        assert feat.norm == NormMethod.NONE

    def test_norm_methods(self):
        """测试各种归一化方式。"""
        feat_minmax = DenseFeat(name="x", norm=NormMethod.MINMAX)
        assert feat_minmax.norm == NormMethod.MINMAX

        feat_standard = DenseFeat(name="x", norm=NormMethod.STANDARD)
        assert feat_standard.norm == NormMethod.STANDARD

        feat_log = DenseFeat(name="x", norm=NormMethod.LOG)
        assert feat_log.norm == NormMethod.LOG

    def test_multi_dimension(self):
        """测试多维特征。"""
        feat = DenseFeat(name="embedding", dimension=128)
        assert feat.dimension == 128

    def test_invalid_dimension(self):
        """测试无效 dimension。"""
        with pytest.raises(ValueError, match="dimension must be positive"):
            DenseFeat(name="test", dimension=0)


class TestUtilityFunctions:
    """工具函数测试。"""

    def test_get_feature_names(self):
        """测试获取特征名称。"""
        features = [
            SparseFeat(name="cat1", vocab_size=10),
            DenseFeat(name="num1"),
            SparseFeat(name="cat2", vocab_size=20),
        ]
        names = get_feature_names(features)
        assert names == ["cat1", "num1", "cat2"]

    def test_get_sparse_features(self):
        """测试筛选 Sparse 特征。"""
        features = [
            SparseFeat(name="cat1", vocab_size=10),
            DenseFeat(name="num1"),
            SparseFeat(name="cat2", vocab_size=20),
        ]
        sparse = get_sparse_features(features)
        assert len(sparse) == 2
        assert all(isinstance(f, SparseFeat) for f in sparse)

    def test_get_dense_features(self):
        """测试筛选 Dense 特征。"""
        features = [
            SparseFeat(name="cat1", vocab_size=10),
            DenseFeat(name="num1"),
            SparseFeat(name="cat2", vocab_size=20),
        ]
        dense = get_dense_features(features)
        assert len(dense) == 1
        assert all(isinstance(f, DenseFeat) for f in dense)

    def test_empty_features(self):
        """测试空特征列表。"""
        assert get_feature_names([]) == []
        assert get_sparse_features([]) == []
        assert get_dense_features([]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
