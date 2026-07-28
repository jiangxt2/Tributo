"""PyTorch Dataset 适配器测试。"""

from __future__ import annotations

import numpy as np
import pytest

from tributo.training.features.column_types import (
    DenseFeat,
    NormMethod,
    SparseFeat,
)
from tributo.training.features.dataset import IdentityDataset


class TestIdentityDataset:
    """IdentityDataset 测试。"""

    def test_basic_creation(self):
        """测试基本创建。"""
        features = [
            SparseFeat(name="cat", vocab_size=10),
            DenseFeat(name="num", norm=NormMethod.NONE),
        ]
        data = {
            "cat": np.array([0, 1, 2, 3]),
            "num": np.array([1.0, 2.0, 3.0, 4.0]),
        }
        labels = np.array([0, 1, 0, 1])

        dataset = IdentityDataset(data, labels, features)
        assert len(dataset) == 4

    def test_getitem(self):
        """测试获取单个样本。"""
        features = [
            SparseFeat(name="cat", vocab_size=10),
            DenseFeat(name="num", norm=NormMethod.NONE),
        ]
        data = {
            "cat": np.array([0, 1, 2]),
            "num": np.array([1.0, 2.0, 3.0]),
        }
        labels = np.array([0, 1, 0])

        dataset = IdentityDataset(data, labels, features)
        sample = dataset[0]

        assert "cat" in sample
        assert "num" in sample
        assert "label" in sample
        assert sample["cat"] == 0
        assert sample["num"] == 1.0
        assert sample["label"] == 0

    def test_to_torch_dataset(self):
        """测试转换为 PyTorch Dataset。"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        features = [
            SparseFeat(name="cat", vocab_size=10),
            DenseFeat(name="num", norm=NormMethod.NONE),
        ]
        data = {
            "cat": np.array([0, 1, 2]),
            "num": np.array([1.0, 2.0, 3.0]),
        }
        labels = np.array([0, 1, 0])

        dataset = IdentityDataset(data, labels, features)
        torch_dataset = dataset.to_torch_dataset()

        assert len(torch_dataset) == 3

        sample = torch_dataset[0]
        assert isinstance(sample["cat"], torch.Tensor)
        assert isinstance(sample["num"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)

    def test_sparse_dense_features_list(self):
        """测试特征列表属性。"""
        features = [
            SparseFeat(name="cat1", vocab_size=10),
            SparseFeat(name="cat2", vocab_size=20),
            DenseFeat(name="num1"),
            DenseFeat(name="num2"),
        ]
        data = {
            "cat1": np.array([0]),
            "cat2": np.array([0]),
            "num1": np.array([1.0]),
            "num2": np.array([2.0]),
        }
        labels = np.array([0])

        dataset = IdentityDataset(data, labels, features)
        assert dataset.sparse_features == ["cat1", "cat2"]
        assert dataset.dense_features == ["num1", "num2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
