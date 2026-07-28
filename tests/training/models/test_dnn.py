"""DNN 模型测试。"""

from __future__ import annotations

import pytest

try:
    import torch

    from tributo.training.features.column_types import (
        DenseFeat,
        NormMethod,
        SparseFeat,
    )
    from tributo.training.models.dnn import DNNModel

    class TestDNNModel:
        """DNNModel 测试（需要 PyTorch）。"""

        def test_basic_creation(self):
            """测试基本创建。"""
            features = [
                SparseFeat(name="cat", vocab_size=10, embedding_dim=8),
                DenseFeat(name="num", norm=NormMethod.NONE),
            ]
            model = DNNModel(features)
            assert model is not None

        def test_forward(self):
            """测试前向传播。"""
            features = [
                SparseFeat(name="cat", vocab_size=10, embedding_dim=8),
                DenseFeat(name="num", norm=NormMethod.NONE),
            ]
            model = DNNModel(features)

            inputs = {
                "cat": torch.tensor([0, 1, 2], dtype=torch.long),
                "num": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            }
            logits = model(inputs)

            assert logits.shape == (3,)
            assert not torch.isnan(logits).any()

        def test_predict_proba(self):
            """测试概率预测。"""
            features = [
                SparseFeat(name="cat", vocab_size=10, embedding_dim=8),
            ]
            model = DNNModel(features)

            inputs = {"cat": torch.tensor([0, 1], dtype=torch.long)}
            probs = model.predict_proba(inputs)

            assert probs.shape == (2,)
            assert torch.all(probs >= 0)
            assert torch.all(probs <= 1)

        def test_custom_hidden_units(self):
            """测试自定义隐藏层。"""
            features = [DenseFeat(name="num")]
            model = DNNModel(features, dnn_hidden_units=[64, 32])

            inputs = {"num": torch.tensor([1.0, 2.0], dtype=torch.float32)}
            logits = model(inputs)
            assert logits.shape == (2,)

        def test_with_dropout(self):
            """测试带 Dropout 的模型。"""
            features = [DenseFeat(name="num")]
            model = DNNModel(features, dnn_dropout=0.5)

            inputs = {"num": torch.tensor([1.0, 2.0], dtype=torch.float32)}

            # 训练模式
            model.train()
            logits_train = model(inputs)

            # 评估模式
            model.eval()
            logits_eval = model(inputs)

            assert logits_train.shape == (2,)
            assert logits_eval.shape == (2,)

        def test_with_batch_norm(self):
            """测试带 BatchNorm 的模型。"""
            features = [DenseFeat(name="num")]
            model = DNNModel(features, use_batch_norm=True)

            # 需要 batch_size > 1 才能使用 BatchNorm
            inputs = {"num": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
            logits = model(inputs)
            assert logits.shape == (3,)

        def test_multiple_features(self):
            """测试多特征输入。"""
            features = [
                SparseFeat(name="cat1", vocab_size=10, embedding_dim=4),
                SparseFeat(name="cat2", vocab_size=20, embedding_dim=8),
                DenseFeat(name="num1"),
                DenseFeat(name="num2"),
            ]
            model = DNNModel(features)

            inputs = {
                "cat1": torch.tensor([0, 1], dtype=torch.long),
                "cat2": torch.tensor([0, 1], dtype=torch.long),
                "num1": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "num2": torch.tensor([3.0, 4.0], dtype=torch.float32),
            }
            logits = model(inputs)
            assert logits.shape == (2,)

except ImportError:
    pass  # PyTorch 未安装时跳过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
