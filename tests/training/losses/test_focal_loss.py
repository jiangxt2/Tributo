"""Focal Loss 测试。"""

from __future__ import annotations

import pytest

try:
    import torch

    from tributo.training.losses.focal_loss import FocalLoss, focal_loss

    class TestFocalLoss:
        """FocalLoss 测试（需要 PyTorch）。"""

        def test_basic_forward(self):
            """测试基本前向传播。"""
            criterion = FocalLoss()
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            loss = criterion(logits, labels)
            assert loss.dim() == 0
            assert loss.item() >= 0

        def test_balanced_labels(self):
            """测试均衡标签。"""
            criterion = FocalLoss()
            logits = torch.randn(100)
            labels = torch.cat([torch.ones(50), torch.zeros(50)])

            loss = criterion(logits, labels)
            assert not torch.isnan(loss)
            assert loss.item() > 0

        def test_imbalanced_labels(self):
            """测试不均衡标签。"""
            criterion = FocalLoss()
            logits = torch.randn(100)
            # 90% 负例，10% 正例
            labels = torch.cat([torch.ones(10), torch.zeros(90)])

            loss = criterion(logits, labels)
            assert not torch.isnan(loss)
            assert loss.item() > 0

        def test_alpha_parameter(self):
            """测试 alpha 参数。"""
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            # 不同的 alpha 应该产生有效的损失
            loss1 = FocalLoss(alpha=0.25)(logits, labels)
            loss2 = FocalLoss(alpha=0.75)(logits, labels)

            # 两个损失都应该是有效的非负值
            assert not torch.isnan(loss1)
            assert not torch.isnan(loss2)
            assert loss1.item() >= 0
            assert loss2.item() >= 0

        def test_gamma_parameter(self):
            """测试 gamma 参数。"""
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            # 不同的 gamma 应该产生有效的损失
            loss1 = FocalLoss(gamma=0.0)(logits, labels)  # gamma=0 退化为加权 BCE
            loss2 = FocalLoss(gamma=2.0)(logits, labels)

            # 两个损失都应该是有效的非负值
            assert not torch.isnan(loss1)
            assert not torch.isnan(loss2)
            assert loss1.item() >= 0
            assert loss2.item() >= 0

        def test_reduction_modes(self):
            """测试不同的 reduction 模式。"""
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            loss_mean = FocalLoss(reduction="mean")(logits, labels)
            loss_sum = FocalLoss(reduction="sum")(logits, labels)
            loss_none = FocalLoss(reduction="none")(logits, labels)

            assert loss_mean.dim() == 0
            assert loss_sum.dim() == 0
            assert loss_none.shape == (10,)

        def test_focal_loss_function(self):
            """测试 focal_loss 函数接口。"""
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            loss = focal_loss(logits, labels)
            assert not torch.isnan(loss)
            assert loss.item() >= 0

except ImportError:
    pass  # PyTorch 未安装时跳过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
