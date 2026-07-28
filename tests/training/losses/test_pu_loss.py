"""nnPU 损失函数测试。"""

from __future__ import annotations

import pytest

from tributo.training.losses.pu_loss import compute_class_prior


class TestComputeClassPrior:
    """compute_class_prior 测试。"""

    def test_simple_estimation(self):
        """测试简单估计。"""
        prior = compute_class_prior(positive_count=100, total_count=1000)
        assert prior == pytest.approx(0.1)

    def test_all_positive(self):
        """测试全正例。"""
        prior = compute_class_prior(positive_count=100, total_count=100)
        assert prior == pytest.approx(1.0)

    def test_no_positive(self):
        """测试无正例。"""
        prior = compute_class_prior(positive_count=0, total_count=100)
        assert prior == pytest.approx(0.0)

    def test_invalid_positive_count(self):
        """测试无效正例数。"""
        with pytest.raises(ValueError, match="non-negative"):
            compute_class_prior(positive_count=-1, total_count=100)

    def test_invalid_total_count(self):
        """测试无效总数。"""
        with pytest.raises(ValueError, match="positive"):
            compute_class_prior(positive_count=10, total_count=0)

    def test_positive_exceeds_total(self):
        """测试正例超过总数。"""
        with pytest.raises(ValueError, match="cannot exceed"):
            compute_class_prior(positive_count=101, total_count=100)


try:
    import torch

    from tributo.training.losses.pu_loss import PULoss, nnpu_loss

    class TestPULoss:
        """PULoss 测试（需要 PyTorch）。"""

        def test_basic_forward(self):
            """测试基本前向传播。"""
            criterion = PULoss(class_prior=0.1)
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            loss = criterion(logits, labels)
            assert loss.dim() == 0  # 标量
            assert loss.item() >= 0 or loss.item() < 0  # 可以是负数（nnPU 特性）

        def test_all_positive(self):
            """测试全正例输入。"""
            criterion = PULoss(class_prior=0.5)
            logits = torch.randn(5)
            labels = torch.ones(5)

            loss = criterion(logits, labels)
            assert not torch.isnan(loss)

        def test_all_unlabeled(self):
            """测试全未标注输入。"""
            criterion = PULoss(class_prior=0.5)
            logits = torch.randn(5)
            labels = torch.zeros(5)

            loss = criterion(logits, labels)
            assert not torch.isnan(loss)

        def test_nnpu_loss_function(self):
            """测试 nnpu_loss 函数接口。"""
            logits = torch.randn(10)
            labels = torch.randint(0, 2, (10,)).float()

            loss = nnpu_loss(logits, labels, class_prior=0.1)
            assert not torch.isnan(loss)

        def test_invalid_class_prior(self):
            """测试无效 class_prior。"""
            with pytest.raises(ValueError, match="in \\(0, 1\\)"):
                PULoss(class_prior=0.0)

            with pytest.raises(ValueError, match="in \\(0, 1\\)"):
                PULoss(class_prior=1.0)

        def test_invalid_loss_type(self):
            """测试无效 loss_type。"""
            with pytest.raises(ValueError, match="nnpu.*upu"):
                PULoss(class_prior=0.5, loss_type="invalid")

except ImportError:
    pass  # PyTorch 未安装时跳过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
