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
    import torch.nn.functional as F

    from tributo.training.losses.pu_loss import PULoss, nnpu_loss

    class TestPULoss:
        """PULoss 测试（需要 PyTorch）。"""

        def test_basic_forward(self):
            """测试基本前向传播。"""
            criterion = PULoss(class_prior=0.1)
            logits = torch.randn(10)
            labels = torch.tensor([1.0, 0.0] * 5)

            loss = criterion(logits, labels)
            assert loss.dim() == 0  # 标量
            assert torch.isfinite(loss)

        def test_upu_matches_empirical_risk_definition(self) -> None:
            """uPU 必须包含正样本在负类损失下的校正项。"""
            logits = torch.tensor([2.0, -1.0, 0.5, -0.5])
            labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
            prior = 0.4

            actual = PULoss(class_prior=prior, loss_type="upu")(logits, labels)
            positive_loss = F.softplus(-logits[:2]).mean()
            positive_as_negative = F.softplus(logits[:2]).mean()
            unlabeled_negative = F.softplus(logits[2:]).mean()
            expected = (
                prior * positive_loss
                + unlabeled_negative
                - prior * positive_as_negative
            )

            assert actual == pytest.approx(expected)

        def test_nnpu_applies_negative_risk_correction(self) -> None:
            """校正分支保留 Algorithm 1 的代理值和梯度方向。"""
            logits = torch.tensor(
                [10.0, 10.0, -10.0, -10.0],
                requires_grad=True,
            )
            labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
            prior = 0.5
            gamma = 0.5

            actual = PULoss(
                class_prior=prior,
                gamma=gamma,
                loss_type="nnpu",
            )(logits, labels)
            negative_risk = (
                F.softplus(logits[2:]).mean() - prior * F.softplus(logits[:2]).mean()
            )

            torch.testing.assert_close(
                actual.detach(),
                (-gamma * negative_risk).detach(),
            )
            expected_gradient = torch.autograd.grad(
                -gamma * negative_risk,
                logits,
            )[0]
            actual.backward()
            assert logits.grad is not None
            assert torch.allclose(logits.grad, expected_gradient)

        def test_nnpu_empirical_risk_keeps_positive_term_in_correction_region(
            self,
        ) -> None:
            logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
            labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
            prior = 0.5
            criterion = PULoss(class_prior=prior, gamma=0.5, loss_type="nnpu")

            actual = criterion.empirical_risk(logits, labels)
            expected = prior * F.softplus(-logits[:2]).mean()

            assert actual == pytest.approx(expected)

        def test_split_accumulator_combines_single_group_batches(self) -> None:
            criterion = PULoss(class_prior=0.4, loss_type="nnpu")
            accumulator = criterion.new_risk_accumulator()
            positive_logits = torch.tensor([2.0, -1.0])
            unlabeled_logits = torch.tensor([0.5, -0.5, 1.0])

            accumulator.update(positive_logits, torch.ones(2))
            accumulator.update(unlabeled_logits, torch.zeros(3))

            combined_logits = torch.cat((positive_logits, unlabeled_logits))
            combined_labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
            assert accumulator.value() == pytest.approx(
                criterion.empirical_risk(combined_logits, combined_labels)
            )

        def test_extreme_logits_remain_finite(self) -> None:
            criterion = PULoss(class_prior=0.5)
            logits = torch.tensor([1000.0, -1000.0, 1000.0, -1000.0])
            labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

            assert torch.isfinite(criterion(logits, labels))

        def test_all_positive(self):
            """仅正例的训练 batch 违反配对采样契约。"""
            criterion = PULoss(class_prior=0.5)
            logits = torch.randn(5)
            labels = torch.ones(5)

            with pytest.raises(ValueError, match="both positive and unlabeled"):
                criterion(logits, labels)

        def test_all_unlabeled(self):
            """仅未标注样本的训练 batch 违反配对采样契约。"""
            criterion = PULoss(class_prior=0.5)
            logits = torch.randn(5)
            labels = torch.zeros(5)

            with pytest.raises(ValueError, match="both positive and unlabeled"):
                criterion(logits, labels)

        def test_non_binary_labels_are_rejected(self) -> None:
            criterion = PULoss(class_prior=0.5)

            with pytest.raises(ValueError, match="only 1.*or 0"):
                criterion(torch.randn(3), torch.tensor([1.0, 0.0, -1.0]))

        def test_nnpu_loss_function(self):
            """测试 nnpu_loss 函数接口。"""
            logits = torch.randn(10)
            labels = torch.tensor([1.0, 0.0] * 5)

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

        @pytest.mark.parametrize(
            ("kwargs", "message"),
            (
                ({"beta": -0.1}, "beta must be non-negative"),
                ({"gamma": -0.1}, "gamma must be in"),
                ({"gamma": 1.1}, "gamma must be in"),
            ),
        )
        def test_invalid_correction_parameters(
            self,
            kwargs: dict[str, float],
            message: str,
        ) -> None:
            with pytest.raises(ValueError, match=message):
                PULoss(class_prior=0.5, **kwargs)

except ImportError:
    pass  # PyTorch 未安装时跳过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
