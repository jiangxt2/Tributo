"""Class prior 估计模块测试。"""

from __future__ import annotations

import numpy as np
import pytest

from tributo.training.priors import (
    em_prior,
    estimate_class_prior,
    histogram_match_prior,
    label_frequency_prior,
)


class TestLabelFrequencyPrior:
    """label_frequency_prior 测试。"""

    def test_basic(self):
        """正例 150 / 总数 1000 = 0.15。"""
        assert label_frequency_prior(150, 1000) == pytest.approx(0.15)

    def test_all_positive(self):
        """全部是正例，prior = 1.0。"""
        assert label_frequency_prior(100, 100) == pytest.approx(1.0)

    def test_no_positive(self):
        """没有正例，prior = 0.0。"""
        assert label_frequency_prior(0, 100) == pytest.approx(0.0)

    def test_invalid_negative_count(self):
        """负例数不能为负。"""
        with pytest.raises(ValueError, match="non-negative"):
            label_frequency_prior(-1, 100)

    def test_invalid_zero_total(self):
        """总数不能为零。"""
        with pytest.raises(ValueError, match="positive"):
            label_frequency_prior(0, 0)

    def test_invalid_exceeds_total(self):
        """正例数不能超过总数。"""
        with pytest.raises(ValueError, match="exceed"):
            label_frequency_prior(101, 100)


class TestHistogramMatchPrior:
    """histogram_match_prior 测试。"""

    def test_identical_distributions(self):
        """正例和未标注分布完全一致，prior ≈ 1.0。"""
        rng = np.random.RandomState(42)
        scores = rng.uniform(0, 1, 500)
        prior = histogram_match_prior(scores, scores, n_bins=20)
        assert prior == pytest.approx(1.0, abs=0.1)

    def test_disjoint_distributions(self):
        """正例和未标注分布完全不重叠，prior ≈ 0.0。"""
        pos_scores = np.linspace(0.6, 1.0, 200)
        unl_scores = np.linspace(0.0, 0.4, 200)
        prior = histogram_match_prior(pos_scores, unl_scores, n_bins=20)
        assert prior < 0.2

    def test_partial_overlap(self):
        """部分重叠，prior 在 0 和 1 之间。"""
        rng = np.random.RandomState(42)
        pos_scores = rng.uniform(0.5, 1.0, 300)
        unl_scores = rng.uniform(0.0, 1.0, 700)
        prior = histogram_match_prior(pos_scores, unl_scores, n_bins=20)
        assert 0.0 < prior < 1.0

    def test_empty_positive_raises(self):
        """空正例数组应抛异常。"""
        with pytest.raises(ValueError, match="non-empty"):
            histogram_match_prior(np.array([]), np.array([0.5]))


class TestEMPrior:
    """em_prior 测试。"""

    def test_basic_convergence(self):
        """EM 应收敛到合理值。"""
        rng = np.random.RandomState(42)
        pos_scores = rng.uniform(0.6, 1.0, 200)
        unl_scores = rng.uniform(0.0, 1.0, 800)
        prior = em_prior(pos_scores, unl_scores, max_iter=50)
        assert 0.0 < prior < 1.0

    def test_custom_init_prior(self):
        """自定义初始 prior 应影响结果。"""
        rng = np.random.RandomState(42)
        pos_scores = rng.uniform(0.5, 1.0, 100)
        unl_scores = rng.uniform(0.0, 1.0, 400)
        prior = em_prior(pos_scores, unl_scores, init_prior=0.5)
        assert 0.0 < prior < 1.0

    def test_empty_positive_raises(self):
        """空正例数组应抛异常。"""
        with pytest.raises(ValueError, match="non-empty"):
            em_prior(np.array([]), np.array([0.5]))


class TestEstimateClassPrior:
    """estimate_class_prior 统一入口测试。"""

    def test_label_frequency_method(self):
        """label_frequency 方法。"""
        prior = estimate_class_prior(150, 1000, method="label_frequency")
        assert prior == pytest.approx(0.15)

    def test_histogram_match_method(self):
        """histogram_match 方法。"""
        rng = np.random.RandomState(42)
        prior = estimate_class_prior(
            150,
            1000,
            method="histogram_match",
            positive_scores=rng.uniform(0.5, 1.0, 150),
            unlabeled_scores=rng.uniform(0.0, 1.0, 850),
        )
        assert 0.0 < prior < 1.0

    def test_em_method(self):
        """em 方法。"""
        rng = np.random.RandomState(42)
        prior = estimate_class_prior(
            150,
            1000,
            method="em",
            positive_scores=rng.uniform(0.5, 1.0, 150),
            unlabeled_scores=rng.uniform(0.0, 1.0, 850),
        )
        assert 0.0 < prior < 1.0

    def test_unknown_method_raises(self):
        """未知方法应抛异常。"""
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_class_prior(150, 1000, method="invalid")

    def test_histogram_match_requires_scores(self):
        """histogram_match 需要分数数组。"""
        with pytest.raises(ValueError, match="requires"):
            estimate_class_prior(150, 1000, method="histogram_match")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
