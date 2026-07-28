"""PU 评估指标测试。"""

from __future__ import annotations

import numpy as np
import pytest

from tributo.training.pu_metrics import (
    compute_pu_metrics,
    pu_auc_score,
    pu_calibration,
    pu_f1_score,
    pu_precision_score,
)


def _make_test_data(
    n_pos: int = 100,
    n_unl: int = 400,
    class_prior: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """生成 PU 测试数据。

    正例分数偏高（0.6-1.0），未标注分数均匀（0.0-1.0）。
    """
    rng = np.random.RandomState(seed)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_unl)])
    y_scores = np.concatenate(
        [
            rng.uniform(0.6, 1.0, n_pos),
            rng.uniform(0.0, 1.0, n_unl),
        ]
    )
    return y_true, y_scores, class_prior


class TestPUPrecisionScore:
    """pu_precision_score 测试。"""

    def test_perfect_classifier(self):
        """完美分类器，precision = 1.0。"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        prec = pu_precision_score(y_true, y_scores, class_prior=0.5, threshold=0.5)
        assert prec == pytest.approx(1.0)

    def test_worst_classifier(self):
        """最差分类器（全部误判），precision 接近 0。"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        prec = pu_precision_score(y_true, y_scores, class_prior=0.5, threshold=0.5)
        assert prec < 0.3

    def test_pu_correction_reduces_fp(self):
        """PU 修正应比标准 precision 更高（FP 被折扣）。"""
        y_true, y_scores, class_prior = _make_test_data()
        pu_prec = pu_precision_score(y_true, y_scores, class_prior, threshold=0.5)
        # 标准 precision（class_prior=0 时等价于标准）
        std_prec = pu_precision_score(y_true, y_scores, class_prior=0.0, threshold=0.5)
        # PU 修正后 precision 应更高（FP 被折扣）
        assert pu_prec >= std_prec - 0.01  # 允许浮点误差

    def test_no_predictions(self):
        """没有预测为正例时，precision = 0.0。"""
        y_true = np.array([1, 0])
        y_scores = np.array([0.1, 0.1])
        prec = pu_precision_score(y_true, y_scores, class_prior=0.5, threshold=0.5)
        assert prec == 0.0


class TestPUF1Score:
    """pu_f1_score 测试。"""

    def test_perfect_classifier(self):
        """完美分类器，F1 = 1.0。"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        f1 = pu_f1_score(y_true, y_scores, class_prior=0.5, threshold=0.5)
        assert f1 == pytest.approx(1.0)

    def test_range(self):
        """F1 应在 [0, 1] 范围内。"""
        y_true, y_scores, class_prior = _make_test_data()
        f1 = pu_f1_score(y_true, y_scores, class_prior, threshold=0.5)
        assert 0.0 <= f1 <= 1.0

    def test_zero_precision_and_recall(self):
        """precision 和 recall 都为 0 时，F1 = 0.0。"""
        y_true = np.array([1, 0])
        y_scores = np.array([0.1, 0.1])
        f1 = pu_f1_score(y_true, y_scores, class_prior=0.5, threshold=0.5)
        assert f1 == 0.0


class TestPUAucScore:
    """pu_auc_score 测试。"""

    def test_good_classifier(self):
        """好分类器的 AUC 应 > 0.5。"""
        y_true, y_scores, class_prior = _make_test_data()
        auc = pu_auc_score(y_true, y_scores, class_prior)
        assert auc > 0.5

    def test_random_classifier(self):
        """随机分类器的 AUC 应 ≈ 0.5。"""
        rng = np.random.RandomState(42)
        y_true = np.concatenate([np.ones(200), np.zeros(800)])
        y_scores = rng.uniform(0, 1, 1000)
        auc = pu_auc_score(y_true, y_scores, class_prior=0.2)
        assert 0.3 < auc < 0.7  # 随机分类器应在 0.5 附近

    def test_range(self):
        """AUC 应在 [0, 1] 范围内。"""
        y_true, y_scores, class_prior = _make_test_data()
        auc = pu_auc_score(y_true, y_scores, class_prior)
        assert 0.0 <= auc <= 1.0


class TestPUCalibration:
    """pu_calibration 测试。"""

    def test_returns_expected_keys(self):
        """应返回所有预期的键。"""
        y_true, y_scores, class_prior = _make_test_data()
        result = pu_calibration(y_true, y_scores, class_prior, n_bins=5)
        assert "bin_centers" in result
        assert "bin_positive_rates" in result
        assert "bin_expected_rates" in result
        assert "calibration_error" in result

    def test_calibration_error_range(self):
        """校准误差应在 [0, 1] 范围内。"""
        y_true, y_scores, class_prior = _make_test_data()
        result = pu_calibration(y_true, y_scores, class_prior)
        assert 0.0 <= result["calibration_error"] <= 1.0

    def test_empty_data(self):
        """空数据应返回零值。"""
        result = pu_calibration(
            np.array([]),
            np.array([]),
            class_prior=0.5,
            n_bins=5,
        )
        assert result["calibration_error"] == 0.0


class TestComputePUMetrics:
    """compute_pumetrics 一次性计算测试。"""

    def test_returns_all_metrics(self):
        """应返回所有 PU 指标。"""
        y_true, y_scores, class_prior = _make_test_data()
        metrics = compute_pu_metrics(y_true, y_scores, class_prior)
        assert "pu_precision" in metrics
        assert "pu_recall" in metrics
        assert "pu_f1" in metrics
        assert "pu_auc" in metrics

    def test_metrics_range(self):
        """所有指标应在 [0, 1] 范围内。"""
        y_true, y_scores, class_prior = _make_test_data()
        metrics = compute_pu_metrics(y_true, y_scores, class_prior)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
