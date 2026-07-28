"""PU Learning specific evaluation metrics.

Standard classification metrics (precision, recall, F1) are biased in PU scenarios,
because unlabeled samples may contain true positives. This module provides PU-corrected metrics.

References:
- Bekker & Davis (2020): "Learning from Positive and Unlabeled Data: A Survey"
- pulearn.metrics: https://github.com/pulearn/pulearn
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def pu_precision_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_prior: float,
    threshold: float = 0.5,
) -> float:
    """PU-corrected Precision.

    Standard precision is overestimated in PU scenarios (because unlabeled true positives are counted as FP).
    PU correction formula:
        pu_precision = TP / (TP + FP * (1 - π_p))

    Args:
        y_true: Ground truth labels (1=positive, 0=unlabeled).
        y_pred_proba: Prediction probabilities.
        class_prior: Proportion of positives in unlabeled data (π_p).
        threshold: Classification threshold.

    Returns:
        PU-corrected precision value.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # PU correction: part of FP are actually true positives
    corrected_fp = fp * (1 - class_prior)
    denominator = tp + corrected_fp

    if denominator == 0:
        return 0.0
    return float(tp / denominator)


def pu_f1_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_prior: float,
    threshold: float = 0.5,
) -> float:
    """PU-corrected F1 Score.

    F1 = 2 * pu_precision * pu_recall / (pu_precision + pu_recall)

    where pu_recall uses standard recall (positive labels are certain).

    Args:
        y_true: Ground truth labels (1=positive, 0=unlabeled).
        y_pred_proba: Prediction probabilities.
        class_prior: Class prior (π_p).
        threshold: Classification threshold.

    Returns:
        PU-corrected F1 value.
    """
    precision = pu_precision_score(y_true, y_pred_proba, class_prior, threshold)
    recall = _standard_recall(y_true, y_pred_proba, threshold)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def pu_auc_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_prior: float,
) -> float:
    """PU-corrected AUC.

    Computes AUC using PU-corrected ROC curve.
    Correction: discounts false positives in unlabeled samples by class_prior.

    Args:
        y_true: Ground truth labels (1=positive, 0=unlabeled).
        y_pred_proba: Prediction probabilities.
        class_prior: Class prior (π_p).

    Returns:
        PU-corrected AUC value.
    """
    # Weight unlabeled contributions by (1 - class_prior)
    # Only (1 - π_p) of unlabeled samples are true negatives
    thresholds = np.sort(np.unique(y_pred_proba))
    if len(thresholds) > 200:
        # Downsample for performance
        indices = np.linspace(0, len(thresholds) - 1, 200, dtype=int)
        thresholds = thresholds[indices]

    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)

        # TPR: recall on positive samples (labeled positives are certain)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # FPR: false positive rate in unlabeled samples (corrected by class_prior)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        # True negatives count in unlabeled ≈ n_unlabeled * (1 - π_p)
        # But we use weighting directly: FP contribution multiplied by (1 - π_p)
        total_unlabeled = fp + tn
        if total_unlabeled > 0:
            # Standard FPR
            fpr = fp / total_unlabeled
            # PU correction: some FP in unlabeled are actually true positives
            fpr = fpr * (1 - class_prior)
        else:
            fpr = 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Sort and compute AUC
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    sorted_indices = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sorted_indices]
    tpr_arr = tpr_arr[sorted_indices]

    # np.trapezoid (NumPy 2.0+) / np.trapz (NumPy 1.x)
    _trapz = getattr(np, "trapezoid", None)
    if _trapz is None:
        _trapz = np.trapz  # type: ignore[attr-defined]
    auc = float(_trapz(tpr_arr, fpr_arr))
    return max(0.0, min(1.0, auc))


def pu_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_prior: float,
    n_bins: int = 10,
) -> dict[str, Any]:
    """PU calibration analysis.

    Checks the consistency between predicted probabilities and actual positive rates.
    In PU scenarios, the expected positive rate needs to be corrected by class_prior.

    Args:
        y_true: Ground truth labels (1=positive, 0=unlabeled).
        y_pred_proba: Prediction probabilities.
        class_prior: Class prior (π_p).
        n_bins: Number of bins.

    Returns:
        Dictionary containing calibration statistics:
        - bin_centers: Center probability of each bin
        - bin_positive_rates: Actual positive rate per bin
        - bin_expected_rates: Expected positive rate per bin (PU-corrected)
        - calibration_error: Calibration error
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_positive_rates = []
    bin_expected_rates = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        center = (bin_edges[i] + bin_edges[i + 1]) / 2
        positive_rate = y_true[mask].mean()

        # Expected positive rate = standard expectation + contribution of true positives in unlabeled
        # Samples labeled 0 in unlabeled data have probability π_p of being true positives
        n_pos_in_bin = y_true[mask].sum()
        n_unl_in_bin = mask.sum() - n_pos_in_bin
        expected_rate = (n_pos_in_bin + n_unl_in_bin * class_prior) / mask.sum()

        bin_centers.append(center)
        bin_positive_rates.append(positive_rate)
        bin_expected_rates.append(expected_rate)

    if len(bin_centers) == 0:
        return {
            "bin_centers": [],
            "bin_positive_rates": [],
            "bin_expected_rates": [],
            "calibration_error": 0.0,
        }

    bin_centers = np.array(bin_centers)
    bin_positive_rates = np.array(bin_positive_rates)
    bin_expected_rates = np.array(bin_expected_rates)

    # Calibration error: difference between actual and expected positive rates
    calibration_error = float(np.mean(np.abs(bin_positive_rates - bin_expected_rates)))

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_positive_rates": bin_positive_rates.tolist(),
        "bin_expected_rates": bin_expected_rates.tolist(),
        "calibration_error": calibration_error,
    }


def compute_pu_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_prior: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all PU metrics in one call.

    Args:
        y_true: Ground truth labels (1=positive, 0=unlabeled).
        y_pred_proba: Prediction probabilities.
        class_prior: Class prior (π_p).
        threshold: Classification threshold.

    Returns:
        Dictionary containing all PU metrics.
    """
    return {
        "pu_precision": pu_precision_score(
            y_true, y_pred_proba, class_prior, threshold
        ),
        "pu_recall": _standard_recall(y_true, y_pred_proba, threshold),
        "pu_f1": pu_f1_score(y_true, y_pred_proba, class_prior, threshold),
        "pu_auc": pu_auc_score(y_true, y_pred_proba, class_prior),
    }


def _standard_recall(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Standard Recall (positive labels are certain, no PU correction needed).

    Args:
        y_true: Ground truth labels.
        y_pred_proba: Prediction probabilities.
        threshold: Classification threshold.

    Returns:
        Recall value.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))
