"""Class prior (π_p) estimation module.

Used in Positive-Unlabeled Learning scenarios to estimate the proportion of positive examples in unlabeled data.

Supports three estimation methods:
- label_frequency: Naive lower bound estimate (positive count / total)
- histogram_match: Histogram matching method
- em: EM iterative refinement (SCAR assumption)

References:
- Bekker & Davis (2020): "Learning from Positive and Unlabeled Data: A Survey"
- Ramaswamy et al. (2016): "A Statistical Approach to PU Learning"
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

PriorMethod = Literal["label_frequency", "histogram_match", "em"]


def label_frequency_prior(positive_count: int, total_count: int) -> float:
    """Naive label frequency estimation.

    π_p ≥ P(s=1) = positive_count / total. This is a lower bound for the class prior.

    Args:
        positive_count: Number of positive (labeled) examples.
        total_count: Total number of samples.

    Returns:
        Estimated class prior value.

    Raises:
        ValueError: Invalid parameters.
    """
    _validate_counts(positive_count, total_count)
    return positive_count / total_count


def histogram_match_prior(
    positive_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    n_bins: int = 50,
) -> float:
    """Estimate class prior via histogram matching.

    Compares the histograms of prediction scores between positive and unlabeled samples
    to estimate the proportion of positives.
    Principle: the score distribution of positives in unlabeled data should match that of known positives.

    Args:
        positive_scores: Prediction probabilities or scores for positives, shape (n_positive,).
        unlabeled_scores: Prediction probabilities or scores for unlabeled samples, shape (n_unlabeled,).
        n_bins: Number of histogram bins.

    Returns:
        Estimated class prior value.
    """
    if len(positive_scores) == 0:
        raise ValueError("positive_scores must be non-empty")
    if len(unlabeled_scores) == 0:
        raise ValueError("unlabeled_scores must be non-empty")

    # Unified bin range
    all_scores = np.concatenate([positive_scores, unlabeled_scores])
    bin_edges = np.linspace(all_scores.min(), all_scores.max(), n_bins + 1)

    pos_hist, _ = np.histogram(positive_scores, bins=bin_edges, density=True)
    unl_hist, _ = np.histogram(unlabeled_scores, bins=bin_edges, density=True)

    # Normalize
    pos_hist = pos_hist / (pos_hist.sum() + 1e-10)
    unl_hist = unl_hist / (unl_hist.sum() + 1e-10)

    # Estimate: portion of unlabeled distribution matching positive distribution
    ratio = np.minimum(pos_hist, unl_hist).sum()
    prior = min(max(ratio, 0.0), 1.0)

    logger.info("Histogram match prior: %.4f (bins=%d)", prior, n_bins)
    return prior


def em_prior(
    positive_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    init_prior: float | None = None,
) -> float:
    """Estimate class prior via EM algorithm (SCAR assumption).

    E-step: Estimate the probability of each unlabeled sample belonging to the positive class
    M-step: Update the class prior

    Args:
        positive_scores: Prediction probabilities for positives, shape (n_positive,).
        unlabeled_scores: Prediction probabilities for unlabeled samples, shape (n_unlabeled,).
        max_iter: Maximum number of iterations.
        tol: Convergence threshold.
        init_prior: Initial prior value; uses label_frequency if None.

    Returns:
        Estimated class prior value.
    """
    if len(positive_scores) == 0:
        raise ValueError("positive_scores must be non-empty")
    if len(unlabeled_scores) == 0:
        raise ValueError("unlabeled_scores must be non-empty")

    n_pos = len(positive_scores)
    n_unl = len(unlabeled_scores)
    total = n_pos + n_unl

    # Initial prior
    if init_prior is not None:
        prior = init_prior
    else:
        prior = n_pos / total

    for iteration in range(max_iter):
        # E-step: for unlabeled samples, estimate P(y=1|x, s=0)
        # Using Bayes rule: P(y=1|x) = π_p * P(s=1|x,y=1) / P(s=1|x)
        # Simplified: assume P(s=1|x,y=1) ≈ mean of positive scores
        pos_mean = positive_scores.mean()
        unl_mean = unlabeled_scores.mean()

        # Expected proportion of positives in unlabeled data
        # P(y=1|s=0,x) ≈ π_p * P(x|y=1) / P(x|s=0)
        # Simplified estimate
        expected_pos_in_unl = (
            prior * pos_mean / (prior * pos_mean + (1 - prior) * unl_mean + 1e-10)
        )
        expected_pos_in_unl = np.clip(expected_pos_in_unl, 0, 1)

        # M-step: update prior
        new_prior = (n_pos + n_unl * expected_pos_in_unl) / total

        if abs(new_prior - prior) < tol:
            logger.info(
                "EM converged at iteration %d: prior=%.4f", iteration + 1, new_prior
            )
            return min(max(new_prior, 0.0), 1.0)

        prior = new_prior

    logger.warning(
        "EM did not converge after %d iterations, prior=%.4f", max_iter, prior
    )
    return min(max(prior, 0.0), 1.0)


@PublicAPI(stability="beta")
def estimate_class_prior(
    positive_count: int,
    total_count: int,
    method: PriorMethod = "label_frequency",
    positive_scores: np.ndarray | None = None,
    unlabeled_scores: np.ndarray | None = None,
    **kwargs: float,
) -> float:
    """Unified entry point for class prior estimation.

    Args:
        positive_count: Number of positive examples.
        total_count: Total number of samples.
        method: Estimation method.
        positive_scores: Prediction probabilities for positives (required for histogram_match and em).
        unlabeled_scores: Prediction probabilities for unlabeled samples (required for histogram_match and em).
        **kwargs: Method-specific parameters (n_bins, max_iter, tol, etc.).

    Returns:
        Estimated class prior value.
    """
    if method == "label_frequency":
        return label_frequency_prior(positive_count, total_count)
    elif method == "histogram_match":
        if positive_scores is None or unlabeled_scores is None:
            raise ValueError(
                "histogram_match requires positive_scores and unlabeled_scores"
            )
        return histogram_match_prior(
            positive_scores,
            unlabeled_scores,
            n_bins=int(kwargs.get("n_bins", 50)),
        )
    elif method == "em":
        if positive_scores is None or unlabeled_scores is None:
            raise ValueError("em requires positive_scores and unlabeled_scores")
        return em_prior(
            positive_scores,
            unlabeled_scores,
            max_iter=int(kwargs.get("max_iter", 100)),
            tol=float(kwargs.get("tol", 1e-6)),
            init_prior=kwargs.get("init_prior"),
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _validate_counts(positive_count: int, total_count: int) -> None:
    """Validate count parameters."""
    if positive_count < 0:
        raise ValueError(f"positive_count must be non-negative, got {positive_count}")
    if total_count <= 0:
        raise ValueError(f"total_count must be positive, got {total_count}")
    if positive_count > total_count:
        raise ValueError(
            f"positive_count ({positive_count}) cannot exceed total_count ({total_count})"
        )
