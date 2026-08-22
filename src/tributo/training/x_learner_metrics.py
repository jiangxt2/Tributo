"""Deterministic X-Learner combination, quadrant, and uplift metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_QUADRANTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")


@dataclass(frozen=True)
class UpliftEvaluation:
    """Bounded uplift evaluation result on an independent holdout."""

    ate: float
    auuc: float
    qini: float
    qini_raw: float
    coverage: tuple[float, ...]
    uplift_curve: tuple[float, ...]
    qini_curve: tuple[float, ...]


def combine_cate(
    tau0: Any,
    tau1: Any,
    propensity: Any,
    *,
    clip: tuple[float, float] = (1e-3, 1.0 - 1e-3),
) -> np.ndarray:
    """Apply the X-Learner propensity-weighted CATE formula."""
    low, high = clip
    if not (0.0 < low < high < 1.0):
        raise ValueError("propensity clip must satisfy 0 < low < high < 1")
    left = np.asarray(tau0, dtype=np.float64).reshape(-1)
    right = np.asarray(tau1, dtype=np.float64).reshape(-1)
    score = np.asarray(propensity, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.shape != score.shape or not left.size:
        raise ValueError("tau0, tau1, and propensity must have one shared shape")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.isfinite(score).all()
    ):
        raise ValueError("X-Learner predictions must be finite")
    weight = np.clip(score, low, high)
    return weight * left + (1.0 - weight) * right


def classify_quadrants(
    mu0: Any,
    mu1: Any,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Classify potential-response probabilities into four uplift quadrants."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("response threshold must be finite and inside [0, 1]")
    control = np.asarray(mu0, dtype=np.float64).reshape(-1)
    treated = np.asarray(mu1, dtype=np.float64).reshape(-1)
    if control.shape != treated.shape or not control.size:
        raise ValueError("mu0 and mu1 must have one shared non-empty shape")
    if not np.isfinite(control).all() or not np.isfinite(treated).all():
        raise ValueError("potential-response predictions must be finite")
    control_positive = control >= threshold
    treated_positive = treated >= threshold
    result = np.empty(control.shape, dtype=object)
    result[~control_positive & treated_positive] = _QUADRANTS[0]
    result[control_positive & treated_positive] = _QUADRANTS[1]
    result[~control_positive & ~treated_positive] = _QUADRANTS[2]
    result[control_positive & ~treated_positive] = _QUADRANTS[3]
    return result.astype(str)


def evaluate_uplift(
    treatment: Any,
    outcome: Any,
    cate: Any,
    identity: Any,
    *,
    curve_points: int = 100,
) -> UpliftEvaluation:
    """Compute deterministic holdout uplift, Qini, AUUC, and model-mean ATE."""
    if (
        not isinstance(curve_points, int)
        or isinstance(curve_points, bool)
        or curve_points < 2
    ):
        raise ValueError("curve_points must be an integer of at least two")
    t = np.asarray(treatment).reshape(-1)
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    score = np.asarray(cate, dtype=np.float64).reshape(-1)
    ids = np.asarray(identity).reshape(-1)
    if not t.size or not (t.shape == y.shape == score.shape == ids.shape):
        raise ValueError("uplift inputs must have one shared non-empty shape")
    if set(np.unique(t)) != {0, 1} or set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError(
            "uplift evaluation requires both binary treatment arms and binary outcomes"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("uplift evaluation identity values must be unique")
    if not np.isfinite(score).all():
        raise ValueError("CATE predictions must be finite")
    order = np.lexsort((ids, -score))
    t, y = t[order].astype(np.int64), y[order]
    treated_count = np.cumsum(t)
    control_count = np.cumsum(1 - t)
    treated_y = np.cumsum(y * t)
    control_y = np.cumsum(y * (1 - t))
    indexes = np.unique(
        np.ceil(np.linspace(1, len(t), min(curve_points, len(t)))).astype(int) - 1
    )
    valid = (treated_count[indexes] > 0) & (control_count[indexes] > 0)
    indexes = indexes[valid]
    if len(indexes) < 2:
        raise ValueError(
            "uplift curve requires treated and control observations in at least two prefixes"
        )
    nt, nc = treated_count[indexes], control_count[indexes]
    yt, yc = treated_y[indexes], control_y[indexes]
    selected = nt + nc
    uplift = (yt / nt - yc / nc) * selected
    qini_curve = yt - yc * nt / nc
    coverage = selected / len(t)
    x = np.concatenate(([0.0], coverage.astype(np.float64)))
    u = np.concatenate(([0.0], uplift.astype(np.float64)))
    q = np.concatenate(([0.0], qini_curve.astype(np.float64)))
    auuc = float(np.trapezoid(u, x))
    qini_raw = float(np.trapezoid(q, x))
    qini = qini_raw - float(np.trapezoid(q[-1] * x, x))
    return UpliftEvaluation(
        ate=float(np.mean(score)),
        auuc=auuc,
        qini=qini,
        qini_raw=qini_raw,
        coverage=tuple(float(value) for value in x),
        uplift_curve=tuple(float(value) for value in u),
        qini_curve=tuple(float(value) for value in q),
    )
