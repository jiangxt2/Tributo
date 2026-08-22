"""Golden tests for X-Learner combination and uplift semantics."""

from __future__ import annotations

import numpy as np
import pytest

from tributo.training.x_learner_metrics import (
    classify_quadrants,
    combine_cate,
    evaluate_uplift,
)


def test_combine_cate_uses_propensity_weighted_x_learner_formula() -> None:
    actual = combine_cate([1.0, 2.0], [3.0, 4.0], [0.25, 0.75])
    np.testing.assert_allclose(actual, [2.5, 2.5])


def test_four_quadrants_include_threshold_equality() -> None:
    actual = classify_quadrants(
        [0.4, 0.5, 0.4, 0.5],
        [0.5, 0.5, 0.4, 0.4],
    )
    assert actual.tolist() == [
        "persuadable",
        "sure_thing",
        "lost_cause",
        "sleeping_dog",
    ]


def test_uplift_metrics_are_deterministic_and_baseline_adjusted() -> None:
    result = evaluate_uplift(
        treatment=[1, 0, 1, 0, 1, 0],
        outcome=[1, 0, 1, 0, 0, 1],
        cate=[0.9, 0.9, 0.7, 0.5, -0.2, -0.4],
        identity=["b", "a", "c", "d", "e", "f"],
        curve_points=6,
    )
    assert result.ate == pytest.approx(0.4)
    assert result.coverage == pytest.approx((0.0, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0))
    assert result.uplift_curve == pytest.approx((0.0, 2.0, 3.0, 4.0, 10 / 3, 2.0))
    assert result.qini_curve == pytest.approx((0.0, 1.0, 2.0, 2.0, 2.0, 1.0))
    assert result.auuc == pytest.approx(43 / 18)
    assert result.qini_raw == pytest.approx(4 / 3)
    assert result.qini == pytest.approx(5 / 6)


def test_numeric_identity_breaks_cate_ties_in_numeric_order() -> None:
    result = evaluate_uplift(
        treatment=[1, 0, 0, 1],
        outcome=[0, 0, 1, 1],
        cate=[0.5, 0.5, 0.5, 0.5],
        identity=[10, 2, 11, 1],
        curve_points=4,
    )

    assert result.coverage[1] == pytest.approx(0.5)
    assert result.qini_curve[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "treatment": [1, 1],
                "outcome": [1, 0],
                "cate": [1, 0],
                "identity": [1, 2],
            },
            "both binary treatment arms",
        ),
        (
            {
                "treatment": [1, 0],
                "outcome": [1, 0],
                "cate": [1, 0],
                "identity": [1, 1],
            },
            "unique",
        ),
    ],
)
def test_uplift_metrics_fail_closed(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_uplift(**kwargs)
