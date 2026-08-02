"""Tests for the causal refutation contract (T2).

``BaseCausalEstimator.refute()`` deliberately has no default
implementation: an unimplemented refutation must fail loudly instead of
reporting the estimate as passed.
"""

from __future__ import annotations

from typing import Any

import pytest

from tributo.training.causal_estimator import (
    BaseCausalEstimator,
    CausalEffect,
    CausalGraph,
    RefutationResult,
)


class _MinimalCausal(BaseCausalEstimator):
    """Implements the abstract hooks but deliberately not ``refute()``."""

    def identify(
        self,
        data: Any,
        treatment: str,
        outcome: str,
        **kwargs: Any,
    ) -> CausalGraph:
        return CausalGraph(treatment=treatment, outcome=outcome)

    def estimate(self, data: Any, causal_graph: CausalGraph) -> CausalEffect:
        return CausalEffect(method="test.ols", estimate_value=1.0)


class _RefutingCausal(_MinimalCausal):
    """Overrides ``refute()`` with a real implementation."""

    def refute(
        self,
        estimate: CausalEffect,
        method: str = "placebo",
    ) -> RefutationResult:
        return RefutationResult(method=method, passed=True)


def _estimator(
    cls: type[BaseCausalEstimator] = _MinimalCausal,
) -> BaseCausalEstimator:
    return cls(
        datasets={},
        config={"causal": {"treatment": "t", "outcome": "y"}},
    )


class TestRefuteContract:
    def test_unimplemented_refute_raises(self) -> None:
        est = _estimator()
        est.setup()
        with pytest.raises(NotImplementedError, match="does not implement refute"):
            est.refute(CausalEffect(method="test.ols"))

    def test_error_message_names_estimator(self) -> None:
        est = _estimator()
        est.setup()
        with pytest.raises(NotImplementedError, match="_MinimalCausal"):
            est.refute(CausalEffect(method="test.ols"))

    def test_override_refute_returns_result(self) -> None:
        est = _estimator(_RefutingCausal)
        est.setup()
        result = est.refute(CausalEffect(method="test.ols", estimate_value=2.0))
        assert result.passed is True
        assert result.method == "placebo"

    def test_training_loop_propagates_unimplemented_refute(
        self,
        monkeypatch: Any,
    ) -> None:
        est = _estimator()
        est.setup()
        monkeypatch.setattr(est, "_load_data", lambda: None)
        monkeypatch.setattr(
            est,
            "identify",
            lambda *args, **kwargs: CausalGraph(treatment="t", outcome="y"),
        )
        monkeypatch.setattr(
            est,
            "estimate",
            lambda *args, **kwargs: CausalEffect(method="test.ols"),
        )
        with pytest.raises(NotImplementedError, match="does not implement refute"):
            est.training_loop()

    def test_training_loop_with_refute_override(self, monkeypatch: Any) -> None:
        est = _estimator(_RefutingCausal)
        est.setup()
        monkeypatch.setattr(est, "_load_data", lambda: None)
        monkeypatch.setattr(
            est,
            "identify",
            lambda *args, **kwargs: CausalGraph(treatment="t", outcome="y"),
        )
        monkeypatch.setattr(
            est,
            "estimate",
            lambda *args, **kwargs: CausalEffect(method="test.ols"),
        )
        result = est.training_loop()
        assert set(result) == {"effect", "refutation"}
        assert result["refutation"].passed is True
