"""Real, bounded DoWhy calls for the locked SciPy dependency closure.

Optional DoWhy imports remain inside tests so dev-only collection does not
require the causal extra. The manifest owns execution with that extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# DoWhy 0.14 contains invalid backslash escapes in Python source docstrings.
# Scope this compilation warning filter to these upstream dependency checks.
pytestmark = pytest.mark.filterwarnings("ignore:invalid escape sequence:SyntaxWarning")


def test_linear_estimation_and_placebo_refutation() -> None:
    from dowhy import CausalModel

    rng = np.random.default_rng(7)
    confounder = rng.normal(size=400)
    treatment = rng.binomial(1, 0.5, size=400)
    frame = pd.DataFrame(
        {
            "confounder": confounder,
            "treatment": treatment,
            "outcome": 3.0 * treatment + 2.0 * confounder,
        }
    )
    model = CausalModel(
        data=frame,
        treatment="treatment",
        outcome="outcome",
        common_causes=["confounder"],
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    refutation = model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        random_seed=7,
        num_simulations=10,
        n_jobs=1,
    )

    assert float(estimate.value) == pytest.approx(3.0, abs=1e-8)
    assert np.isfinite(float(refutation.new_effect))
    assert abs(float(refutation.new_effect)) < 0.5


def test_gcm_fit_attribution_and_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import networkx as nx
    from dowhy import gcm
    from dowhy.gcm.shapley import ShapleyConfig

    monkeypatch.setattr(gcm.config, "default_n_jobs", 1)
    monkeypatch.setattr(gcm.config, "show_progress_bars", False)
    rng = np.random.default_rng(11)
    cause = rng.normal(size=120)
    frame = pd.DataFrame({"cause": cause, "effect": 2.0 * cause})
    model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([("cause", "effect")]))
    model.set_causal_mechanism("cause", gcm.EmpiricalDistribution())
    model.set_causal_mechanism(
        "effect", gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
    )
    gcm.fit(model, frame)
    anomaly = pd.DataFrame({"cause": [6.0], "effect": [12.0]})
    attributions = gcm.attribute_anomalies(
        model,
        "effect",
        anomaly,
        num_distribution_samples=100,
        shapley_config=ShapleyConfig(num_permutations=4, n_jobs=1),
    )
    counterfactual = gcm.counterfactual_samples(
        model,
        {"cause": lambda values: np.zeros_like(values)},
        observed_data=anomaly,
    )

    assert set(attributions) == {"cause", "effect"}
    assert all(np.asarray(values).shape == (1,) for values in attributions.values())
    assert all(np.isfinite(values).all() for values in attributions.values())
    np.testing.assert_allclose(counterfactual["effect"], [0.0], atol=1e-8)
