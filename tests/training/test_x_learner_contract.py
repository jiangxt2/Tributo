"""Strict X-Learner configuration and composition tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tributo.algorithms.api import AlgorithmConfigurationError, ExecutionProfile
from tributo.algorithms.builtin.x_learner import (
    DistributedXLearner,
    _validate_local_control_plane_headroom,
)
from tributo.training.x_learner import (
    _STAGE_LABEL,
    XLearnerConfig,
    XLearnerModel,
    _PseudoOutcomePredictor,
    validate_x_learner_dataset,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Tip.*future versions of Ray.*:FutureWarning",
    "ignore::pytest.PytestUnraisableExceptionWarning",
)


def _config() -> dict:
    return {
        "data": {
            "feature_columns": ["x1", "x2"],
            "treatment_col": "t",
            "outcome_col": "y",
            "identity_col": "id",
        },
        "output": {"bundle_uri": "./bundle"},
    }


def test_config_freezes_binary_xgboost_and_disjoint_roles() -> None:
    config = XLearnerConfig.model_validate(_config())
    assert config.model.outcome["objective"] == "binary:logistic"
    assert config.model.effect["objective"] == "reg:squarederror"
    assert config.training.test_size > 0

    invalid = _config()
    invalid["data"]["feature_columns"] = ["x1", "t"]
    with pytest.raises(ValueError, match="disjoint"):
        XLearnerConfig.model_validate(invalid)


def test_model_requires_exact_component_set() -> None:
    with pytest.raises(ValueError, match="exactly five"):
        XLearnerModel(
            {"mu0": object()},
            feature_names=("x1",),
            response_threshold=0.5,
            propensity_clip=(0.01, 0.99),
        )


def test_model_combines_component_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("xgboost")

    class Booster:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, matrix):
            return np.full(matrix.num_row(), self.value)

    model = XLearnerModel(
        {
            "mu0": Booster(0.2),
            "mu1": Booster(0.8),
            "tau0": Booster(1.0),
            "tau1": Booster(3.0),
            "propensity": Booster(0.25),
        },
        feature_names=("x1", "x2"),
        response_threshold=0.5,
        propensity_clip=(0.01, 0.99),
    )
    result = model.predict([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(result.cate, [2.5, 2.5])
    assert result.quadrant.tolist() == ["persuadable", "persuadable"]


def _plan(*, num_gpus: float = 0.0, sample_weight: str | None = None) -> object:
    return SimpleNamespace(
        algorithm_config=_config(),
        runtime=SimpleNamespace(worker_count=1, num_gpus=num_gpus),
        input_binding=SimpleNamespace(
            feature_names=("x1", "x2", "t", "id"),
            label_name="y",
            sample_weight_name=sample_weight,
        ),
    )


def test_formal_x_learner_rejects_ungated_gpu_and_sample_weights() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="CPU-only"):
        DistributedXLearner(_plan(num_gpus=1.0))
    with pytest.raises(AlgorithmConfigurationError, match="sample weights"):
        DistributedXLearner(_plan(sample_weight="weight"))


def test_owned_local_requires_one_control_plane_cpu_beyond_workers() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="control-plane CPU"):
        _validate_local_control_plane_headroom(
            profile=ExecutionProfile.LOCAL,
            worker_count=2,
            cluster_resources={"CPU": 2.0},
        )
    _validate_local_control_plane_headroom(
        profile=ExecutionProfile.LOCAL,
        worker_count=2,
        cluster_resources={"CPU": 3.0},
    )
    _validate_local_control_plane_headroom(
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
        cluster_resources={"CPU": 0.0},
    )


def test_dataset_validation_rejects_null_treatment_before_training(
    ray_local_runtime: None,
) -> None:
    del ray_local_runtime
    import ray

    config = XLearnerConfig.model_validate(_config()).data
    dataset = ray.data.from_items(
        [
            {"id": 1, "x1": 1.0, "x2": 2.0, "t": None, "y": 0},
            {"id": 2, "x1": 2.0, "x2": 3.0, "t": 1, "y": 1},
        ]
    )

    with pytest.raises(ValueError, match="binary treatment/outcome"):
        validate_x_learner_dataset(dataset, config)


def test_pseudo_outcome_directions_match_x_learner_contract() -> None:
    pytest.importorskip("xgboost")
    pd = pytest.importorskip("pandas")

    class Booster:
        @staticmethod
        def predict(matrix):
            return np.asarray([0.8, 0.2], dtype=np.float64)

    batch = pd.DataFrame({"x1": [1.0, 2.0], "y": [1.0, 0.0]})

    def actor(*, treated: bool) -> _PseudoOutcomePredictor:
        value = object.__new__(_PseudoOutcomePredictor)
        value.feature_names = ("x1",)
        value.outcome_name = "y"
        value.booster = Booster()
        value.treated = treated
        return value

    control = actor(treated=False)(batch)
    treated = actor(treated=True)(batch)

    np.testing.assert_allclose(control[_STAGE_LABEL], [-0.2, 0.2])
    np.testing.assert_allclose(treated[_STAGE_LABEL], [0.2, -0.2])
