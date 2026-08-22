"""Strict first-party X-Learner contract over five XGBoost boosters."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from tributo._common.config import StrictConfigModel
from tributo.training.x_learner_metrics import classify_quadrants, combine_cate

X_LEARNER_STAGES = ("mu0", "mu1", "tau0", "tau1", "propensity")
X_LEARNER_FORMULA = "propensity*tau0+(1-propensity)*tau1"
X_LEARNER_STAGE_OBJECTIVES = {
    "mu0": "binary:logistic",
    "mu1": "binary:logistic",
    "tau0": "reg:squarederror",
    "tau1": "reg:squarederror",
    "propensity": "binary:logistic",
}
X_LEARNER_QUADRANT_CODES = {
    "lost_cause": 0,
    "persuadable": 1,
    "sleeping_dog": 2,
    "sure_thing": 3,
}


class XLearnerDataConfig(StrictConfigModel):
    """Column roles for binary-treatment, binary-outcome X-Learner."""

    feature_columns: tuple[str, ...]
    treatment_col: str = "treatment"
    outcome_col: str = "outcome"
    identity_col: str = "identity"

    @model_validator(mode="after")
    def validate_roles(self) -> XLearnerDataConfig:
        roles = (self.treatment_col, self.outcome_col, self.identity_col)
        if not self.feature_columns or any(not value for value in self.feature_columns):
            raise ValueError("X-Learner requires non-empty feature columns")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("X-Learner feature columns must be unique")
        if len(set(roles)) != len(roles) or set(roles) & set(self.feature_columns):
            raise ValueError(
                "X-Learner data roles and feature columns must be disjoint"
            )
        return self


class XLearnerModelConfig(StrictConfigModel):
    """XGBoost parameters separated by causal stage type."""

    outcome: dict[str, Any] = Field(
        default_factory=lambda: {"objective": "binary:logistic"}
    )
    effect: dict[str, Any] = Field(
        default_factory=lambda: {"objective": "reg:squarederror"}
    )
    propensity: dict[str, Any] = Field(
        default_factory=lambda: {"objective": "binary:logistic"}
    )

    @model_validator(mode="after")
    def validate_objectives(self) -> XLearnerModelConfig:
        if self.outcome.get("objective") != "binary:logistic":
            raise ValueError("X-Learner outcome objective must be binary:logistic")
        if self.effect.get("objective") != "reg:squarederror":
            raise ValueError("X-Learner effect objective must be reg:squarederror")
        if self.propensity.get("objective") != "binary:logistic":
            raise ValueError("X-Learner propensity objective must be binary:logistic")
        return self


class XLearnerTrainingConfig(StrictConfigModel):
    """Deterministic stage and evaluation parameters."""

    num_rounds: int = Field(default=100, ge=1)
    val_size: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    response_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    propensity_clip: tuple[float, float] = (1e-3, 1.0 - 1e-3)
    curve_points: int = Field(default=100, ge=2, le=1000)
    max_evaluation_rows: int = Field(default=1_000_000, ge=2)

    @model_validator(mode="after")
    def validate_split_and_clip(self) -> XLearnerTrainingConfig:
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("X-Learner val_size + test_size must be below one")
        low, high = self.propensity_clip
        if not 0.0 < low < high < 1.0:
            raise ValueError("propensity_clip must satisfy 0 < low < high < 1")
        return self


class XLearnerRayConfig(StrictConfigModel):
    """Ray-owned execution options accepted by the first implementation."""

    num_workers: int = Field(default=1, ge=1)
    storage_path: str | None = None
    max_failures: int = 0

    @model_validator(mode="after")
    def reject_ungated_retries(self) -> XLearnerRayConfig:
        if self.max_failures != 0:
            raise ValueError("X-Learner requires ray.max_failures=0")
        return self


class XLearnerOutputConfig(StrictConfigModel):
    """Required formal Bundle destination."""

    bundle_uri: str = Field(min_length=1)


class XLearnerConfig(StrictConfigModel):
    """Complete first-party X-Learner configuration."""

    data: XLearnerDataConfig
    model: XLearnerModelConfig = Field(default_factory=XLearnerModelConfig)
    training: XLearnerTrainingConfig = Field(default_factory=XLearnerTrainingConfig)
    ray: XLearnerRayConfig = Field(default_factory=XLearnerRayConfig)
    output: XLearnerOutputConfig


@dataclass(frozen=True)
class XLearnerPrediction:
    """One batch of potential outcomes, components, CATE, and quadrants."""

    mu0: np.ndarray
    mu1: np.ndarray
    tau0: np.ndarray
    tau1: np.ndarray
    propensity: np.ndarray
    cate: np.ndarray
    quadrant: np.ndarray


@dataclass(frozen=True)
class XLearnerTrainingResult:
    """Bounded handoff from staged Ray training to Bundle exporting."""

    checkpoints: Mapping[str, object]
    metrics: Mapping[str, Any]
    feature_names: tuple[str, ...]
    response_threshold: float
    propensity_clip: tuple[float, float]
    stage_evidence: Mapping[str, object]


class XLearnerModel:
    """In-memory fixed-composition model shared by export and inference."""

    def __init__(
        self,
        boosters: Mapping[str, Any],
        *,
        feature_names: tuple[str, ...],
        response_threshold: float,
        propensity_clip: tuple[float, float],
    ) -> None:
        if set(boosters) != set(X_LEARNER_STAGES):
            raise ValueError("X-Learner model requires exactly five component boosters")
        if not feature_names:
            raise ValueError("X-Learner model requires feature names")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X-Learner model feature names must be unique")
        if any(not isinstance(name, str) or not name for name in feature_names):
            raise ValueError("X-Learner model feature names must be non-empty strings")
        if (
            not math.isfinite(response_threshold)
            or not 0.0 <= response_threshold <= 1.0
        ):
            raise ValueError("X-Learner response threshold must be inside [0, 1]")
        low, high = propensity_clip
        if not 0.0 < low < high < 1.0:
            raise ValueError("X-Learner propensity clip is invalid")
        for name, booster in boosters.items():
            num_features = getattr(booster, "num_features", None)
            if callable(num_features) and int(num_features()) != len(feature_names):
                raise ValueError(
                    f"X-Learner component {name!r} has an incompatible feature count"
                )
            booster_feature_names = getattr(booster, "feature_names", None)
            if booster_feature_names is not None and tuple(
                booster_feature_names
            ) != tuple(feature_names):
                raise ValueError(
                    f"X-Learner component {name!r} has incompatible feature names"
                )
        self.boosters = dict(boosters)
        self.feature_names = tuple(feature_names)
        self.response_threshold = float(response_threshold)
        self.propensity_clip = propensity_clip

    def predict(self, features: Any) -> XLearnerPrediction:
        """Predict all fixed X-Learner outputs for one numeric feature matrix."""
        import xgboost

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("X-Learner feature matrix has an incompatible shape")
        dmatrix = xgboost.DMatrix(matrix, feature_names=list(self.feature_names))
        values = {
            name: np.asarray(
                self.boosters[name].predict(dmatrix), dtype=np.float64
            ).reshape(-1)
            for name in X_LEARNER_STAGES
        }
        cate = combine_cate(
            values["tau0"],
            values["tau1"],
            values["propensity"],
            clip=self.propensity_clip,
        )
        quadrant = classify_quadrants(
            values["mu0"],
            values["mu1"],
            threshold=self.response_threshold,
        )
        return XLearnerPrediction(cate=cate, quadrant=quadrant, **values)


_STAGE_LABEL = "__tributo_x_learner_label"


def _load_booster(raw: bytes) -> Any:
    import xgboost

    booster = xgboost.Booster()
    booster.load_model(bytearray(raw))
    return booster


def _label_batch(
    batch: Any,
    *,
    feature_names: tuple[str, ...],
    label_name: str,
) -> Any:
    result = batch.loc[:, list(feature_names)].copy()
    result[_STAGE_LABEL] = batch[label_name].to_numpy()
    return result


class _PseudoOutcomePredictor:
    """Ray Data actor that keeps one outcome Booster warm across batches."""

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        outcome_name: str,
        booster_raw: bytes,
        treated: bool,
    ) -> None:
        self.feature_names = feature_names
        self.outcome_name = outcome_name
        self.booster = _load_booster(booster_raw)
        self.treated = treated

    def __call__(self, batch: Any) -> Any:
        import xgboost

        features = batch.loc[:, list(self.feature_names)]
        prediction = self.booster.predict(
            xgboost.DMatrix(features, feature_names=list(self.feature_names))
        )
        observed = batch[self.outcome_name].to_numpy(dtype=np.float64)
        result = features.copy()
        result[_STAGE_LABEL] = (
            observed - prediction if self.treated else prediction - observed
        )
        return result


class _XLearnerPredictionActor:
    """Ray Data actor that loads the five-Booster model only once."""

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        treatment_name: str,
        outcome_name: str,
        identity_name: str,
        booster_raw: Mapping[str, bytes],
        response_threshold: float,
        propensity_clip: tuple[float, float],
    ) -> None:
        self.feature_names = feature_names
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.identity_name = identity_name
        self.model = XLearnerModel(
            {stage: _load_booster(booster_raw[stage]) for stage in X_LEARNER_STAGES},
            feature_names=feature_names,
            response_threshold=response_threshold,
            propensity_clip=propensity_clip,
        )

    def __call__(self, batch: Any) -> Any:
        prediction = self.model.predict(
            batch.loc[:, list(self.feature_names)].to_numpy()
        )
        return {
            self.identity_name: batch[self.identity_name].to_numpy(),
            self.treatment_name: batch[self.treatment_name].to_numpy(),
            self.outcome_name: batch[self.outcome_name].to_numpy(),
            **{
                name: np.asarray(getattr(prediction, name))
                for name in (
                    "mu0",
                    "mu1",
                    "tau0",
                    "tau1",
                    "propensity",
                    "cate",
                    "quadrant",
                )
            },
        }


def _add_split_key_batch(
    batch: Any,
    *,
    identity_name: str,
    split_key_name: str,
    seed: int,
    phase: str,
) -> Any:
    result = batch.copy()
    result[split_key_name] = [
        f"{seed}:{phase}:{value}" for value in batch[identity_name].tolist()
    ]
    return result


def split_x_learner_dataset(
    dataset: Any,
    *,
    identity_name: str,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[Any, Any | None, Any]:
    """Use Ray Data's streaming hash split with a seed-namespaced identity."""
    existing = set(dataset.schema().names)
    split_key = "__tributo_x_learner_split_key"
    while split_key in existing:
        split_key = f"_{split_key}"

    def split(source: Any, size: float, phase: str) -> tuple[Any, Any]:
        keyed = source.map_batches(
            _add_split_key_batch,
            batch_format="pandas",
            fn_kwargs={
                "identity_name": identity_name,
                "split_key_name": split_key,
                "seed": seed,
                "phase": phase,
            },
        )
        retained, selected = keyed.streaming_train_test_split(
            test_size=size,
            split_type="hash",
            hash_column=split_key,
        )
        return retained.drop_columns([split_key]), selected.drop_columns([split_key])

    remaining, test = split(dataset, test_size, "test")
    if val_size == 0:
        return remaining, None, test
    train, validation = split(
        remaining,
        val_size / (1.0 - test_size),
        "validation",
    )
    return train, validation, test


def validate_x_learner_dataset(dataset: Any, config: XLearnerDataConfig) -> None:
    """Validate global role semantics without materializing rows on Driver."""
    import pyarrow as pa
    from ray.data.expressions import col

    schema = dataset.schema()
    types = dict(zip(schema.names, schema.types, strict=True))
    numeric = config.feature_columns + (config.treatment_col, config.outcome_col)
    invalid_types = [
        name
        for name in numeric
        if not (pa.types.is_integer(types[name]) or pa.types.is_floating(types[name]))
    ]
    if invalid_types:
        raise ValueError(f"X-Learner requires numeric columns: {invalid_types}")

    identity = col(config.identity_col)
    invalid_identity = identity.is_null()
    identity_type = types[config.identity_col]
    if pa.types.is_string(identity_type) or pa.types.is_large_string(identity_type):
        invalid_identity = invalid_identity | (identity.str.strip() == "")
    elif not pa.types.is_integer(identity_type):
        raise ValueError("X-Learner identity must be an integer or string column")
    invalid_expression = (
        col(config.treatment_col).is_null()
        | col(config.outcome_col).is_null()
        | col(config.treatment_col).not_in([0, 1])
        | col(config.outcome_col).not_in([0, 1])
        | invalid_identity
    )
    invalid = dataset.filter(expr=invalid_expression).limit(1)
    if int(invalid.count()):
        raise ValueError("X-Learner requires binary treatment/outcome and identity")
    duplicate = (
        dataset.groupby(config.identity_col)
        .count()
        .filter(expr=col("count()") > 1)
        .limit(1)
    )
    if int(duplicate.count()):
        raise ValueError("X-Learner identity values must be globally unique")


def _evaluate_prediction_batch(
    batch: Any,
    *,
    treatment_name: str,
    outcome_name: str,
    identity_name: str,
    curve_points: int,
) -> Any:
    import pandas as pd

    from tributo.training.x_learner_metrics import evaluate_uplift

    evaluation = evaluate_uplift(
        batch[treatment_name],
        batch[outcome_name],
        batch["cate"],
        batch[identity_name],
        curve_points=curve_points,
    )
    quadrant_counts = {
        str(name): int(count)
        for name, count in batch["quadrant"].value_counts().items()
    }
    return pd.DataFrame(
        {
            "ate": [evaluation.ate],
            "auuc": [evaluation.auuc],
            "qini": [evaluation.qini],
            "qini_raw": [evaluation.qini_raw],
            "coverage": [json.dumps(evaluation.coverage)],
            "uplift_curve": [json.dumps(evaluation.uplift_curve)],
            "qini_curve": [json.dumps(evaluation.qini_curve)],
            "quadrant_counts": [json.dumps(quadrant_counts, sort_keys=True)],
        }
    )


class XLearnerFitDriver:
    """Fit the fixed five stages by sequentially invoking native Ray trainers."""

    def __init__(
        self,
        *,
        datasets: Mapping[str, object],
        config: XLearnerConfig,
        worker_count: int,
        resources_per_worker: Any,
        run_identity: str,
        input_binding_digest: str,
    ) -> None:
        self.datasets = dict(datasets)
        self.config = config
        self.worker_count = worker_count
        self.resources_per_worker = resources_per_worker
        self.run_identity = run_identity
        self.input_binding_digest = input_binding_digest

    def fit(self) -> XLearnerTrainingResult:
        """Execute every stage and produce a bounded composite handoff."""
        from tributo.integrations.algorithm_runtimes.xgboost_stage import (
            XGBoostStageRunner,
        )

        train: Any = self.datasets["train"]
        val: Any | None = self.datasets.get("val")
        test: Any = self.datasets["test"]
        data, training = self.config.data, self.config.training
        from ray.data import ActorPoolStrategy
        from ray.data.expressions import col

        test_rows = int(test.count())
        if test_rows > training.max_evaluation_rows:
            raise ValueError("X-Learner test split exceeds max_evaluation_rows")
        treated = train.filter(expr=col(data.treatment_col) == 1)
        control = train.filter(expr=col(data.treatment_col) == 0)
        treated_val = (
            val.filter(expr=col(data.treatment_col) == 1) if val is not None else None
        )
        control_val = (
            val.filter(expr=col(data.treatment_col) == 0) if val is not None else None
        )
        treated_rows = int(treated.count())
        control_rows = int(control.count())
        train_rows = treated_rows + control_rows
        runner = XGBoostStageRunner(
            worker_count=self.worker_count,
            resources_per_worker=self.resources_per_worker,
            storage_path=self.config.ray.storage_path,
            run_identity=self.run_identity,
            input_binding_digest=self.input_binding_digest,
        )
        features = data.feature_columns

        def labelled(dataset: Any, label: str) -> Any:
            return dataset.map_batches(
                _label_batch,
                batch_format="pandas",
                fn_kwargs={"feature_names": features, "label_name": label},
            )

        stages = {}
        stages["mu0"] = runner.fit(
            "mu0",
            labelled(control, data.outcome_col),
            validation=labelled(control_val, data.outcome_col)
            if control_val is not None
            else None,
            label_col=_STAGE_LABEL,
            xgb_params=self.config.model.outcome,
            num_rounds=training.num_rounds,
            expected_training_rows=control_rows,
        )
        stages["mu1"] = runner.fit(
            "mu1",
            labelled(treated, data.outcome_col),
            validation=labelled(treated_val, data.outcome_col)
            if treated_val is not None
            else None,
            label_col=_STAGE_LABEL,
            xgb_params=self.config.model.outcome,
            num_rounds=training.num_rounds,
            expected_training_rows=treated_rows,
        )

        def pseudo(dataset: Any, raw: bytes, treated_flag: bool) -> Any:
            return dataset.map_batches(
                _PseudoOutcomePredictor,
                batch_format="pandas",
                compute=ActorPoolStrategy(size=1),
                fn_constructor_kwargs={
                    "feature_names": features,
                    "outcome_name": data.outcome_col,
                    "booster_raw": raw,
                    "treated": treated_flag,
                },
            )

        stages["tau0"] = runner.fit(
            "tau0",
            pseudo(control, stages["mu1"].booster_raw, False),
            validation=(
                pseudo(control_val, stages["mu1"].booster_raw, False)
                if control_val is not None
                else None
            ),
            label_col=_STAGE_LABEL,
            xgb_params=self.config.model.effect,
            num_rounds=training.num_rounds,
            expected_training_rows=control_rows,
        )
        stages["tau1"] = runner.fit(
            "tau1",
            pseudo(treated, stages["mu0"].booster_raw, True),
            validation=(
                pseudo(treated_val, stages["mu0"].booster_raw, True)
                if treated_val is not None
                else None
            ),
            label_col=_STAGE_LABEL,
            xgb_params=self.config.model.effect,
            num_rounds=training.num_rounds,
            expected_training_rows=treated_rows,
        )
        stages["propensity"] = runner.fit(
            "propensity",
            labelled(train, data.treatment_col),
            validation=labelled(val, data.treatment_col) if val is not None else None,
            label_col=_STAGE_LABEL,
            xgb_params=self.config.model.propensity,
            num_rounds=training.num_rounds,
            expected_training_rows=train_rows,
        )
        raw = {name: stage.booster_raw for name, stage in stages.items()}
        predictions = test.map_batches(
            _XLearnerPredictionActor,
            batch_format="pandas",
            compute=ActorPoolStrategy(size=self.worker_count),
            fn_constructor_kwargs={
                "feature_names": features,
                "treatment_name": data.treatment_col,
                "outcome_name": data.outcome_col,
                "identity_name": data.identity_col,
                "booster_raw": raw,
                "response_threshold": training.response_threshold,
                "propensity_clip": training.propensity_clip,
            },
        ).sort(["cate", data.identity_col], descending=[True, False])
        metric_rows = (
            predictions.repartition(1)
            .map_batches(
                _evaluate_prediction_batch,
                batch_format="pandas",
                batch_size=None,
                fn_kwargs={
                    "treatment_name": data.treatment_col,
                    "outcome_name": data.outcome_col,
                    "identity_name": data.identity_col,
                    "curve_points": training.curve_points,
                },
            )
            .take(1)
        )
        if len(metric_rows) != 1:
            raise ValueError("X-Learner evaluation did not produce one summary")
        metric = dict(metric_rows[0])
        metrics = {
            "ate": float(metric["ate"]),
            "ate_definition": "model_mean_cate",
            "auuc": float(metric["auuc"]),
            "qini": float(metric["qini"]),
            "qini_raw": float(metric["qini_raw"]),
            "coverage": json.loads(metric["coverage"]),
            "uplift_curve": json.loads(metric["uplift_curve"]),
            "qini_curve": json.loads(metric["qini_curve"]),
            "quadrant_counts": json.loads(metric["quadrant_counts"]),
            "curve_point_count": len(json.loads(metric["coverage"])),
            "uplift_definition": "response_rate_difference_times_selected",
            "auuc_definition": "trapezoid_over_population_coverage",
            "qini_definition": "random_baseline_adjusted_area",
            "qini_raw_definition": "trapezoid_over_population_coverage",
            "test_rows": test_rows,
        }
        return XLearnerTrainingResult(
            checkpoints={name: stage.checkpoint for name, stage in stages.items()},
            metrics=metrics,
            feature_names=features,
            response_threshold=training.response_threshold,
            propensity_clip=training.propensity_clip,
            stage_evidence={name: stage.evidence for name, stage in stages.items()},
        )
