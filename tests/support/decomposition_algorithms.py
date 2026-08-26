"""Small mathematical Hook implementations for decomposition Runtime tests."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np

from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    EnsembleUnitSpec,
    IterativeOptimizationAlgorithm,
    JoblibEstimatorRecipe,
    MaterializedTabularInputView,
    ParallelEnsembleAlgorithm,
)


class PickleCodec:
    """Bounded deterministic test codec."""

    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


class JoblibProbeRecipe(JoblibEstimatorRecipe):
    """Exercise estimator-internal Ray Joblib tasks."""

    def build_estimator(self, config: Mapping[str, Any]) -> object:
        from tests.support.portable_algorithms import RayJoblibProbeClassifier

        return RayJoblibProbeClassifier(
            task_count=int(config.get("task_count", 4)),
        )

    def fit_arguments(
        self,
        inputs: Mapping[str, object],
        config: Mapping[str, Any],
    ) -> tuple[tuple[object, ...], Mapping[str, object]]:
        del config
        view = cast(MaterializedTabularInputView, inputs["train"])
        columns = view.columns()
        features = np.column_stack(
            [np.asarray(columns[name], dtype=np.float64) for name in view.feature_names]
        )
        labels = np.asarray(columns[cast(str, view.label_name)], dtype=np.int64)
        return (features, labels), {}

    def parallelism_contract(self) -> Mapping[str, object]:
        return {"fit_operations": ("fit",), "exactness": "exact"}

    def extract_model(self, fitted_estimator: object) -> object:
        return fitted_estimator

    def model_codec(self) -> object:
        return PickleCodec()


class ParallelThresholdEnsemble(
    ParallelEnsembleAlgorithm[Mapping[str, float | int], Mapping[str, object]]
):
    """Fit independent deterministic threshold members."""

    def plan_units(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
        seed: int,
    ) -> tuple[EnsembleUnitSpec, ...]:
        del input_descriptor
        count = int(config.get("unit_count", 4))
        return tuple(
            EnsembleUnitSpec(unit_id=f"member-{index}", seed=seed + index)
            for index in range(count)
        )

    def fit_unit(
        self,
        unit: EnsembleUnitSpec,
        inputs: Mapping[str, object],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, float | int]:
        del context
        view = cast(MaterializedTabularInputView, inputs["train"])
        values = np.asarray(view.columns()[view.feature_names[0]], dtype=np.float64)
        return {
            "seed": unit.seed,
            "threshold": float(np.mean(values)),
        }

    def merge_units(
        self,
        ordered_units: tuple[Mapping[str, float | int], ...],
    ) -> object:
        return ordered_units

    def finalize_ensemble(self, merged: object) -> Mapping[str, object]:
        units = cast(tuple[Mapping[str, float | int], ...], merged)
        return {"members": units, "member_count": len(units)}

    def unit_schema(self) -> Mapping[str, object]:
        return {"seed": "int", "threshold": "float"}

    @property
    def retry_safe(self) -> bool:
        return True


class BinaryLogisticOptimization(
    IterativeOptimizationAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
    ]
):
    """Binary L2 logistic gradient descent used as a real iterative baseline."""

    def initialize_state(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
    ) -> Mapping[str, object]:
        del input_descriptor
        feature_count = int(config.get("feature_count", 2))
        return {
            "coef": np.zeros(feature_count, dtype=np.float64),
            "intercept": np.asarray([0.0], dtype=np.float64),
            "round": np.asarray([0], dtype=np.int64),
        }

    def compute_partition_update(
        self,
        batches: Iterable[Mapping[str, object]],
        state: Mapping[str, object],
        round_index: int,
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del round_index, context
        coef = np.asarray(state["coef"], dtype=np.float64)
        intercept = float(np.asarray(state["intercept"])[0])
        gradient = np.zeros_like(coef)
        intercept_gradient = 0.0
        loss_sum = 0.0
        row_count = 0
        for batch in batches:
            features = np.column_stack(
                [np.asarray(batch[name], dtype=np.float64) for name in ("x0", "x1")]
            )
            labels = np.asarray(batch["label"], dtype=np.float64)
            logits = features @ coef + intercept
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            errors = probabilities - labels
            gradient += features.T @ errors
            intercept_gradient += float(np.sum(errors))
            loss_sum += float(
                np.sum(
                    np.logaddexp(0.0, logits) - labels * logits,
                )
            )
            row_count += len(labels)
        return {
            "gradient_sum": gradient,
            "intercept_gradient_sum": np.asarray(
                [intercept_gradient], dtype=np.float64
            ),
            "loss_sum": np.asarray([loss_sum], dtype=np.float64),
            "row_count": np.asarray([row_count], dtype=np.int64),
        }

    def merge_updates(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {name: np.asarray(left[name]) + np.asarray(right[name]) for name in left}

    def apply_update(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, object]:
        rows = int(np.asarray(update["row_count"])[0])
        learning_rate = 0.4
        l2 = 0.01
        coef = np.asarray(state["coef"], dtype=np.float64)
        gradient = np.asarray(update["gradient_sum"], dtype=np.float64) / rows
        return {
            "coef": coef - learning_rate * (gradient + l2 * coef),
            "intercept": np.asarray(state["intercept"], dtype=np.float64)
            - learning_rate
            * np.asarray(update["intercept_gradient_sum"], dtype=np.float64)
            / rows,
            "round": np.asarray([round_index + 1], dtype=np.int64),
        }

    def evaluate_round(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, int | float]:
        del state, round_index
        rows = int(np.asarray(update["row_count"])[0])
        gradient = np.asarray(update["gradient_sum"], dtype=np.float64) / rows
        return {
            "loss": float(np.asarray(update["loss_sum"])[0] / rows),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "row_count": rows,
        }

    def should_stop(
        self,
        state: Mapping[str, object],
        metrics: Mapping[str, int | float],
        round_index: int,
    ) -> bool:
        del state, round_index
        return float(metrics["gradient_norm"]) < 1e-6

    def finalize_model(self, state: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "classes_": np.asarray([0, 1], dtype=np.int64),
            "coef_": np.asarray(state["coef"], dtype=np.float64)[None, :],
            "intercept_": np.asarray(state["intercept"], dtype=np.float64),
            "n_iter_": np.asarray(state["round"], dtype=np.int64),
        }

    def state_schema(self) -> Mapping[str, object]:
        return {"coef": "float64[*]", "intercept": "float64[1]", "round": "int64[1]"}

    def update_schema(self) -> Mapping[str, object]:
        return {
            "gradient_sum": "float64[*]",
            "intercept_gradient_sum": "float64[1]",
            "loss_sum": "float64[1]",
            "row_count": "int64[1]",
        }

    def checkpoint_codec(self) -> object:
        return PickleCodec()

    @property
    def retry_safe(self) -> bool:
        return True


def decomposition_factory(
    *,
    plan: object,
    implementation: object,
    artifacts: tuple[object, ...],
) -> object:
    del plan, artifacts
    if not isinstance(implementation, type):
        raise TypeError("test decomposition implementation must be a class")
    return implementation()


__all__ = [
    "BinaryLogisticOptimization",
    "JoblibProbeRecipe",
    "ParallelThresholdEnsemble",
    "PickleCodec",
    "decomposition_factory",
]
