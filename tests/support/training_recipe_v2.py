"""Independent TrainingRecipeV2 fixture without Ray orchestration code."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, cast

from tributo.algorithms import (
    MetricPlan,
    OptimizationPlan,
    TrainingRecipeV2,
    TrainingStepResult,
)


def _accuracy(predictions: object, targets: object) -> object:
    import torch

    prediction_tensor = cast(Any, predictions)
    target_tensor = cast(Any, targets)
    predicted = (torch.sigmoid(prediction_tensor) >= 0.5).to(dtype=target_tensor.dtype)
    return (predicted == target_tensor).to(dtype=torch.float32).mean()


class PickleCheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


class BinaryLinearRecipeV2(TrainingRecipeV2):
    """Define only modules, batch conversion, Step, Plan, and Codec."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be a mapping")
        feature_count = int(model_config.get("input_features", 2))
        return {
            "model": torch.nn.Linear(feature_count, 1),
            "loss": torch.nn.BCEWithLogitsLoss(),
        }

    def batch_adapter(
        self,
        batch: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str | None,
        weight_name: str | None,
        config: Mapping[str, Any],
    ) -> tuple[object, object, object | None, int]:
        import torch

        del config
        if not isinstance(batch, Mapping):
            raise ValueError("batch must be columnar")
        features = torch.stack(
            [batch[name].to(dtype=torch.float32) for name in feature_names],
            dim=1,
        )
        if label_name is None:
            raise ValueError("binary fixture requires a label")
        targets = batch[label_name].to(dtype=torch.float32).reshape(-1, 1)
        weights = batch.get(weight_name) if weight_name is not None else None
        return features, targets, weights, int(features.shape[0])

    def training_step(
        self,
        modules: Mapping[str, object],
        features: object,
        targets: object,
        weights: object | None,
        config: Mapping[str, Any],
    ) -> TrainingStepResult:
        del weights, config
        model = cast(Any, modules["model"])
        loss = cast(Any, modules["loss"])
        predictions = model(features)
        return TrainingStepResult(
            predictions=predictions, loss=loss(predictions, targets)
        )

    def validation_step(
        self,
        modules: Mapping[str, object],
        features: object,
        targets: object,
        weights: object | None,
        config: Mapping[str, Any],
    ) -> TrainingStepResult:
        return self.training_step(modules, features, targets, weights, config)

    def optimization_plan(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> OptimizationPlan:
        import torch

        return OptimizationPlan(
            optimizer=torch.optim.SGD(
                cast(Any, model).parameters(),
                lr=float(config.get("learning_rate", 0.1)),
            ),
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            max_gradient_norm=float(config.get("max_gradient_norm", 1.0)),
        )

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(factories={"accuracy": _accuracy})

    def checkpoint_codec(self) -> object:
        return PickleCheckpointCodec()


__all__ = ["BinaryLinearRecipeV2", "PickleCheckpointCodec"]
