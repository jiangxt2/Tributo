"""Migrated TorchRecipe fixture kept for compatibility-focused test imports."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, cast

from tributo.algorithms import (
    TorchArtifactContext,
    TorchArtifactPlan,
    TorchBatch,
    TorchBatchContext,
    TorchBuildContext,
    TorchLossContribution,
    TorchMetricContribution,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchStepContext,
    TorchStepResult,
)


def _accuracy(predictions: object, targets: object) -> object:
    import torch

    prediction_tensor = cast(Any, predictions)
    target_tensor = cast(Any, targets)
    predicted = (torch.sigmoid(prediction_tensor) >= 0.5).to(dtype=target_tensor.dtype)
    return (predicted == target_tensor).to(dtype=torch.float32).mean()


class CheckpointCodec:
    """Legacy fixture codec retained only for migration tests."""

    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


class BinaryLinearRecipeV2(TorchRecipe):
    """The old fixture name using the new typed TorchRecipe contract."""

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        model_config = context.runtime.algorithm_config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be a mapping")
        feature_count = int(model_config.get("input_features", 2))
        return TorchModuleSet(
            {
                "model": torch.nn.Linear(feature_count, 1),
                "loss": torch.nn.BCEWithLogitsLoss(),
            }
        )

    def adapt_batch(
        self,
        batch: object,
        context: TorchBatchContext,
    ) -> TorchBatch:
        import torch

        if not isinstance(batch, Mapping):
            raise ValueError("batch must be columnar")
        features = torch.stack(
            [batch[name].to(dtype=torch.float32) for name in context.feature_names],
            dim=1,
        )
        if context.label_name is None:
            raise ValueError("binary fixture requires a label")
        targets = batch[context.label_name].to(dtype=torch.float32).reshape(-1, 1)
        weights = (
            batch.get(context.weight_name) if context.weight_name is not None else None
        )
        return TorchBatch(
            positional=(features,),
            targets=targets,
            weights=weights,
            local_rows=int(features.shape[0]),
        )

    def training_step(
        self,
        modules: TorchModuleSet,
        batch: TorchBatch,
        context: TorchStepContext,
    ) -> TorchStepResult:
        del context
        model = cast(Any, modules["model"])
        loss = cast(Any, modules["loss"])
        predictions = model(batch.positional[0])
        loss_numerator = loss(predictions, batch.targets) * batch.local_rows
        accuracy = _accuracy(predictions, batch.targets)
        return TorchStepResult(
            outputs={"prediction": predictions},
            loss=TorchLossContribution(loss_numerator, batch.local_rows),
            metrics={
                "accuracy": TorchMetricContribution(
                    float(accuracy.detach().item()) * batch.local_rows,
                    batch.local_rows,
                )
            },
        )

    def validation_step(
        self,
        modules: TorchModuleSet,
        batch: TorchBatch,
        context: TorchStepContext,
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def configure_optimizers(
        self,
        modules: TorchModuleSet,
        context: TorchBuildContext,
    ) -> TorchOptimizationPlan:
        import torch

        return TorchOptimizationPlan(
            optimizer=torch.optim.SGD(
                cast(Any, modules["model"]).parameters(),
                lr=float(context.runtime.algorithm_config.get("learning_rate", 0.1)),
            ),
            gradient_accumulation_steps=int(
                context.runtime.algorithm_config.get("accumulation_steps", 1)
            ),
            max_gradient_norm=float(
                context.runtime.algorithm_config.get("max_gradient_norm", 1.0)
            ),
        )

    def metric_plan(self, context: object) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"accuracy": "sum_count", "train_loss": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        del context
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(),
            output_signature=(
                {"name": "prediction", "dtype": "float32", "shape": ("batch", 1)},
            ),
            targets=(),
            roles={},
        )


__all__ = ["BinaryLinearRecipeV2", "CheckpointCodec"]
