"""Trusted test TorchRecipe that imports PyTorch only inside hooks."""

from __future__ import annotations

from collections.abc import Mapping

from tributo.algorithms import (
    TorchArtifactPlan,
    TorchBatch,
    TorchBatchContext,
    TorchLossContribution,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchStepResult,
)


class BinaryLinearRecipe(TorchRecipe):
    """Minimal dense binary classifier used by Core contract tests."""

    def build_modules(self, context: object) -> TorchModuleSet:
        import torch

        return TorchModuleSet(
            {"model": torch.nn.Linear(2, 1), "loss": torch.nn.BCEWithLogitsLoss()}
        )

    def adapt_batch(self, batch: object, context: object) -> TorchBatch:
        import torch

        if not isinstance(batch, Mapping):
            raise TypeError("Torch test batch must be a mapping")
        if not isinstance(context, TorchBatchContext):
            raise TypeError("Torch test batch context is invalid")
        features = torch.stack(
            [batch[name] for name in context.feature_names],
            dim=1,
        )
        if context.label_name is None:
            raise ValueError("Torch test recipe requires a label")
        targets = batch[context.label_name].reshape(-1, 1)
        return TorchBatch(
            positional=(features,), targets=targets, local_rows=len(targets)
        )

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        import torch

        del context
        predictions = modules["model"](batch.positional[0])
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, batch.targets.float(), reduction="sum"
        )
        return TorchStepResult(
            outputs={"prediction": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def configure_optimizers(
        self, modules: TorchModuleSet, context: object
    ) -> TorchOptimizationPlan:
        import torch

        del context
        return TorchOptimizationPlan(
            torch.optim.SGD(modules["model"].parameters(), lr=0.1)
        )

    def metric_plan(self, context: object) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"accuracy": "sum_count", "train_loss": "sum_count"})

    def artifact_plan(self, context: object) -> TorchArtifactPlan:
        del context
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {
                    "name": "features",
                    "dtype": "float32",
                    "shape": ("batch", 2),
                },
            ),
            output_signature=(
                {"name": "prediction", "dtype": "float32", "shape": ("batch", 1)},
            ),
            targets=(
                {
                    "name": "onnx-model",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                },
            ),
            roles={"inference": "onnx-model"},
        )


__all__ = ["BinaryLinearRecipe"]
