"""Trusted test recipe that imports PyTorch only when its factories run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tributo.algorithms import TorchTrainingRecipe


class BinaryLinearRecipe(TorchTrainingRecipe):
    """Minimal dense binary classifier used by contract and integration tests."""

    def model_factory(self, config: Mapping[str, Any]) -> object:
        import torch

        return torch.nn.Linear(int(config.get("input_features", 2)), 1)

    def loss_factory(self, config: Mapping[str, Any]) -> object:
        import torch

        del config
        return torch.nn.BCEWithLogitsLoss()

    def optimizer_factory(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> object:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be torch.nn.Module")
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.get("learning_rate", 0.1)),
        )

    def metric_factories(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        del config

        def accuracy(predictions: object, targets: object) -> object:
            import torch

            if not isinstance(predictions, torch.Tensor) or not isinstance(
                targets, torch.Tensor
            ):
                raise TypeError("accuracy requires Tensor values")
            return (torch.sigmoid(predictions) >= 0.5) == targets.bool()

        return {"accuracy": accuracy}


__all__ = ["BinaryLinearRecipe"]
