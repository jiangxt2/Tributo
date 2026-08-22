"""Narrow PyTorch recipe contract lowered to the Ray Train collective runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class TorchTrainingRecipe(ABC):
    """Define model semantics without owning the distributed worker loop.

    A default recipe author implements only the four factory methods. Tributo
    combines their products with Ray Train, Ray Data, and PyTorch DDP. The
    optional ``forward`` and ``compute_loss`` methods are the bounded advanced
    escape hatches; data sharding, reporting, checkpointing, and Bundle
    publication remain framework-owned infrastructure. Multi-worker exact
    coverage can invoke ``forward`` with a zero-row batch after another rank
    exhausts its shard, so recipe models must preserve their output contract
    for an empty leading batch dimension.
    """

    api_version = 1

    @abstractmethod
    def model_factory(self, config: Mapping[str, Any]) -> object:
        """Build one worker-local ``torch.nn.Module`` from model config."""

    @abstractmethod
    def loss_factory(self, config: Mapping[str, Any]) -> object:
        """Build a scalar batch-mean loss callable from loss config."""

    @abstractmethod
    def optimizer_factory(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> object:
        """Build an optimizer for the unwrapped model from optimizer config."""

    @abstractmethod
    def metric_factories(
        self,
        config: Mapping[str, Any],
    ) -> Mapping[str, Callable[[object, object], object]]:
        """Build metric callables keyed by declared metric name."""

    def forward(self, model: object, features: object) -> object:
        """Invoke the default one-tensor model signature."""
        if not callable(model):
            raise TypeError("recipe model must be callable")
        return model(features)

    def compute_loss(
        self,
        loss: object,
        predictions: object,
        targets: object,
    ) -> object:
        """Invoke the default ``loss(predictions, targets)`` signature."""
        if not callable(loss):
            raise TypeError("recipe loss must be callable")
        return loss(predictions, targets)


__all__ = ["TorchTrainingRecipe"]
