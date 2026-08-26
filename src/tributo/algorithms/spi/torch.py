"""Narrow PyTorch recipe contract lowered to the Ray Train collective runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
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


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class OptimizationPlan:
    """Optimizer and bounded loop controls selected by one RecipeV2."""

    optimizer: object
    scheduler: object | None = None
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or isinstance(self.gradient_accumulation_steps, bool)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_gradient_norm is not None and (
            not isinstance(self.max_gradient_norm, (int, float))
            or isinstance(self.max_gradient_norm, bool)
            or float(self.max_gradient_norm) <= 0
        ):
            raise ValueError("max_gradient_norm must be positive when provided")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class MetricPlan:
    """Bounded metric callables keyed by descriptor-declared identities."""

    factories: Mapping[str, Callable[[object, object], object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if any(
            not isinstance(name, str) or not name or not callable(metric)
            for name, metric in self.factories.items()
        ):
            raise ValueError("MetricPlan requires named callable metrics")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TrainingStepResult:
    """One algorithm-owned forward/loss result consumed by the Core loop."""

    predictions: object
    loss: object
    coverage_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, int] = {}
        for name, count in self.coverage_counts.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError(
                    "TrainingStepResult coverage counts require named non-negative "
                    "integers"
                )
            normalized[name] = count
        object.__setattr__(self, "coverage_counts", MappingProxyType(normalized))


@PublicAPI(stability="alpha")
class TrainingRecipeV2(ABC):
    """Define PyTorch mathematics while Core owns the distributed loop."""

    api_version = 2

    @abstractmethod
    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        """Build at least ``model`` and ``loss`` modules."""

    @abstractmethod
    def batch_adapter(
        self,
        batch: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str | None,
        weight_name: str | None,
        config: Mapping[str, Any],
    ) -> tuple[object, object, object | None, int]:
        """Convert a Ray batch into features, targets, weights, and row count."""

    @abstractmethod
    def training_step(
        self,
        modules: Mapping[str, object],
        features: object,
        targets: object,
        weights: object | None,
        config: Mapping[str, Any],
    ) -> TrainingStepResult:
        """Compute predictions and one scalar batch-mean training loss."""

    @abstractmethod
    def validation_step(
        self,
        modules: Mapping[str, object],
        features: object,
        targets: object,
        weights: object | None,
        config: Mapping[str, Any],
    ) -> TrainingStepResult:
        """Compute predictions and one scalar validation loss."""

    @abstractmethod
    def optimization_plan(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> OptimizationPlan:
        """Build optimizer, optional scheduler, accumulation, and clipping."""

    @abstractmethod
    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        """Build bounded metrics consumed by Core collective reduction."""

    @abstractmethod
    def checkpoint_codec(self) -> object:
        """Return a codec for algorithm-specific checkpoint payload state."""


__all__ = [
    "MetricPlan",
    "OptimizationPlan",
    "TorchTrainingRecipe",
    "TrainingRecipeV2",
    "TrainingStepResult",
]
