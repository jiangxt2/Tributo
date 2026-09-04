"""Portable execution capability and Runtime Adapter protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Protocol, TypeVar, runtime_checkable

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    WorkerExecutionResult,
)
from tributo.algorithms.api.distribution import StateField
from tributo.algorithms.spi.input import WorkerInputPayload, WorkerInputPayloadSet
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmExecutionContext:
    """Worker-local context visible to managed executable capabilities."""

    inputs: Mapping[str, object]
    artifacts: tuple[ArtifactDraft, ...] = ()
    worker_metadata: Mapping[str, object] = field(default_factory=dict)
    cancelled: bool = False


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExecutionEnvelope:
    """Serializable control/data handoff consumed by one WorkerBootstrap."""

    plan: ResolvedAlgorithmPlan
    input_payload: WorkerInputPayload | WorkerInputPayloadSet
    artifacts: tuple[ArtifactDraft, ...] = field(default_factory=tuple)
    cancelled: bool = False


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class RuntimeExecutionEnvelope:
    """Driver-to-Runtime handoff for one Worker or a data-parallel group."""

    run_id: str
    plan: ResolvedAlgorithmPlan
    input_payloads: tuple[WorkerInputPayload | WorkerInputPayloadSet, ...]
    artifacts: tuple[ArtifactDraft, ...] = field(default_factory=tuple)
    cancelled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise AlgorithmConfigurationError(
                "Runtime envelope run_id must be non-empty"
            )
        payloads = tuple(self.input_payloads)
        object.__setattr__(self, "input_payloads", payloads)
        expected = (
            self.plan.runtime.worker_count
            if self.plan.runtime.topology
            in {
                RuntimeTopology.DATA_PARALLEL,
                RuntimeTopology.RAY_MAP_REDUCE,
                RuntimeTopology.RAY_ITERATIVE_OPTIMIZATION,
                RuntimeTopology.RAY_TRAIN_TORCH,
            }
            else 1
        )
        if len(payloads) != expected:
            raise AlgorithmConfigurationError(
                "Runtime envelope payload count does not match the resolved topology"
            )

    def worker_envelope(self, rank: int) -> ExecutionEnvelope:
        """Create one rank-specific envelope without changing the resolved plan."""
        return ExecutionEnvelope(
            plan=self.plan,
            input_payload=self.input_payloads[rank],
            artifacts=self.artifacts,
            cancelled=self.cancelled,
        )


@PublicAPI(stability="alpha")
@runtime_checkable
class Fittable(Protocol):
    """Capability protocol for bounded model fitting."""

    def fit(self, context: AlgorithmExecutionContext) -> AlgorithmExecutionResult: ...


@PublicAPI(stability="alpha")
@runtime_checkable
class Evaluable(Protocol):
    """Capability protocol for bounded evaluation."""

    def evaluate(
        self, context: AlgorithmExecutionContext
    ) -> AlgorithmExecutionResult: ...


@PublicAPI(stability="alpha")
@runtime_checkable
class Predictable(Protocol):
    """Capability protocol for bounded prediction."""

    def predict(
        self, context: AlgorithmExecutionContext
    ) -> AlgorithmExecutionResult: ...


@PublicAPI(stability="alpha")
@runtime_checkable
class Transformable(Protocol):
    """Capability protocol for bounded transformation."""

    def transform(
        self, context: AlgorithmExecutionContext
    ) -> AlgorithmExecutionResult: ...


@PublicAPI(stability="alpha")
class PortableRuntimeAdapter(Protocol):
    """Driver-side Runtime Adapter selected by RuntimeBinding.runtime_id."""

    @property
    def runtime_id(self) -> str: ...

    def execute(self, envelope: RuntimeExecutionEnvelope) -> WorkerExecutionResult: ...


@PublicAPI(stability="alpha")
class CollectiveAlgorithm(ABC):
    """Required surface for iterative Ray Train collective algorithms."""

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        """Optionally apply bounded Ray Data transformations before sharding."""
        return datasets

    @abstractmethod
    def build_model(self, config: Mapping[str, Any]) -> object:
        """Construct the worker-local model before collective preparation."""

    @abstractmethod
    def build_optimizer(self, model: object, config: Mapping[str, Any]) -> object:
        """Construct the optimizer for the collectively prepared model."""

    @abstractmethod
    def build_loss(self, config: Mapping[str, Any]) -> object:
        """Construct the loss or risk implementation."""

    @abstractmethod
    def train_loop_per_worker(self, config: Mapping[str, Any]) -> None:
        """Run one rank using Ray Train dataset/report/checkpoint contracts."""

    @abstractmethod
    def checkpoint_state(self, model: object, optimizer: object) -> Mapping[str, Any]:
        """Return bounded state owned by the coordinated checkpoint rank."""


BatchT = TypeVar("BatchT")
PartialStateT = TypeVar("PartialStateT")
ModelT = TypeVar("ModelT")


@PublicAPI(stability="alpha")
class MapReduceAlgorithm(ABC, Generic[BatchT, PartialStateT, ModelT]):
    """Required Hadoop-like surface for bounded sufficient-statistics models."""

    @abstractmethod
    def map_partition(
        self,
        batches: Iterable[BatchT],
        context: AlgorithmExecutionContext,
    ) -> PartialStateT:
        """Map one input shard to one bounded partial state."""

    @abstractmethod
    def merge_states(self, left: PartialStateT, right: PartialStateT) -> PartialStateT:
        """Associatively merge two bounded states."""

    @abstractmethod
    def finalize_model(self, state: PartialStateT) -> ModelT:
        """Construct one model from the globally reduced state."""

    @abstractmethod
    def state_schema(self) -> tuple[StateField, ...]:
        """Return the exact bounded state schema declared by the descriptor."""

    @abstractmethod
    def empty_partition(self) -> PartialStateT:
        """Return the reducer identity for an empty input shard."""

    def coverage_counts(self, state: PartialStateT) -> Mapping[str, int]:
        """Return optional orthogonal coverage dimensions for one map state."""
        del state
        return {}

    @property
    @abstractmethod
    def retry_safe(self) -> bool:
        """Return whether map and merge operations are side-effect free."""


@PublicAPI(stability="alpha")
class FrameworkNativeAlgorithm(ABC):
    """Required binding surface for a framework-owned distributed trainer."""

    @abstractmethod
    def validate_environment(self) -> None:
        """Fail before data access when framework dependencies are incompatible."""

    @abstractmethod
    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        """Bind canonical datasets without driver materialization."""

    @abstractmethod
    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        """Construct the framework-native distributed trainer."""

    @abstractmethod
    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        """Collect actual worker, node, shard, and coordination evidence."""

    @abstractmethod
    def checkpoint_source(self, result: object) -> object:
        """Return the framework checkpoint used by existing Bundle exporting."""


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class EnsembleUnitSpec:
    """One deterministic independently trainable ensemble unit."""

    unit_id: str
    seed: int
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise AlgorithmConfigurationError("ensemble unit_id must be non-empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise AlgorithmConfigurationError("ensemble unit seed must be an integer")


@PublicAPI(stability="alpha")
class JoblibEstimatorRecipe(ABC):
    """Estimator mathematics executed inside Tributo's Ray Joblib runtime."""

    @abstractmethod
    def build_estimator(self, config: Mapping[str, Any]) -> object:
        """Construct an unfitted estimator."""

    @abstractmethod
    def fit_arguments(
        self,
        inputs: Mapping[str, object],
        config: Mapping[str, Any],
    ) -> tuple[tuple[object, ...], Mapping[str, object]]:
        """Return positional and keyword arguments for estimator.fit()."""

    @abstractmethod
    def parallelism_contract(self) -> Mapping[str, object]:
        """Describe estimator-internal parallel operations and exactness."""

    @abstractmethod
    def extract_model(self, fitted_estimator: object) -> object:
        """Extract the fitted model staged by Tributo."""

    @abstractmethod
    def model_codec(self) -> object:
        """Return a bounded codec exposing dumps(model) and loads(payload)."""


UnitModelT = TypeVar("UnitModelT")
EnsembleModelT = TypeVar("EnsembleModelT")


@PublicAPI(stability="alpha")
class ParallelEnsembleAlgorithm(ABC, Generic[UnitModelT, EnsembleModelT]):
    """Independent ensemble units coordinated by Tributo Core."""

    @abstractmethod
    def plan_units(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
        seed: int,
    ) -> tuple[EnsembleUnitSpec, ...]:
        """Plan deterministic unit identities and seeds."""

    @abstractmethod
    def fit_unit(
        self,
        unit: EnsembleUnitSpec,
        inputs: Mapping[str, object],
        context: AlgorithmExecutionContext,
    ) -> UnitModelT:
        """Fit one independent unit without invoking Ray."""

    @abstractmethod
    def merge_units(self, ordered_units: tuple[UnitModelT, ...]) -> object:
        """Merge rank-ordered unit models into bounded intermediate state."""

    @abstractmethod
    def finalize_ensemble(self, merged: object) -> EnsembleModelT:
        """Create the final model from ordered unit state."""

    @abstractmethod
    def unit_schema(self) -> Mapping[str, object]:
        """Return the stable unit-model schema."""

    @property
    @abstractmethod
    def retry_safe(self) -> bool:
        """Return whether one unit can be replayed without side effects."""


GlobalStateT = TypeVar("GlobalStateT")
LocalUpdateT = TypeVar("LocalUpdateT")


@PublicAPI(stability="alpha")
class IterativeOptimizationAlgorithm(
    ABC,
    Generic[BatchT, GlobalStateT, LocalUpdateT, ModelT],
):
    """Synchronous shard updates coordinated by Tributo Core."""

    @abstractmethod
    def initialize_state(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
    ) -> GlobalStateT:
        """Create bounded global state before round zero."""

    @abstractmethod
    def compute_partition_update(
        self,
        batches: Iterable[BatchT],
        state: GlobalStateT,
        round_index: int,
        context: AlgorithmExecutionContext,
    ) -> LocalUpdateT:
        """Compute one replay-safe shard update against immutable state."""

    @abstractmethod
    def merge_updates(self, left: LocalUpdateT, right: LocalUpdateT) -> LocalUpdateT:
        """Associatively merge two local updates."""

    @abstractmethod
    def apply_update(
        self,
        state: GlobalStateT,
        update: LocalUpdateT,
        round_index: int,
    ) -> GlobalStateT:
        """Advance global state after the round barrier."""

    @abstractmethod
    def evaluate_round(
        self,
        state: GlobalStateT,
        update: LocalUpdateT,
        round_index: int,
    ) -> Mapping[str, int | float]:
        """Return bounded round metrics."""

    @abstractmethod
    def should_stop(
        self,
        state: GlobalStateT,
        metrics: Mapping[str, int | float],
        round_index: int,
    ) -> bool:
        """Return whether the next round should be skipped."""

    @abstractmethod
    def finalize_model(self, state: GlobalStateT) -> ModelT:
        """Create the final model from global state."""

    @abstractmethod
    def state_schema(self) -> Mapping[str, object]:
        """Return the bounded global-state schema."""

    @abstractmethod
    def update_schema(self) -> Mapping[str, object]:
        """Return the bounded local-update schema."""

    @abstractmethod
    def checkpoint_codec(self) -> object:
        """Return a codec exposing dumps(state) and loads(payload)."""

    @property
    @abstractmethod
    def retry_safe(self) -> bool:
        """Return whether one shard update can be replayed before the barrier."""


__all__ = [
    "AlgorithmExecutionContext",
    "CollectiveAlgorithm",
    "Evaluable",
    "ExecutionEnvelope",
    "Fittable",
    "FrameworkNativeAlgorithm",
    "EnsembleUnitSpec",
    "IterativeOptimizationAlgorithm",
    "JoblibEstimatorRecipe",
    "MapReduceAlgorithm",
    "ParallelEnsembleAlgorithm",
    "PortableRuntimeAdapter",
    "Predictable",
    "RuntimeExecutionEnvelope",
    "Transformable",
]
