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
from tributo.algorithms.spi.input import WorkerInputPayload
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
    input_payload: WorkerInputPayload
    artifacts: tuple[ArtifactDraft, ...] = field(default_factory=tuple)
    cancelled: bool = False


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class RuntimeExecutionEnvelope:
    """Driver-to-Runtime handoff for one Worker or a data-parallel group."""

    run_id: str
    plan: ResolvedAlgorithmPlan
    input_payloads: tuple[WorkerInputPayload, ...]
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


__all__ = [
    "AlgorithmExecutionContext",
    "CollectiveAlgorithm",
    "Evaluable",
    "ExecutionEnvelope",
    "Fittable",
    "FrameworkNativeAlgorithm",
    "MapReduceAlgorithm",
    "PortableRuntimeAdapter",
    "Predictable",
    "RuntimeExecutionEnvelope",
    "Transformable",
]
