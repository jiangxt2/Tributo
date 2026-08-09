"""Portable execution capability and Runtime Adapter protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    WorkerExecutionResult,
)
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

    plan: ResolvedAlgorithmPlan
    input_payloads: tuple[WorkerInputPayload, ...]
    artifacts: tuple[ArtifactDraft, ...] = field(default_factory=tuple)
    cancelled: bool = False

    def __post_init__(self) -> None:
        payloads = tuple(self.input_payloads)
        object.__setattr__(self, "input_payloads", payloads)
        expected = (
            self.plan.runtime.worker_count
            if self.plan.runtime.topology is RuntimeTopology.DATA_PARALLEL
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


__all__ = [
    "AlgorithmExecutionContext",
    "Evaluable",
    "ExecutionEnvelope",
    "Fittable",
    "PortableRuntimeAdapter",
    "Predictable",
    "RuntimeExecutionEnvelope",
    "Transformable",
]
