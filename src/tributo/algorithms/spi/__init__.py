"""Provisional extension protocols for portable algorithms."""

from tributo.algorithms.spi.execution import (
    AlgorithmExecutionContext,
    Evaluable,
    ExecutionEnvelope,
    Fittable,
    PortableRuntimeAdapter,
    Predictable,
    RuntimeExecutionEnvelope,
    Transformable,
)
from tributo.algorithms.spi.input import (
    InputExecutionContext,
    InputResolutionContext,
    InputResolverPort,
    InputRuntimeAdapter,
    MaterializedTabularInputView,
    PreparedInput,
    ResolvedInputLease,
    RuntimeInputBinding,
    WorkerInputAdapter,
    WorkerInputPayload,
)

__all__ = [
    "AlgorithmExecutionContext",
    "Evaluable",
    "ExecutionEnvelope",
    "Fittable",
    "InputExecutionContext",
    "InputResolutionContext",
    "InputResolverPort",
    "InputRuntimeAdapter",
    "MaterializedTabularInputView",
    "PortableRuntimeAdapter",
    "Predictable",
    "RuntimeExecutionEnvelope",
    "PreparedInput",
    "ResolvedInputLease",
    "RuntimeInputBinding",
    "Transformable",
    "WorkerInputAdapter",
    "WorkerInputPayload",
]
