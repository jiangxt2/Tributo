"""Framework-neutral algorithm control-plane contracts."""

from tributo.algorithms.api import (
    AlgorithmExecutionResult,
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmRequest,
    AlgorithmRunResult,
    ArtifactDraft,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    ImplementationDescriptor,
    InputBinding,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
    UserExecutionContext,
)
from tributo.algorithms.core.builder import AlgorithmBuilder

__all__ = [
    "AlgorithmExecutionResult",
    "AlgorithmBuilder",
    "AlgorithmOperation",
    "AlgorithmRegistration",
    "AlgorithmRequest",
    "AlgorithmRunResult",
    "ArtifactDraft",
    "BackendInputCompatibility",
    "EnvironmentSpec",
    "ExecutionMode",
    "ImplementationDescriptor",
    "InputBinding",
    "QualifiedReference",
    "RuntimeBinding",
    "RuntimeTopology",
    "UserExecutionContext",
]
