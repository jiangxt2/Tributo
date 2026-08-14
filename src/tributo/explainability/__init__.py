"""Pluggable model explainability contracts and execution adapters."""

from tributo.explainability.contracts import (
    ExplainabilityConfig,
    ExplainabilityDescriptor,
    ExplainabilityLimits,
    ExplainabilityOperationRecord,
    ExplainabilityReceipt,
    ExplainabilityRequest,
    FeatureAttribution,
    ReferenceBinding,
    ResourcePolicy,
    ResultPolicy,
)
from tributo.explainability.planner import ExplainabilityPlan, ExplainabilityPlanner
from tributo.explainability.reference import (
    FileReferenceProvider,
    ReferenceProvider,
    ResolvedReference,
)
from tributo.explainability.registry import (
    ExplainerRegistry,
    get_default_explainer_registry,
)

__all__ = [
    "ExplainabilityConfig",
    "ExplainabilityDescriptor",
    "ExplainabilityLimits",
    "ExplainabilityOperationRecord",
    "ExplainabilityPlan",
    "ExplainabilityPlanner",
    "ExplainabilityReceipt",
    "ExplainabilityRequest",
    "ExplainerRegistry",
    "FeatureAttribution",
    "ReferenceBinding",
    "ResultPolicy",
    "ResourcePolicy",
    "FileReferenceProvider",
    "ReferenceProvider",
    "ResolvedReference",
    "get_default_explainer_registry",
]
