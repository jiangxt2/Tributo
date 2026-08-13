"""Adapter and model-context protocols for explainability plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from tributo.explainability.contracts import (
    Exactness,
    ExplainabilityRequest,
    FeatureAttribution,
)
from tributo.explainability.reference import ReferenceProvider
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExplainableModelContext:
    """Verified model context built inside a worker process."""

    bundle_uri: str
    model_role: str
    artifact_name: str
    artifact_format: str
    flavor_id: str
    artifact_path: Path | None
    model_object: Any = None
    feature_names: tuple[str, ...] = ()
    objective: str | None = None
    predict: Any = None
    model_digest: str = ""
    preprocessor_digest: str | None = None
    feature_map_digest: str | None = None
    metadata: dict[str, Any] | None = None


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class SupportDecision:
    """Static adapter capability decision."""

    supported: bool
    reason: str = ""
    required_artifacts: tuple[str, ...] = ()
    required_dependencies: tuple[str, ...] = ()
    backend: str = ""
    exactness: Exactness = "conditional"
    warnings: tuple[str, ...] = ()


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class PreparedExplainer:
    """Worker-local prepared adapter object."""

    backend: str
    exactness: Exactness
    explain: Any
    feature_names: tuple[str, ...]
    predict: Any = None
    preprocessor_digest: str | None = None
    feature_map_digest: str | None = None
    base_values: tuple[float, ...] = ()


@PublicAPI(stability="alpha")
@runtime_checkable
class ExplainerAdapter(Protocol):
    """Minimal plugin SPI; implementations must lazy-import dependencies."""

    api_version: ClassVar[int]
    adapter_id: ClassVar[str]
    adapter_version: ClassVar[str]

    @classmethod
    def supports(
        cls,
        context: ExplainableModelContext,
        request: ExplainabilityRequest,
    ) -> SupportDecision: ...

    def prepare(
        self,
        context: ExplainableModelContext,
        request: ExplainabilityRequest,
    ) -> PreparedExplainer: ...

    def explain_batch(
        self,
        prepared: PreparedExplainer,
        batch: np.ndarray,
        *,
        input_ids: tuple[str, ...],
        model_digest: str,
        request: ExplainabilityRequest,
        labels: np.ndarray | None = None,
    ) -> tuple[FeatureAttribution, ...]: ...

    def summarize(
        self,
        attribution_batch: tuple[FeatureAttribution, ...],
        *,
        exactness: Exactness | None = None,
    ) -> tuple[FeatureAttribution, ...]: ...


__all__ = [
    "ExplainableModelContext",
    "ExplainerAdapter",
    "PreparedExplainer",
    "ReferenceProvider",
    "SupportDecision",
]
