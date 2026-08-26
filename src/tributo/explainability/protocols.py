"""Adapter and model-context protocols for explainability plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from tributo.explainability.contracts import (
    Exactness,
    ExplainabilityDescriptor,
    ExplainabilityLimits,
    ExplainabilityReceipt,
    ExplainabilityRequest,
    FeatureAttribution,
    ReferenceBinding,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ResolvedReference:
    """Materialized reference data and its immutable provenance."""

    data: np.ndarray
    digest: str
    rows: int


@runtime_checkable
@PublicAPI(stability="alpha")
class ReferenceProvider(Protocol):
    """Load and identify reference data without exposing storage details."""

    provider_id: ClassVar[str]

    def resolve(
        self,
        binding: ReferenceBinding,
        limits: ExplainabilityLimits,
    ) -> ResolvedReference: ...

    def digest(self, binding: ReferenceBinding) -> str: ...


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
    native_attribution_id: str | None = None
    metadata: dict[str, Any] | None = None


@PublicAPI(stability="alpha")
class ExplainabilityModelSession(Protocol):
    """Opened model context with deterministic resource cleanup."""

    @property
    def context(self) -> ExplainableModelContext: ...

    @property
    def output_count_upper_bound(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
@PublicAPI(stability="alpha")
class ExplainabilityModelSessionFactory(Protocol):
    """Serializable factory that opens one model session inside a Ray worker."""

    factory_id: ClassVar[str]

    def create(
        self, reference_provider: ReferenceProvider
    ) -> ExplainabilityModelSession: ...


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExplainabilityModelBinding:
    """Verified model metadata plus an opaque worker-local session factory."""

    bundle_id: str
    bundle_digest: str
    manifest_sha256: str
    model_role: str
    model_digest: str
    preprocessor_digest: str | None
    feature_map_digest: str | None
    descriptor: ExplainabilityDescriptor
    backend: str
    exactness: Exactness
    output_count_upper_bound: int
    session_factory: ExplainabilityModelSessionFactory
    dependency_versions: tuple[tuple[str, str], ...] = ()


@PublicAPI(stability="alpha")
class ExplainabilityModelProvider(Protocol):
    """Build one verified explainability model context outside the core worker."""

    provider_id: ClassVar[str]

    def resolve(self, request: ExplainabilityRequest) -> ExplainabilityModelBinding: ...


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExplainabilityMaterialization:
    """Credential-free facts returned by an explainability result adapter."""

    digest: str
    total_bytes: int
    rows: int


@runtime_checkable
@PublicAPI(stability="alpha")
class ExplainabilityResultStore(Protocol):
    """Persist and inspect explainability results outside the domain executor."""

    provider_id: ClassVar[str]

    def materialize(
        self,
        dataset: object,
        *,
        uri: str,
        storage_profile: str | None,
        max_bytes: int | None,
        run_id: str,
        plan_digest: str,
    ) -> ExplainabilityMaterialization: ...

    def write_receipt(
        self,
        uri: str,
        receipt: ExplainabilityReceipt,
        *,
        storage_profile: str | None,
    ) -> None: ...

    def read_receipt(
        self,
        uri: str,
        *,
        storage_profile: str | None,
    ) -> ExplainabilityReceipt | None: ...

    def cleanup(self, uri: str, *, storage_profile: str | None) -> None: ...


@runtime_checkable
@PublicAPI(stability="alpha")
class NativeAttributionModel(Protocol):
    """Loaded runtime capability for model-native feature attribution."""

    @property
    def native_attribution_id(self) -> str | None: ...

    @property
    def native_model_object(self) -> Any: ...

    @property
    def native_feature_names(self) -> tuple[str, ...]: ...

    @property
    def native_objective(self) -> str | None: ...

    def native_attribution_support(
        self,
        request: ExplainabilityRequest,
    ) -> SupportDecision: ...

    def prepare_native_attribution(
        self,
        request: ExplainabilityRequest,
        *,
        feature_names: tuple[str, ...],
        reference_data: np.ndarray | None,
    ) -> PreparedExplainer: ...


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
    "ExplainabilityModelProvider",
    "ExplainabilityModelSession",
    "ExplainabilityModelSessionFactory",
    "ExplainabilityMaterialization",
    "ExplainabilityModelBinding",
    "ExplainabilityResultStore",
    "NativeAttributionModel",
    "ExplainerAdapter",
    "PreparedExplainer",
    "ReferenceProvider",
    "ResolvedReference",
    "SupportDecision",
]
