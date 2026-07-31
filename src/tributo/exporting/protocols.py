"""Export protocols — contracts for exporters, validators, and source providers.

All protocols use ``typing.Protocol`` with ``@runtime_checkable`` where
runtime ``isinstance`` checks are needed.  ClassVars declare metadata that
the registry and planner consume without instantiation.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel

from tributo.exporting.models import (
    ArtifactDraft,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    UpstreamRequirement,
    ValidationResult,
    ValidatorBinding,
)
from tributo.util.annotations import PublicAPI

# ── ModelExporter ────────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class ModelExporter(Protocol):
    """A callable that converts a source or upstream artifact into a new artifact.

    Class variables declare metadata consumed by the registry and planner:

    - ``api_version``: Set to 1 for the first-generation protocol.
    - ``exporter_id``: Unique string (e.g. ``"xgboost-onnx-v1"``).
    - ``priority``: Higher = preferred when multiple candidates exist.
    - ``output_format``: e.g. ``"onnx"``, ``"xgboost"``, ``"safetensors"``.
    - ``options_model``: Pydantic model for typed options.
    - ``validator_bindings``: Ordered validator chain.
    - ``mutates_source``: ``True`` if ``export()`` temporarily mutates
      ``source.model_object`` (e.g. XGBoost feature names).
    - ``upstream_requirements``: Declared upstream artifacts this exporter
      needs (e.g. an ONNX quantizer needs an FP32 ONNX artifact).  The
      planner uses this to inject implicit intermediate nodes.

    Instance attributes / construction are not part of the protocol —
    registries store *classes* and instantiate them through the
    framework's dependency-injection mechanism.
    """

    api_version: ClassVar[int]
    exporter_id: ClassVar[str]
    priority: ClassVar[int]
    output_format: ClassVar[str]
    options_model: ClassVar[type[BaseModel]]
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]]
    mutates_source: ClassVar[bool]
    upstream_requirements: ClassVar[tuple[UpstreamRequirement, ...]]

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult: ...

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft: ...


# ── ExportValidator ──────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class ExportValidator(Protocol):
    """Validates an exported artifact (structure / roundtrip / parity)."""

    api_version: ClassVar[int]
    validator_id: ClassVar[str]
    options_model: ClassVar[type[BaseModel]]

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult: ...


# ── SourceProvider ───────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class SourceProvider(Protocol):
    """Resolves a training result (Ray ``Result`` / HF model id) into
    an ``ExportSource`` context manager.

    The context manager holds ``checkpoint.as_directory()`` or HF model
    resources alive for the duration of the export session.
    """

    api_version: ClassVar[int]
    provider_id: ClassVar[str]
    trainer_type: ClassVar[str]
    priority: ClassVar[int]

    def open_source(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Any:  # ContextManager[ExportSource]
        ...


# ── ModelFactory ─────────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class ModelFactory(Protocol):
    """Reconstructs a ``torch.nn.Module`` from an architecture id and config.

    Used by the Safetensors loader to build the model skeleton before
    loading ``state_dict``.
    """

    api_version: ClassVar[int]
    architecture_id: ClassVar[str]

    def build(self, model_config: dict[str, Any]) -> Any:  # returns nn.Module
        ...
