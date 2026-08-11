"""Export protocols — contracts for exporters, validators, and source providers.

All protocols use ``typing.Protocol`` with ``@runtime_checkable`` where
runtime ``isinstance`` checks are needed.  ClassVars declare metadata that
the registry and planner consume without instantiation.
"""

from __future__ import annotations

from typing import Any, ClassVar, ContextManager, Mapping, Protocol, runtime_checkable

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

    - ``api_version``: Set to 2 for the current protocol. Version 2 adds
      explicit output flavor, typed options, validator bindings, and
      mutation/upstream declarations.
    - ``exporter_id``: Unique string (e.g. ``"xgboost-onnx-v1"``).
    - ``priority``: Higher = preferred when multiple candidates exist.
    - ``output_format``: Canonical open format id such as ``"onnx"`` or
      ``"ubj"``; one exporter class produces exactly one format.
    - ``output_flavor_id``: Runtime/loader contract written to the artifact.
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
    output_flavor_id: ClassVar[str]
    # Source kinds this exporter consumes (values of ``ExportSource.source_kind``,
    # e.g. "xgboost_result", "dnn_result").  The registry uses this for
    # coarse filtering; transform exporters that consume an upstream
    # artifact (not the source) declare ``()`` and are never filtered out.
    source_kinds: ClassVar[tuple[str, ...]]
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


# ── ExportSourceProvider ───────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class ExportSourceProvider(Protocol):
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
    ) -> ContextManager[ExportSource]: ...


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


# ── Deprecated alias ──────────────────────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """PEP 562 fallback — deprecated ``SourceProvider`` name.

    Emits a ``DeprecationWarning`` (STABILITY.md: 2 minor versions after the
    E1 rename).  ``F822`` for this module is suppressed via
    ``per-file-ignores`` in ``pyproject.toml``: the name is provided
    dynamically by this function, which static AST analysis cannot see.
    """
    if name == "SourceProvider":
        import warnings

        warnings.warn(
            "tributo.exporting.protocols.SourceProvider is deprecated; "
            "use ExportSourceProvider instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return ExportSourceProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExportSourceProvider",
    "ExportValidator",
    "ModelExporter",
    "ModelFactory",
    # Deprecated alias — resolved through __getattr__ (with a warning) so
    # explicit and wildcard imports keep working until removal (STABILITY.md).
    "SourceProvider",
]
