"""Phase 2a provider architecture prototypes.

These types are for validation only — they stay in this worktree and do NOT
merge to master.  Phase 2b can re-design based on validation conclusions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, Sequence, Union

import pyarrow as pa
from pydantic import BaseModel, Field, model_validator

from tributo.data.source_config import SourceInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine / Provider identity
# ---------------------------------------------------------------------------


class TransformBackend(Enum):
    """Execution backend for Transform compilation.

    No LEGACY member — Legacy providers return Ray Datasets and use the
    RAY transform backend.
    """

    DAFT = "daft"
    RAY = "ray"


@dataclass(frozen=True)
class ProviderIdentity:
    """Globally unique provider identifier.

    ``selector_alias`` is the user-facing short name for ``engine=`` routing.
    Third-party providers that don't need a short alias set it to ``None``.
    """

    provider_id: str  # "tributo.daft", "tributo.legacy.clickhouse"
    selector_alias: str | None  # "daft", "legacy"
    transform_backend: TransformBackend


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Availability:
    """Pre-open eligibility check result."""

    is_available: bool
    missing_extra: str | None = None
    reason: str | None = None

    @classmethod
    def ready(cls) -> "Availability":
        return cls(is_available=True)

    @classmethod
    def unavailable(cls, *, reason: str, missing_extra: str | None = None) -> "Availability":
        return cls(is_available=False, reason=reason, missing_extra=missing_extra)


# ---------------------------------------------------------------------------
# TransformSpec — Pydantic discriminated union, deserializable from YAML/JSON
# ---------------------------------------------------------------------------


class FilterEq(BaseModel):
    """Arrow equality. ``value=None`` is illegal — use ``FilterNull``."""

    type: Literal["filter_eq"] = "filter_eq"
    column: str
    value: Any

    @model_validator(mode="after")
    def _reject_null_value(self) -> "FilterEq":
        if self.value is None:
            raise ValueError("FilterEq(value=None) is illegal; use FilterNull")
        return self


class FilterRange(BaseModel):
    """Inclusive range ``low ≤ column ≤ high``. ``low > high`` is a compile error."""

    type: Literal["filter_range"] = "filter_range"
    column: str
    low: Any
    high: Any

    @model_validator(mode="after")
    def _reject_inverted_range(self) -> "FilterRange":
        try:
            if self.low > self.high:
                raise ValueError(
                    f"FilterRange low ({self.low!r}) > high ({self.high!r})"
                )
        except TypeError:
            # Incomparable types — defer to runtime / schema check
            pass
        return self


class FilterIsIn(BaseModel):
    """Set membership. Empty list → always false. ``None`` in values is illegal."""

    type: Literal["filter_isin"] = "filter_isin"
    column: str
    values: list[Any]

    @model_validator(mode="after")
    def _reject_null_in_values(self) -> "FilterIsIn":
        if any(v is None for v in self.values):
            raise ValueError("FilterIsIn values cannot contain None; use FilterNull")
        return self


class FilterNull(BaseModel):
    """Match only Arrow null, not IEEE NaN."""

    type: Literal["filter_null"] = "filter_null"
    column: str


class SelectColumns(BaseModel):
    """Select and reorder columns. At least one column required."""

    type: Literal["select_columns"] = "select_columns"
    columns: list[str] = Field(min_length=1)


TransformSpec = Annotated[
    Union[FilterEq, FilterRange, FilterIsIn, FilterNull, SelectColumns],
    Field(discriminator="type"),
]


class TransformPipeline(BaseModel):
    """Ordered sequence of TransformSpecs.

    An empty pipeline is legal and acts as identity.
    """

    steps: list[TransformSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Compiled pipeline — compiler output with ordinal trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledStep:
    """A single compiled TransformSpec with schema trace."""

    ordinal: int  # Position in the original TransformPipeline
    spec: TransformSpec
    input_schema: pa.Schema
    output_schema: pa.Schema
    backend_op: Any  # Engine-specific expression / operation


class CompiledPipeline:
    """Ordered compiled steps. Each step is tagged with its original ordinal.

    Backend-specific application is handled by the standalone functions
    ``apply_pipeline_to_daft_df`` / ``apply_pipeline_to_ray_ds`` in
    ``transform_compiler``, which walk ``.steps`` and dispatch by spec type.
    """

    def __init__(self, steps: Sequence[CompiledStep]) -> None:
        self._steps: tuple[CompiledStep, ...] = tuple(steps)
        seen = set()
        for s in self._steps:
            if s.ordinal in seen:
                raise ValueError(f"Duplicate ordinal {s.ordinal} in compiled pipeline")
            seen.add(s.ordinal)

    @property
    def steps(self) -> tuple[CompiledStep, ...]:
        return self._steps


# ---------------------------------------------------------------------------
# TransformCompiler — single compilation entry point
# ---------------------------------------------------------------------------


class TransformCompiler(ABC):
    """Compile a TransformPipeline to a CompiledPipeline for a specific backend.

    The provider calls ``compile()`` once for the full pipeline, then uses
    ``CompiledPipeline.apply()`` for residual steps after pushdown.
    """

    @abstractmethod
    def compile(
        self,
        pipeline: TransformPipeline,
        backend: TransformBackend,
        input_schema: pa.Schema,
    ) -> CompiledPipeline: ...


# ---------------------------------------------------------------------------
# SourcePlan — lazy plan handle
# ---------------------------------------------------------------------------


class SourcePlan(ABC):
    """Engine-agnostic lazy plan handle.

    ``apply()`` must be called before ``materialize()`` so that transforms
    are folded into the native plan (preserving scan pushdown).
    """

    @property
    @abstractmethod
    def identity(self) -> ProviderIdentity: ...

    @abstractmethod
    def schema(self) -> pa.Schema: ...

    @abstractmethod
    def apply(self, pipeline: TransformPipeline) -> "SourcePlan":
        """Compile and apply transforms to the native lazy plan.

        Returns a new SourcePlan with the transforms folded in.
        """

    @abstractmethod
    def materialize(self) -> Any:  # ray.data.dataset.MaterializedDataset
        """Execute the plan and return a materialized Ray Dataset."""


# ---------------------------------------------------------------------------
# SourceProvider ABC
# ---------------------------------------------------------------------------


class SourceProvider(ABC):
    """Engine-level data source provider.

    Each provider has a unique ``identity()``, declares what source types
    it handles via ``can_handle()``, and reports environment availability
    via ``availability()``.
    """

    @classmethod
    @abstractmethod
    def identity(cls) -> ProviderIdentity:
        """Return identity before ``open()``, for deterministic routing."""

    @abstractmethod
    def open(self, config: SourceInput) -> SourcePlan: ...

    @classmethod
    @abstractmethod
    def can_handle(cls, config: SourceInput) -> bool:
        """Does this provider support this data source type? (pure capability)"""

    @classmethod
    def availability(cls, config: SourceInput) -> Availability:
        """Is this provider currently usable in this environment?"""
        return Availability.ready()


# ---------------------------------------------------------------------------
# SourceRouter — deterministic provider selection
# ---------------------------------------------------------------------------


@dataclass
class SourceRouter:
    """Policy-driven provider selection.

    The strategy table maps ``(source_kind, dialect)`` to an ordered list of
    ``provider_id`` strings.  Selection walks the list, calling
    ``can_handle()`` then ``availability()``.  The first eligible provider
    is used.

    Explicit ``engine=`` bypasses the strategy table and selects by alias
    or full provider_id.
    """

    strategy: dict[tuple[str, str | None], tuple[str, ...]] = field(default_factory=dict)
    _registry: dict[str, type[SourceProvider]] = field(default_factory=dict, init=False, repr=False)

    def register(self, cls: type[SourceProvider]) -> None:
        ident = cls.identity()
        if ident.provider_id in self._registry:
            raise ValueError(f"Duplicate provider_id: {ident.provider_id!r}")
        self._registry[ident.provider_id] = cls
        logger.info("Registered provider %r", ident.provider_id)

    def list_providers(self) -> list[str]:
        return sorted(self._registry)

    def open(
        self,
        config: SourceInput,
        *,
        engine: str = "auto",
        pipeline: TransformPipeline | None = None,
    ) -> SourcePlan:
        """Select and open a provider for ``config``."""
        provider = self._resolve(config, engine)
        plan = provider().open(config)
        if pipeline is not None and len(pipeline.steps) > 0:
            plan = plan.apply(pipeline)
        return plan

    def _resolve(self, config: SourceInput, engine: str) -> type[SourceProvider]:
        if engine == "auto":
            return self._auto_select(config)
        return self._explicit_select(config, engine)

    def _auto_select(self, config: SourceInput) -> type[SourceProvider]:
        source_kind = _source_kind(config)
        # source_kind already encodes dialect for SQL types (e.g. "sql_clickhouse")
        chain = self.strategy.get((source_kind, None), ())
        if not chain:
            raise ValueError(
                f"No strategy entry for source_kind={source_kind!r}"
            )
        reasons: list[str] = []
        for pid in chain:
            cls = self._registry.get(pid)
            if cls is None:
                reasons.append(f"{pid}: not registered")
                continue
            if not cls.can_handle(config):
                reasons.append(f"{pid}: can_handle=False")
                continue
            avail = cls.availability(config)
            if not avail.is_available:
                reasons.append(f"{pid}: {avail.reason or 'unavailable'}")
                continue
            return cls
        raise ValueError(
            f"No available provider for source_kind={source_kind!r}. "
            + "; ".join(reasons)
        )

    def _explicit_select(self, config: SourceInput, engine: str) -> type[SourceProvider]:
        # Try exact provider_id first, then alias
        cls = self._registry.get(engine)
        if cls is None:
            # Look up by alias
            matches = [
                c for c in self._registry.values()
                if c.identity().selector_alias == engine and c.can_handle(config)
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous alias {engine!r}: matches {[c.identity().provider_id for c in matches]}"
                )
            if not matches:
                raise ValueError(f"No provider found for engine={engine!r}")
            cls = matches[0]

        if not cls.can_handle(config):
            raise ValueError(f"Provider {cls.identity().provider_id} cannot handle this config")
        avail = cls.availability(config)
        if not avail.is_available:
            raise ValueError(
                f"Provider {cls.identity().provider_id} is not available: {avail.reason}"
                + (f" (install: {avail.missing_extra})" if avail.missing_extra else "")
            )
        return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_kind(config: SourceInput) -> str:
    """Extract source kind from config for strategy table lookup."""
    from tributo.data.source_config import RawSourceConfig

    if isinstance(config, RawSourceConfig):
        return config.type
    type_val = getattr(config, "type", None)
    if type_val == "sql":
        dialect = getattr(config, "dialect", None)
        return f"sql_{dialect}" if dialect else "sql"
    return str(type_val) if type_val else "unknown"


# _dialect removed — dialect is encoded in _source_kind for SQL types
