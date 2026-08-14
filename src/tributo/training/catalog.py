"""AlgorithmCatalog — stateless read-only view over the trainer Registry.

Provides multi-dimensional filtering (problem type, modality, tag,
extras group), lifecycle-aware lookups, config schema generation,
and replacement-graph integrity validation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    PROBLEM_FAMILY_MAP,
    AlgorithmSpec,
    AlgorithmStatus,
    Capability,
    ExecutionKind,
    ProblemFamily,
    ProblemType,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI


class _CatalogRegistry(Protocol):
    def snapshot(self) -> Mapping[str, AlgorithmSpec]: ...

    def get(self, key: str) -> AlgorithmSpec: ...


@DeveloperAPI
@dataclass(frozen=True)
class AlgorithmCatalogRecord:
    """Read-only availability and support projection for one algorithm."""

    name: str
    spec: AlgorithmSpec | None
    implementation_ids: tuple[str, ...]
    runtime_topologies: tuple[str, ...]
    distribution_strategies: tuple[str, ...]
    execution_profiles: tuple[str, ...]
    input_views: tuple[str, ...]
    stability: str
    available: bool
    compatibility_only: bool
    tested: bool
    supported: bool
    validated_execution_profiles: tuple[str, ...]
    native_migration_complete: bool
    limitations: tuple[str, ...]


# ---------------------------------------------------------------------------
# AlgorithmCatalog
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class AlgorithmCatalog:
    """Stateless read-only view over the trainer Registry.

    Every public query calls ``self._registry.snapshot()`` internally,
    so runtime registrations are immediately visible without cache
    invalidation.
    """

    def __init__(self, registry: _CatalogRegistry) -> None:
        self._registry = registry

    # -- query ---------------------------------------------------------------

    def list(  # noqa: A003
        self,
        *,
        problem_type: ProblemType | None = None,
        problem_family: ProblemFamily | None = None,
        modality: str | None = None,
        tag: str | None = None,
        extras_group: str | None = None,
        execution_kind: ExecutionKind | None = None,
        capabilities: Capability | tuple[Capability, ...] | None = None,
        include_deprecated: bool = False,
    ) -> list[str]:
        """Return algorithm names matching all filter criteria (intersection).

        Filters are applied conjunctively — an algorithm must satisfy
        every non-None filter to be included.

        Args:
            execution_kind: Filter by ``ExecutionKind`` (e.g. ``TRAIN``,
                ``ESTIMATE``).
            capabilities: Filter by one or more ``Capability`` tags.  When
                multiple capabilities are given, the algorithm must have
                **all** of them (conjunction).
        """
        return [
            spec.name
            for spec in self.list_specs(
                problem_type=problem_type,
                problem_family=problem_family,
                modality=modality,
                tag=tag,
                extras_group=extras_group,
                execution_kind=execution_kind,
                capabilities=capabilities,
                include_deprecated=include_deprecated,
            )
        ]

    def list_specs(
        self,
        *,
        problem_type: ProblemType | None = None,
        problem_family: ProblemFamily | None = None,
        modality: str | None = None,
        tag: str | None = None,
        extras_group: str | None = None,
        execution_kind: ExecutionKind | None = None,
        capabilities: Capability | tuple[Capability, ...] | None = None,
        include_deprecated: bool = False,
    ) -> tuple[AlgorithmSpec, ...]:
        """Return matching specs from one immutable Registry snapshot.

        Unlike a ``list()`` followed by repeated ``get_spec()`` calls, this
        method cannot combine algorithm facts from different Registry states.
        """
        snapshot = self._registry.snapshot()
        self._validate_integrity(snapshot)

        # Normalise capabilities to a tuple of Capability enums for
        # consistent iteration.  Plain strings (natural for str-Enum)
        # are coerced so that `catalog.list(capabilities="tunable")`
        # works as expected.
        cap_set: tuple[Capability, ...] = ()
        if isinstance(capabilities, Capability):
            cap_set = (capabilities,)
        elif isinstance(capabilities, str):
            cap_set = (Capability(capabilities),)
        elif capabilities is not None:
            # Accept (Capability, ...) or list of strings/instances.
            cap_set = tuple(
                Capability(c) if isinstance(c, str) else c for c in capabilities
            )

        results: list[AlgorithmSpec] = []
        for spec in snapshot.values():
            if problem_type is not None and problem_type not in spec.problem_types:
                continue
            if problem_family is not None:
                expected = PROBLEM_FAMILY_MAP[problem_family]
                if not any(pt in spec.problem_types for pt in expected):
                    continue
            if modality is not None and modality not in spec.data_modality:
                continue
            if tag is not None and tag not in spec.tags:
                continue
            if extras_group is not None and spec.extras_group != extras_group:
                continue
            if execution_kind is not None and spec.execution_kind != execution_kind:
                continue
            if cap_set and not all(c in spec.capabilities for c in cap_set):
                continue
            if not include_deprecated and spec.status == AlgorithmStatus.DEPRECATED:
                continue
            results.append(spec)
        return tuple(results)

    def list_records(
        self,
        *,
        problem_type: ProblemType | None = None,
        problem_family: ProblemFamily | None = None,
        modality: str | None = None,
        tag: str | None = None,
        extras_group: str | None = None,
        include_deprecated: bool = False,
    ) -> tuple[AlgorithmCatalogRecord, ...]:
        """Return support-aware records from one immutable Registry projection."""
        record_snapshot = getattr(self._registry, "record_snapshot", None)
        if callable(record_snapshot):
            source = record_snapshot()
            records = tuple(
                AlgorithmCatalogRecord(
                    name=entry.name,
                    spec=entry.spec,
                    implementation_ids=tuple(
                        registration.implementation.implementation_id
                        for registration in entry.registrations
                    ),
                    runtime_topologies=tuple(
                        sorted(
                            {
                                (
                                    registration.runtime.topology.value
                                    if registration.runtime is not None
                                    else registration.distribution_spec.strategy.value
                                )
                                for registration in entry.registrations
                            }
                        )
                    ),
                    distribution_strategies=tuple(
                        sorted(
                            {
                                registration.distribution_spec.strategy.value
                                for registration in entry.registrations
                                if registration.distribution_spec is not None
                            }
                        )
                    ),
                    execution_profiles=tuple(
                        sorted(
                            {
                                profile.value
                                for registration in entry.registrations
                                if registration.distribution_spec is not None
                                for profile in registration.distribution_spec.supported_execution_profiles
                            }
                        )
                    ),
                    input_views=tuple(
                        sorted(
                            {
                                view
                                for registration in entry.registrations
                                for view in registration.implementation.input_compatibility.accepted_input_views
                            }
                        )
                    ),
                    stability=entry.stability,
                    available=entry.available,
                    compatibility_only=entry.compatibility_only,
                    tested=entry.tested,
                    supported=entry.supported,
                    validated_execution_profiles=entry.validated_execution_profiles,
                    native_migration_complete=entry.native_migration_complete,
                    limitations=entry.limitations,
                )
                for entry in source
            )
        else:
            # Preserve the Beta constructor's generic Registry compatibility.
            # Without a unified record snapshot, this path cannot prove that an
            # executable portable descriptor exists.
            records = tuple(
                AlgorithmCatalogRecord(
                    name=spec.name,
                    spec=spec,
                    implementation_ids=(),
                    runtime_topologies=(),
                    distribution_strategies=(),
                    execution_profiles=(),
                    input_views=(),
                    stability="beta",
                    available=False,
                    compatibility_only=True,
                    tested=False,
                    supported=False,
                    validated_execution_profiles=(),
                    native_migration_complete=False,
                    limitations=(
                        "Generic Beta Registry entry has no executable portable descriptor.",
                    ),
                )
                for spec in self._registry.snapshot().values()
            )
        filtered: list[AlgorithmCatalogRecord] = []
        for record in records:
            spec = record.spec
            if spec is None:
                if any(
                    value is not None
                    for value in (
                        problem_type,
                        problem_family,
                        modality,
                        tag,
                        extras_group,
                    )
                ):
                    continue
                filtered.append(record)
                continue
            if problem_type is not None and problem_type not in spec.problem_types:
                continue
            if problem_family is not None and not any(
                item in PROBLEM_FAMILY_MAP[problem_family]
                for item in spec.problem_types
            ):
                continue
            if modality is not None and modality not in spec.data_modality:
                continue
            if tag is not None and tag not in spec.tags:
                continue
            if extras_group is not None and spec.extras_group != extras_group:
                continue
            if not include_deprecated and spec.status is AlgorithmStatus.DEPRECATED:
                continue
            filtered.append(record)
        return tuple(sorted(filtered, key=lambda item: item.name))

    def get_record(self, name: str) -> AlgorithmCatalogRecord:
        """Return one support-aware record without hydrating implementation code."""
        records = {
            record.name: record for record in self.list_records(include_deprecated=True)
        }
        try:
            return records[name]
        except KeyError as exc:
            raise JobConfigurationError(
                f"Unknown algorithm: {name!r}. Available: {sorted(records)}"
            ) from exc

    def get_spec(self, name: str) -> AlgorithmSpec:
        """Return the ``AlgorithmSpec`` for *name*.

        Emits ``FutureWarning`` when the algorithm is deprecated.
        """
        snapshot = self._registry.snapshot()
        resolved = self._validate_integrity(snapshot)

        if name not in snapshot:
            raise JobConfigurationError(
                f"Unknown algorithm: {name!r}. Available: {sorted(snapshot.keys())}"
            )
        spec = self._registry.get(name)
        if spec.status == AlgorithmStatus.DEPRECATED:
            replacement = resolved[name]
            warnings.warn(
                f"Algorithm {name!r} is deprecated "
                f"since {spec.deprecated_since}. "
                f"Use {replacement!r} instead.",
                FutureWarning,
                stacklevel=2,
            )
        return spec

    def get_config_schema(self, name: str) -> dict[str, Any]:
        """Return the JSON Schema for *name*'s ``config_model``.

        Raises:
            JobConfigurationError: The algorithm has no ``config_model``.
        """
        spec = self.get_spec(name)
        if spec.config_model is None:
            raise JobConfigurationError(
                f"Algorithm {name!r} does not declare a config_model. "
                f"Cannot generate config schema."
            )
        return spec.config_model.model_json_schema()

    def supports_classification(self, name: str) -> bool:
        """Return True if *name* supports any classification problem type."""
        spec = self.get_spec(name)
        classification_types = PROBLEM_FAMILY_MAP[ProblemFamily.CLASSIFICATION]
        return any(pt in spec.problem_types for pt in classification_types)

    def supports_regression(self, name: str) -> bool:
        """Return True if *name* supports regression."""
        spec = self.get_spec(name)
        return ProblemType.REGRESSION in spec.problem_types

    def requires_gpu(self, name: str) -> bool:
        """Return True if *name* requires a GPU."""
        spec = self.get_spec(name)
        return spec.resource_hints.gpu_required

    # -- integrity -----------------------------------------------------------

    def validate_integrity(self) -> None:
        """Validate the replacement graph on the current snapshot.

        Raises:
            JobConfigurationError: A replacement is missing, a cycle
                exists, or a deprecated chain does not end at READY.
        """
        self._validate_integrity(self._registry.snapshot())

    def _validate_integrity(
        self,
        snapshot: Mapping[str, AlgorithmSpec],
    ) -> dict[str, str]:
        """Validate the replacement graph and return ``{deprecated → final_READY}``.

        Checks:
        1. Every ``replacement`` must exist in the registry.
        2. No cycles in the replacement graph (DFS with three-colour marking).
        3. Every deprecated chain must ultimately reach a READY algorithm.
        """
        resolved: dict[str, str] = {}  # deprecated name → final READY name

        # Check 1 — existence
        for name, spec in snapshot.items():
            if spec.status == AlgorithmStatus.DEPRECATED:
                replacement = spec.replacement
                assert replacement is not None  # __post_init__ guarantee
                if replacement not in snapshot:
                    raise JobConfigurationError(
                        f"Algorithm {name!r}: replacement {replacement!r} "
                        f"not found in registry."
                    )

        # Check 2 — cycle detection (three-colour DFS)
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = dict.fromkeys(snapshot, WHITE)

        def _resolve_chain(start: str) -> str:
            """Follow replacement chain from *start* to a READY endpoint."""
            if start in resolved:
                return resolved[start]
            spec = snapshot[start]
            if spec.status == AlgorithmStatus.READY:
                return start
            # DEPRECATED → follow replacement
            next_name = spec.replacement
            assert next_name is not None
            if colour[start] == GREY:
                raise JobConfigurationError(
                    f"Replacement cycle detected involving {start!r}"
                )
            colour[start] = GREY
            endpoint = _resolve_chain(next_name)
            colour[start] = BLACK
            resolved[start] = endpoint
            return endpoint

        for name in snapshot:
            if colour[name] == WHITE:
                try:
                    endpoint = _resolve_chain(name)
                except RecursionError:
                    raise JobConfigurationError(
                        f"Replacement chain too deep near {name!r}"
                    ) from None
                if endpoint is not None:
                    resolved[name] = endpoint

        return resolved


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
def get_algorithm_catalog() -> AlgorithmCatalog:
    """Create a stateless view over the shared trainer Registry.

    Multiple calls may return different ``AlgorithmCatalog`` instances,
    but all share the same underlying Registry — queries are always
    consistent and live.
    """
    from tributo.training.registry import _registry

    return AlgorithmCatalog(_registry)


__all__ = ["AlgorithmCatalog", "AlgorithmCatalogRecord", "get_algorithm_catalog"]
