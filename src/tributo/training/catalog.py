"""AlgorithmCatalog — stateless read-only view over the trainer Registry.

Provides multi-dimensional filtering (problem type, modality, tag,
extras group), lifecycle-aware lookups, config schema generation,
and replacement-graph integrity validation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from tributo._common.registry import Registry


# ---------------------------------------------------------------------------
# AlgorithmCatalog
# ---------------------------------------------------------------------------


class AlgorithmCatalog:
    """Stateless read-only view over the trainer Registry.

    Every public query calls ``self._registry.snapshot()`` internally,
    so runtime registrations are immediately visible without cache
    invalidation.
    """

    def __init__(self, registry: Registry[str, AlgorithmSpec]) -> None:
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

        results: list[str] = []
        for name, spec in snapshot.items():
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
            results.append(name)
        return results

    def get_spec(self, name: str) -> AlgorithmSpec:
        """Return the ``AlgorithmSpec`` for *name*.

        Emits ``FutureWarning`` when the algorithm is deprecated.
        """
        snapshot = self._registry.snapshot()
        resolved = self._validate_integrity(snapshot)

        spec = snapshot.get(name)
        if spec is None:
            raise JobConfigurationError(
                f"Unknown algorithm: {name!r}. Available: {sorted(snapshot.keys())}"
            )
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


def get_algorithm_catalog() -> AlgorithmCatalog:
    """Create a stateless view over the shared trainer Registry.

    Multiple calls may return different ``AlgorithmCatalog`` instances,
    but all share the same underlying Registry — queries are always
    consistent and live.
    """
    from tributo.training.registry import _registry

    return AlgorithmCatalog(_registry)
