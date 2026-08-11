"""Deterministic programmatic registry for portable algorithms."""

from __future__ import annotations

import threading
import warnings

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmResolutionError,
)
from tributo.training.algorithm_spec import AlgorithmSpec, AlgorithmStatus
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class AlgorithmRegistrationRegistry:
    """Store immutable registrations without plugin discovery side effects."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], AlgorithmRegistration] = {}
        self._specs: dict[str, AlgorithmSpec] = {}
        self._compatibility_specs: dict[str, AlgorithmSpec] = {}
        self._lock = threading.Lock()

    def register(self, registration: AlgorithmRegistration) -> None:
        """Register one implementation and reject duplicate or split facts."""
        with self._lock:
            if registration.spec.name in self._compatibility_specs:
                raise AlgorithmResolutionError(
                    f"algorithm {registration.spec.name!r} already has a "
                    "compatibility-only fact"
                )
            self._validate_and_add(
                registration,
                registrations=self._registrations,
                specs=self._specs,
            )

    def register_many(
        self,
        registrations: tuple[AlgorithmRegistration, ...],
    ) -> None:
        """Validate and atomically publish a deterministic registration batch.

        Unlike repeated :meth:`register` calls, a failed batch never leaves a
        partially visible Registry. The completed candidate state must also be
        unambiguous for every algorithm operation offered by multiple
        implementations.
        """
        batch = tuple(registrations)
        if any(not isinstance(item, AlgorithmRegistration) for item in batch):
            raise TypeError("registration batches require AlgorithmRegistration values")
        if not batch:
            return
        with self._lock:
            candidate_registrations = dict(self._registrations)
            candidate_specs = dict(self._specs)
            for registration in batch:
                if registration.spec.name in self._compatibility_specs:
                    raise AlgorithmResolutionError(
                        f"algorithm {registration.spec.name!r} already has a "
                        "compatibility-only fact"
                    )
                self._validate_and_add(
                    registration,
                    registrations=candidate_registrations,
                    specs=candidate_specs,
                )
            self._validate_default_selection(candidate_registrations)
            self._registrations = candidate_registrations
            self._specs = candidate_specs

    def register_compatibility(self, spec: AlgorithmSpec) -> None:
        """Register a Beta-only fact that has no executable implementation."""
        with self._lock:
            if spec.name in self._specs or spec.name in self._compatibility_specs:
                raise AlgorithmResolutionError(
                    f"algorithm fact already registered: {spec.name!r}"
                )
            self._compatibility_specs[spec.name] = spec

    def unregister_compatibility(self, algorithm: str) -> None:
        """Remove one compatibility-only fact, primarily for Beta teardown."""
        with self._lock:
            self._compatibility_specs.pop(algorithm, None)

    def get_compatibility(self, algorithm: str) -> AlgorithmSpec | None:
        """Return one compatibility-only fact without consulting executions."""
        with self._lock:
            return self._compatibility_specs.get(algorithm)

    @staticmethod
    def _validate_and_add(
        registration: AlgorithmRegistration,
        *,
        registrations: dict[tuple[str, str], AlgorithmRegistration],
        specs: dict[str, AlgorithmSpec],
    ) -> None:
        key = (registration.spec.name, registration.implementation.implementation_id)
        if key in registrations:
            raise AlgorithmResolutionError(
                f"algorithm implementation already registered: {key!r}"
            )
        existing_spec = specs.get(registration.spec.name)
        if existing_spec is not None and existing_spec != registration.spec:
            raise AlgorithmResolutionError(
                f"algorithm {registration.spec.name!r} has conflicting specs"
            )
        specs[registration.spec.name] = registration.spec
        registrations[key] = registration

    @staticmethod
    def _validate_default_selection(
        registrations: dict[tuple[str, str], AlgorithmRegistration],
    ) -> None:
        candidates: dict[
            tuple[str, AlgorithmOperation], list[AlgorithmRegistration]
        ] = {}
        for registration in registrations.values():
            for operation in registration.implementation.operations:
                candidates.setdefault((registration.spec.name, operation), []).append(
                    registration
                )
        for (algorithm, operation), implementations in candidates.items():
            if len(implementations) < 2:
                continue
            defaults = [item for item in implementations if item.is_default]
            if len(defaults) != 1:
                identities = sorted(
                    item.implementation.implementation_id for item in implementations
                )
                raise AlgorithmResolutionError(
                    f"algorithm {algorithm!r}, operation {operation.value!r} has "
                    f"ambiguous implementations: {identities}"
                )

    def unregister(self, algorithm: str, implementation_id: str) -> None:
        """Remove one provisional registration, primarily for test teardown."""
        key = (algorithm, implementation_id)
        with self._lock:
            self._registrations.pop(key, None)
            if not any(name == algorithm for name, _ in self._registrations):
                self._specs.pop(algorithm, None)

    def resolve(
        self,
        *,
        algorithm: str,
        operation: AlgorithmOperation,
        implementation_id: str | None,
    ) -> AlgorithmRegistration:
        """Select one eligible implementation independently of registration order."""
        with self._lock:
            candidates = [
                registration
                for (name, _), registration in self._registrations.items()
                if name == algorithm
                and operation in registration.implementation.operations
            ]
        candidates.sort(key=lambda item: item.implementation.implementation_id)
        if implementation_id is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.implementation.implementation_id == implementation_id
            ]
        if not candidates:
            suffix = (
                f" and implementation {implementation_id!r}"
                if implementation_id is not None
                else ""
            )
            raise AlgorithmResolutionError(
                f"no implementation for algorithm {algorithm!r}, "
                f"operation {operation.value!r}{suffix}"
            )
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            defaults = [candidate for candidate in candidates if candidate.is_default]
            if len(defaults) != 1:
                identities = [
                    candidate.implementation.implementation_id
                    for candidate in candidates
                ]
                raise AlgorithmResolutionError(
                    f"algorithm {algorithm!r} has ambiguous implementations: "
                    f"{identities}"
                )
            selected = defaults[0]
        if selected.spec.status is AlgorithmStatus.DEPRECATED:
            warnings.warn(
                f"Algorithm {selected.spec.name!r} is deprecated since "
                f"{selected.spec.deprecated_since}. Use "
                f"{selected.spec.replacement!r} instead.",
                FutureWarning,
                stacklevel=2,
            )
        return selected

    def snapshot(self) -> tuple[AlgorithmRegistration, ...]:
        """Return a deterministic immutable registration snapshot."""
        with self._lock:
            return tuple(value for _, value in sorted(self._registrations.items()))

    def spec_snapshot(self) -> dict[str, AlgorithmSpec]:
        """Return a deterministic copy of the canonical algorithm facts."""
        with self._lock:
            return dict(sorted(self._specs.items()))

    def compatibility_snapshot(self) -> dict[str, AlgorithmSpec]:
        """Return a deterministic copy of Beta-only compatibility facts."""
        with self._lock:
            return dict(sorted(self._compatibility_specs.items()))

    def catalog_spec_snapshot(self) -> dict[str, AlgorithmSpec]:
        """Return every canonical Catalog fact from the same Registry lock."""
        with self._lock:
            return dict(sorted({**self._specs, **self._compatibility_specs}.items()))

    def catalog_snapshot(
        self,
    ) -> tuple[
        tuple[AlgorithmRegistration, ...],
        dict[str, AlgorithmSpec],
    ]:
        """Atomically snapshot executable registrations and compatibility facts."""
        with self._lock:
            registrations = tuple(
                value for _, value in sorted(self._registrations.items())
            )
            compatibility = dict(sorted(self._compatibility_specs.items()))
        return registrations, compatibility


__all__ = ["AlgorithmRegistrationRegistry"]
