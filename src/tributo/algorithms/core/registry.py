"""Deterministic programmatic registry for portable algorithms."""

from __future__ import annotations

import threading
import warnings

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmResolutionError,
)
from tributo.training.algorithm_spec import AlgorithmStatus
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class AlgorithmRegistrationRegistry:
    """Store immutable registrations without plugin discovery side effects."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], AlgorithmRegistration] = {}
        self._specs: dict[str, object] = {}
        self._lock = threading.Lock()

    def register(self, registration: AlgorithmRegistration) -> None:
        """Register one implementation and reject duplicate or split facts."""
        key = (registration.spec.name, registration.implementation.implementation_id)
        with self._lock:
            if key in self._registrations:
                raise AlgorithmResolutionError(
                    f"algorithm implementation already registered: {key!r}"
                )
            existing_spec = self._specs.get(registration.spec.name)
            if existing_spec is not None and existing_spec != registration.spec:
                raise AlgorithmResolutionError(
                    f"algorithm {registration.spec.name!r} has conflicting specs"
                )
            self._specs[registration.spec.name] = registration.spec
            self._registrations[key] = registration

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


__all__ = ["AlgorithmRegistrationRegistry"]
