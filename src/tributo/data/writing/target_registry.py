"""Registry for engine-neutral bounded-write target planners."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from tributo.data.writing.contracts import WriteCapabilityError, WriteRequest
from tributo.data.writing.targets import GenericWriteTargetProvider, WriteTargetProvider
from tributo.util.annotations import DeveloperAPI

WriteTargetProviderFactory = Callable[[], WriteTargetProvider]


def _make_builtin_factory(target_kind: str) -> WriteTargetProviderFactory:
    def factory() -> WriteTargetProvider:
        return GenericWriteTargetProvider(target_kind)

    return factory


@dataclass(frozen=True)
class RegisteredWriteTarget:
    """Registered provider factory for one target kind."""

    target_kind: str
    factory: WriteTargetProviderFactory


@DeveloperAPI
class WriteTargetRegistry:
    """Resolve target semantics independently from engine bindings."""

    def __init__(self, *, register_builtin_formats: bool = True) -> None:
        self._targets: dict[str, RegisteredWriteTarget] = {}
        self._lock = threading.RLock()
        if register_builtin_formats:
            for target_kind in ("parquet", "csv", "iceberg", "lance"):
                self.register(
                    target_kind,
                    _make_builtin_factory(target_kind),
                )

    def register(self, target_kind: str, factory: WriteTargetProviderFactory) -> None:
        """Register one target planner."""
        if not callable(factory):
            raise TypeError("write target provider factory must be callable")
        with self._lock:
            if target_kind in self._targets:
                raise WriteCapabilityError(
                    f"Write target {target_kind!r} is already registered"
                )
            self._targets[target_kind] = RegisteredWriteTarget(target_kind, factory)

    def resolve(self, request: WriteRequest) -> RegisteredWriteTarget:
        """Resolve the exact target planner or fail closed."""
        with self._lock:
            target = self._targets.get(request.target_kind)
        if target is None:
            raise WriteCapabilityError(
                f"No write target provider matches {request.target_kind!r}"
            )
        return target
