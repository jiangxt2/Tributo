"""Input ports and process-boundary ownership contracts."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Protocol, runtime_checkable

from tributo._common.immutable import deep_freeze
from tributo.algorithms.api import (
    AlgorithmInputError,
    InputBinding,
    ResolvedAlgorithmPlan,
    ResolvedInputDescriptor,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class InputResolutionContext:
    """Planning metadata plus opaque invocation values for an input resolver.

    Only ``metadata`` is portable. ``values`` may contain trusted request bodies
    and is deliberately excluded from plan serialization, repr, and equality.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict)
    values: Mapping[str, object] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
        object.__setattr__(self, "values", _freeze_invocation_values(self.values))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class InputExecutionContext:
    """Ephemeral invocation data used only while opening an input lease."""

    values: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_invocation_values(self.values))


def _freeze_invocation_values(values: Mapping[str, object]) -> Mapping[str, object]:
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError("invocation context keys must be non-empty strings")
    return MappingProxyType(dict(values))


@PublicAPI(stability="alpha")
class ResolvedInputLease:
    """Driver-owned runtime input and its exactly-once lifecycle delegate."""

    def __init__(
        self,
        *,
        handle: object,
        provenance: Mapping[str, Any],
        binding: InputBinding | None = None,
        close_callback: Callable[[], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> None:
        self.handle = handle
        self.provenance = deep_freeze(provenance)
        if binding is not None and not isinstance(binding, InputBinding):
            raise TypeError("resolved input lease binding must be InputBinding")
        self.binding = binding
        self._close_callback = close_callback
        self._cancel_callback = cancel_callback
        self._closed = False
        self._lock = threading.Lock()

    def attach_binding(self, binding: InputBinding) -> None:
        """Attach the role binding once for backward-compatible resolvers."""
        if not isinstance(binding, InputBinding):
            raise TypeError("resolved input lease binding must be InputBinding")
        with self._lock:
            if self.binding is not None and self.binding != binding:
                raise AlgorithmInputError("resolved input lease binding drifted")
            self.binding = binding

    @property
    def closed(self) -> bool:
        """Return whether close or cancel has claimed this lease."""
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Release the lease once; repeated calls are no-ops."""
        self._finish(cancel=False)

    def cancel(self) -> None:
        """Cancel and release the lease once; repeated calls are no-ops."""
        self._finish(cancel=True)

    def _finish(self, *, cancel: bool) -> None:
        callback: Callable[[], None] | None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            callback = (
                self._cancel_callback or self._close_callback
                if cancel
                else self._close_callback
            )
        if callback is not None:
            callback()


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class WorkerInputPayload:
    """Serializable runtime payload sent to a Worker input adapter."""

    input_name: str
    binding: InputBinding
    value: object = field(repr=False)
    partition_index: int = 0
    partition_count: int = 1
    expected_total_rows: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_name, str) or not self.input_name:
            raise AlgorithmInputError("Worker input name must be non-empty")
        if (
            not isinstance(self.partition_index, int)
            or isinstance(self.partition_index, bool)
            or not isinstance(self.partition_count, int)
            or isinstance(self.partition_count, bool)
            or self.partition_count < 1
            or self.partition_index < 0
            or self.partition_index >= self.partition_count
        ):
            raise AlgorithmInputError("Worker input partition is invalid")
        if self.expected_total_rows is not None and (
            not isinstance(self.expected_total_rows, int)
            or isinstance(self.expected_total_rows, bool)
            or self.expected_total_rows < 0
        ):
            raise AlgorithmInputError(
                "Worker input expected_total_rows must be a non-negative integer"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class WorkerInputPayloadSet:
    """Serializable role-keyed payloads assigned to one Worker rank."""

    payloads: tuple[WorkerInputPayload, ...]
    primary_role: str = "train"

    def __post_init__(self) -> None:
        payloads = tuple(self.payloads)
        if not payloads or any(
            not isinstance(item, WorkerInputPayload) for item in payloads
        ):
            raise AlgorithmInputError(
                "Worker input payload sets require WorkerInputPayload values"
            )
        roles = tuple(item.input_name for item in payloads)
        if len(set(roles)) != len(roles):
            raise AlgorithmInputError("Worker input payload roles must be unique")
        if self.primary_role not in roles:
            raise AlgorithmInputError(
                "Worker input payload primary_role must reference a payload"
            )
        partition_coordinates = {
            (item.partition_index, item.partition_count) for item in payloads
        }
        if len(partition_coordinates) != 1:
            raise AlgorithmInputError(
                "Worker input roles must share one partition coordinate"
            )
        object.__setattr__(self, "payloads", payloads)

    @property
    def partition_index(self) -> int:
        """Return the shared Worker partition index."""
        return self.payloads[0].partition_index

    @property
    def partition_count(self) -> int:
        """Return the shared Worker partition count."""
        return self.payloads[0].partition_count

    @property
    def expected_total_rows(self) -> int | None:
        """Return primary-role row coverage for compatibility checks."""
        return self.get(self.primary_role).expected_total_rows

    def get(self, role: str) -> WorkerInputPayload:
        """Return one role payload."""
        for payload in self.payloads:
            if payload.input_name == role:
                return payload
        raise AlgorithmInputError(f"unknown Worker input role: {role!r}")


@PublicAPI(stability="alpha")
class RuntimeInputBinding:
    """Driver-owned handoff from an input lease to a Runtime Adapter."""

    def __init__(
        self,
        payloads: (
            WorkerInputPayload
            | WorkerInputPayloadSet
            | tuple[WorkerInputPayload | WorkerInputPayloadSet, ...]
        ),
    ) -> None:
        normalized = payloads if isinstance(payloads, tuple) else (payloads,)
        if not normalized or any(
            not isinstance(payload, (WorkerInputPayload, WorkerInputPayloadSet))
            for payload in normalized
        ):
            raise AlgorithmInputError(
                "runtime input binding requires Worker input payload values"
            )
        partition_count = len(normalized)
        if any(payload.partition_count != partition_count for payload in normalized):
            raise AlgorithmInputError(
                "runtime input payloads disagree on partition count"
            )
        if {payload.partition_index for payload in normalized} != set(
            range(partition_count)
        ):
            raise AlgorithmInputError(
                "runtime input payloads must contain every partition exactly once"
            )
        expected_totals = {payload.expected_total_rows for payload in normalized}
        if len(expected_totals) > 1:
            raise AlgorithmInputError(
                "runtime input payloads disagree on expected total rows"
            )
        self.payloads = tuple(sorted(normalized, key=lambda item: item.partition_index))
        self._closed = False
        self._lock = threading.Lock()

    @property
    def payload(self) -> WorkerInputPayload | WorkerInputPayloadSet:
        """Return the only payload for a single-Worker binding."""
        if len(self.payloads) != 1:
            raise AlgorithmInputError(
                "data-parallel runtime input has more than one Worker payload"
            )
        return self.payloads[0]

    @property
    def closed(self) -> bool:
        """Return whether the Driver released this binding."""
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Release Driver-side handoff state exactly once."""
        with self._lock:
            self._closed = True


@PublicAPI(stability="alpha")
class PreparedInput:
    """Worker-owned input views and their exactly-once cleanup callback."""

    def __init__(
        self,
        views: Mapping[str, object],
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self.views = dict(views)
        self._close_callback = close_callback
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        """Return whether Worker cleanup has completed."""
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Release Worker-native views exactly once."""
        callback: Callable[[], None] | None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            callback = self._close_callback
        if callback is not None:
            callback()


@PublicAPI(stability="alpha")
@runtime_checkable
class MaterializedTabularInputView(Protocol):
    """Small-data view required by the single-Worker sklearn reference path."""

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def label_name(self) -> str | None: ...

    @property
    def row_count(self) -> int: ...

    def columns(self) -> Mapping[str, tuple[object, ...]]: ...


@PublicAPI(stability="alpha")
@runtime_checkable
class TabularBatchInputView(Protocol):
    """Streaming tabular shard consumed without Worker-side materialization."""

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def label_name(self) -> str | None: ...

    def iter_batches(self) -> Iterable[Mapping[str, object]]: ...


@PublicAPI(stability="alpha")
class InputResolverPort(Protocol):
    """Core-facing two-stage input resolution port."""

    @property
    def resolver_id(self) -> str: ...

    def describe(
        self,
        binding: InputBinding,
        context: InputResolutionContext,
    ) -> ResolvedInputDescriptor: ...

    def open(
        self,
        binding: InputBinding,
        descriptor: ResolvedInputDescriptor,
        context: InputExecutionContext,
    ) -> ResolvedInputLease: ...


@PublicAPI(stability="alpha")
class InputRuntimeAdapter(Protocol):
    """Driver-side conversion from an abstract lease to a Worker payload."""

    def bind(
        self,
        lease: ResolvedInputLease,
        plan: ResolvedAlgorithmPlan,
    ) -> RuntimeInputBinding: ...


@PublicAPI(stability="alpha")
class WorkerInputAdapter(Protocol):
    """Worker-side conversion from a payload to framework-neutral views."""

    def __call__(
        self,
        payload: WorkerInputPayload | WorkerInputPayloadSet,
    ) -> PreparedInput: ...


__all__ = [
    "InputExecutionContext",
    "InputResolutionContext",
    "InputResolverPort",
    "InputRuntimeAdapter",
    "MaterializedTabularInputView",
    "PreparedInput",
    "ResolvedInputLease",
    "RuntimeInputBinding",
    "TabularBatchInputView",
    "WorkerInputAdapter",
    "WorkerInputPayload",
    "WorkerInputPayloadSet",
]
