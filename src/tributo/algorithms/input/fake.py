"""Deterministic in-memory input implementation for Core and Ray tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tributo._common.immutable import deep_freeze
from tributo.algorithms.api import (
    AlgorithmInputError,
    InputBinding,
    ResolvedAlgorithmPlan,
    ResolvedInputDescriptor,
    RuntimeTopology,
    canonical_digest,
)
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi.input import (
    InputExecutionContext,
    InputResolutionContext,
    PreparedInput,
    ResolvedInputLease,
    RuntimeInputBinding,
    WorkerInputPayload,
)
from tributo.util.annotations import DeveloperAPI

FAKE_RESOLVER_ID = "tributo.fake_tabular"


@DeveloperAPI
@dataclass(frozen=True)
class FakeTabularPayload:
    """Portable columnar payload used only by tests and development jobs."""

    columns_by_name: Mapping[str, tuple[object, ...]] = field(repr=False)

    def __post_init__(self) -> None:
        try:
            frozen = deep_freeze(self.columns_by_name)
        except TypeError as exc:
            raise AlgorithmInputError(
                "fake input accepts only portable scalar column values"
            ) from exc
        object.__setattr__(self, "columns_by_name", frozen)
        lengths = {len(values) for values in self.columns_by_name.values()}
        if not self.columns_by_name or len(lengths) != 1:
            raise AlgorithmInputError(
                "fake tabular columns must be non-empty and have equal lengths"
            )

    @property
    def row_count(self) -> int:
        """Return the number of rows in the payload."""
        return len(next(iter(self.columns_by_name.values())))


@DeveloperAPI
@dataclass(frozen=True)
class FakeInputInvocation:
    """Invocation-scoped payload plus Driver lifecycle probes."""

    payload: FakeTabularPayload
    close_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    cancel_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )


@DeveloperAPI
class FakeInputResolver:
    """Side-effect-free resolver used to prove Core/data decoupling."""

    @property
    def resolver_id(self) -> str:
        """Return the resolver identity used by InputBinding."""
        return FAKE_RESOLVER_ID

    def describe(
        self,
        binding: InputBinding,
        context: InputResolutionContext,
    ) -> ResolvedInputDescriptor:
        """Describe a binding without reading its invocation payload."""
        del context
        if binding.resolver_id != self.resolver_id:
            raise AlgorithmInputError(
                f"FakeInputResolver cannot resolve {binding.resolver_id!r}"
            )
        return ResolvedInputDescriptor(
            resolver_id=self.resolver_id,
            reference=binding.reference,
            descriptor_version=1,
            binding_digest=canonical_digest(binding.descriptor_payload()),
            engine_id="tributo.fake_tabular",
            view_kind="materialized_tabular",
            input_capabilities=("materializable", "shardable"),
            deferred_validations=("payload_schema", "row_count"),
            compatible_worker_input_adapter_refs=(
                "tributo.algorithms.input.fake:prepare_input",
            ),
        )

    def open(
        self,
        binding: InputBinding,
        descriptor: ResolvedInputDescriptor,
        context: InputExecutionContext,
    ) -> ResolvedInputLease:
        """Open the invocation payload after all descriptor drift checks."""
        expected = self.describe(binding, InputResolutionContext())
        if descriptor != expected:
            raise AlgorithmInputError(
                "input binding or descriptor drifted after planning"
            )
        value = context.values.get(binding.reference)
        if not isinstance(value, FakeInputInvocation):
            raise AlgorithmInputError(
                f"missing fake input invocation for {binding.reference!r}"
            )
        payload = value.payload
        required = set(binding.feature_names)
        if binding.label_name is not None:
            required.add(binding.label_name)
        if binding.sample_weight_name is not None:
            required.add(binding.sample_weight_name)
        missing = sorted(required - set(payload.columns_by_name))
        if missing:
            raise AlgorithmInputError(
                f"fake input is missing required column(s): {missing}"
            )
        return ResolvedInputLease(
            handle=payload,
            provenance={
                "resolver_id": self.resolver_id,
                "reference": binding.reference,
                "binding_digest": descriptor.binding_digest,
                "row_count": payload.row_count,
            },
            close_callback=value.close_callback,
            cancel_callback=value.cancel_callback,
        )


@DeveloperAPI
class FakeInputRuntimeAdapter:
    """Driver adapter for the in-memory test payload."""

    def bind(
        self,
        lease: ResolvedInputLease,
        plan: ResolvedAlgorithmPlan,
    ) -> RuntimeInputBinding:
        """Create a serializable Worker payload without closing the lease."""
        if not isinstance(lease.handle, FakeTabularPayload):
            raise AlgorithmInputError(
                "FakeInputRuntimeAdapter requires FakeTabularPayload"
            )
        binding = plan.input_binding
        partition_count = (
            plan.runtime.worker_count
            if plan.runtime.topology
            in {
                RuntimeTopology.DATA_PARALLEL,
                RuntimeTopology.RAY_MAP_REDUCE,
                RuntimeTopology.RAY_TRAIN_COLLECTIVE,
            }
            else 1
        )
        if lease.handle.row_count < partition_count:
            raise AlgorithmInputError(
                "fake input has fewer rows than the requested Worker count"
            )
        payloads = tuple(
            WorkerInputPayload(
                input_name=binding.name,
                binding=binding,
                value=_partition_payload(lease.handle, rank, partition_count),
                partition_index=rank,
                partition_count=partition_count,
                expected_total_rows=lease.handle.row_count,
            )
            for rank in range(partition_count)
        )
        return RuntimeInputBinding(payloads)


def _partition_payload(
    payload: FakeTabularPayload,
    rank: int,
    world_size: int,
) -> FakeTabularPayload:
    """Return one deterministic contiguous shard without overlap or row loss."""
    start = payload.row_count * rank // world_size
    stop = payload.row_count * (rank + 1) // world_size
    return FakeTabularPayload(
        {
            name: tuple(values[start:stop])
            for name, values in payload.columns_by_name.items()
        }
    )


@DeveloperAPI
def prepare_input(payload: WorkerInputPayload) -> PreparedInput:
    """Create a Worker-owned materialized tabular view."""
    if not isinstance(payload.value, FakeTabularPayload):
        raise AlgorithmInputError(
            "fake Worker input adapter requires FakeTabularPayload"
        )
    view = InMemoryTabularInputView(
        _columns=payload.value.columns_by_name,
        feature_names=payload.binding.feature_names,
        label_name=payload.binding.label_name,
    )
    return PreparedInput({payload.input_name: view})


__all__ = [
    "FakeInputInvocation",
    "FakeInputResolver",
    "FakeInputRuntimeAdapter",
    "FakeTabularPayload",
    "InMemoryTabularInputView",
    "prepare_input",
]
