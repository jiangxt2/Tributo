"""Bridge the public ingestion Gateway to portable algorithm input ports."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from tributo.algorithms.api import (
    AlgorithmInputError,
    InputBinding,
    ResolvedAlgorithmPlan,
    ResolvedInputDescriptor,
    RuntimeTopology,
    canonical_digest,
)
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    PreparedInput,
    ResolvedInputLease,
    RuntimeInputBinding,
    WorkerInputPayload,
)
from tributo.data import (
    DaftDataFrameHandle,
    IngestionDescriptor,
    IngestionGateway,
    IngestionOpenResult,
    IngestionRequest,
    IngestionRuntimeContext,
    RayDataHandle,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

INGESTION_RESOLVER_ID = "tributo.ingestion"
_BRIDGE_DESCRIPTOR_VERSION = 1
_REQUEST_KEY = re.compile(r"^[a-z][a-z0-9_.-]+$")
_SUPPORTED_HANDLE_KINDS = frozenset({"ray_data", "daft_dataframe"})
_MAX_MATERIALIZED_ROWS = 1_000_000
_ADAPTER_MODULE = "tributo.integrations.algorithm_inputs.ingestion"
_GENERIC_ADAPTER_REF = f"{_ADAPTER_MODULE}:prepare_ingestion_input"
_HANDLE_ADAPTER_REFS = {
    "ray_data": f"{_ADAPTER_MODULE}:prepare_ray_data_input",
    "daft_dataframe": f"{_ADAPTER_MODULE}:prepare_daft_input",
}
HandleKind = Literal["ray_data", "daft_dataframe"]


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class IngestionRequestRef:
    """Invocation-scoped identity for a trusted immutable ingestion request."""

    request_key: str
    resolver_id: str = INGESTION_RESOLVER_ID

    def __post_init__(self) -> None:
        if self.resolver_id != INGESTION_RESOLVER_ID:
            raise AlgorithmInputError(
                f"ingestion request refs require resolver {INGESTION_RESOLVER_ID!r}"
            )
        if (
            not isinstance(self.request_key, str)
            or _REQUEST_KEY.fullmatch(self.request_key) is None
        ):
            raise AlgorithmInputError(
                "ingestion request key must be a namespaced lower-case identifier"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class IngestionInputInvocation:
    """Trusted request body and optional runtime services for one invocation."""

    request: IngestionRequest = field(repr=False)
    runtime_context: IngestionRuntimeContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request, IngestionRequest):
            raise AlgorithmInputError(
                "ingestion input invocation requires an IngestionRequest"
            )
        if self.runtime_context is not None and not isinstance(
            self.runtime_context, IngestionRuntimeContext
        ):
            raise AlgorithmInputError("ingestion runtime context has an invalid type")


class _IngestionOpenResultDelegate:
    """Keep the complete Gateway result as the sole native lifecycle owner."""

    def __init__(self, result: IngestionOpenResult) -> None:
        self._result = result

    def close(self) -> None:
        self._result.close()

    def cancel(self) -> None:
        self._result.cancel()


@dataclass(frozen=True)
class _RayDataIteratorPayload:
    """One public Ray Data iterator assigned to exactly one Worker rank."""

    iterator: object = field(repr=False)


@PublicAPI(stability="alpha")
class IngestionInputResolver:
    """Resolve bounded algorithm inputs only through one IngestionGateway."""

    def __init__(
        self,
        gateway: IngestionGateway | None = None,
        *,
        accepted_handle_kinds: Sequence[HandleKind] = (
            "ray_data",
            "daft_dataframe",
        ),
    ) -> None:
        accepted = frozenset(accepted_handle_kinds)
        if not accepted or not accepted.issubset(_SUPPORTED_HANDLE_KINDS):
            raise AlgorithmInputError(
                "accepted ingestion handle kinds must be ray_data and/or daft_dataframe"
            )
        self._gateway = gateway or IngestionGateway()
        self._accepted_handle_kinds = accepted

    @property
    def resolver_id(self) -> str:
        """Return the production resolver identity used by InputBinding."""
        return INGESTION_RESOLVER_ID

    def describe(
        self,
        binding: InputBinding,
        context: InputResolutionContext,
    ) -> ResolvedInputDescriptor:
        """Map one side-effect-free Gateway descriptor into the Core plan."""
        invocation = self._lookup_invocation(binding, context.values)
        ingestion_descriptor = self._describe_request(invocation.request)
        if ingestion_descriptor.handle_kind not in self._accepted_handle_kinds:
            raise AlgorithmInputError(
                "ingestion handle kind is incompatible with this algorithm runtime: "
                f"{ingestion_descriptor.handle_kind}"
            )
        deferred = tuple(
            dict.fromkeys(
                (*ingestion_descriptor.deferred_validations, "algorithm_role_columns")
            )
        )
        input_capabilities = {
            "materializable",
            *(
                capability.value
                for capability in ingestion_descriptor.available_capabilities
            ),
        }
        if ingestion_descriptor.handle_kind == "ray_data":
            input_capabilities.add("shardable")
        return ResolvedInputDescriptor(
            resolver_id=self.resolver_id,
            reference=binding.reference,
            descriptor_version=_BRIDGE_DESCRIPTOR_VERSION,
            binding_digest=canonical_digest(binding.descriptor_payload()),
            engine_id=ingestion_descriptor.engine_id,
            view_kind=ingestion_descriptor.handle_kind,
            input_capabilities=tuple(sorted(input_capabilities)),
            deferred_validations=deferred,
            resolver_payload={
                "bridge_descriptor_version": _BRIDGE_DESCRIPTOR_VERSION,
                "ingestion_descriptor": ingestion_descriptor.model_dump(mode="json"),
            },
            compatible_worker_input_adapter_refs=(
                _GENERIC_ADAPTER_REF,
                _HANDLE_ADAPTER_REFS[ingestion_descriptor.handle_kind],
            ),
        )

    def open(
        self,
        binding: InputBinding,
        descriptor: ResolvedInputDescriptor,
        context: InputExecutionContext,
    ) -> ResolvedInputLease:
        """Validate request drift before opening and retain the Gateway owner."""
        invocation = self._lookup_invocation(binding, context.values)
        expected = self.describe(
            binding,
            InputResolutionContext(values={binding.reference: invocation}),
        )
        if descriptor != expected:
            raise AlgorithmInputError(
                "ingestion request, binding, or descriptor drifted after planning"
            )

        result = self._open_request(invocation)
        try:
            self._validate_open_result(result, expected)
        except AlgorithmInputError as exc:
            try:
                result.cancel()
            except Exception as cleanup_exc:
                exc.add_note(
                    "invalid ingestion result cleanup also failed: "
                    f"{type(cleanup_exc).__name__}"
                )
            raise

        owner = _IngestionOpenResultDelegate(result)
        receipt = result.receipt.model_dump(mode="json")
        return ResolvedInputLease(
            handle=result.handle,
            provenance={
                "resolver_id": self.resolver_id,
                "reference": binding.reference,
                "binding_digest": expected.binding_digest,
                "engine_id": result.receipt.engine_id,
                "handle_kind": expected.view_kind,
                "request_digest": result.receipt.request_digest,
                "dataset_ref": result.receipt.dataset_ref,
                "receipt": receipt,
            },
            close_callback=owner.close,
            cancel_callback=owner.cancel,
        )

    def _lookup_invocation(
        self,
        binding: InputBinding,
        values: Mapping[str, object],
    ) -> IngestionInputInvocation:
        if binding.resolver_id != self.resolver_id:
            raise AlgorithmInputError(
                f"IngestionInputResolver cannot resolve {binding.resolver_id!r}"
            )
        reference = IngestionRequestRef(binding.reference)
        value = values.get(reference.request_key)
        if not isinstance(value, IngestionInputInvocation):
            raise AlgorithmInputError(
                f"missing ingestion invocation for {reference.request_key!r}"
            )
        return value

    def _describe_request(self, request: IngestionRequest) -> IngestionDescriptor:
        try:
            descriptor = self._gateway.describe(request)
        except Exception as exc:
            raise AlgorithmInputError(
                f"ingestion request description failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(descriptor, IngestionDescriptor):
            raise AlgorithmInputError("IngestionGateway returned an invalid descriptor")
        return descriptor

    def _open_request(
        self,
        invocation: IngestionInputInvocation,
    ) -> IngestionOpenResult:
        try:
            result = self._gateway.open(
                invocation.request,
                invocation.runtime_context,
            )
        except Exception as exc:
            raise AlgorithmInputError(
                f"ingestion request open failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(result, IngestionOpenResult):
            raise AlgorithmInputError(
                "IngestionGateway returned an invalid open result"
            )
        return result

    @staticmethod
    def _validate_open_result(
        result: IngestionOpenResult,
        descriptor: ResolvedInputDescriptor,
    ) -> None:
        ingestion_payload = descriptor.resolver_payload.get("ingestion_descriptor")
        try:
            planned = IngestionDescriptor.model_validate(ingestion_payload)
        except Exception as exc:
            raise AlgorithmInputError(
                "planned ingestion descriptor payload is invalid"
            ) from exc
        handle_kind: HandleKind
        if isinstance(result.handle, RayDataHandle):
            handle_kind = "ray_data"
        elif isinstance(result.handle, DaftDataFrameHandle):
            handle_kind = "daft_dataframe"
        else:
            raise AlgorithmInputError("Gateway returned an unsupported input handle")
        receipt = result.receipt
        if (
            receipt.request_digest != planned.request_digest
            or receipt.source_ref != planned.source_ref
            or receipt.dataset_ref != planned.dataset_ref
            or receipt.logical_plan_digest != planned.logical_plan_digest
            or receipt.engine_id != planned.engine_id
            or receipt.provider_id != planned.provider_id
            or receipt.connector_id != planned.connector_id
            or receipt.binding_id != planned.binding_id
            or receipt.scan_kind != planned.scan_kind
            or receipt.binding_distribution != planned.binding_distribution
            or receipt.binding_distribution_version
            != planned.binding_distribution_version
            or handle_kind != planned.handle_kind
        ):
            raise AlgorithmInputError(
                "opened ingestion result drifted from its planned descriptor"
            )


@PublicAPI(stability="alpha")
class IngestionInputRuntimeAdapter:
    """Pass a typed Gateway handle to an explicit Worker input adapter."""

    def bind(
        self,
        lease: ResolvedInputLease,
        plan: ResolvedAlgorithmPlan,
    ) -> RuntimeInputBinding:
        """Create a Worker payload without converting between ingestion engines."""
        if not isinstance(lease.handle, (RayDataHandle, DaftDataFrameHandle)):
            raise AlgorithmInputError(
                "ingestion runtime adapter requires a typed Gateway handle"
            )
        if plan.runtime.topology is RuntimeTopology.DATA_PARALLEL:
            if not isinstance(lease.handle, RayDataHandle):
                raise AlgorithmInputError(
                    "data_parallel requires RayDataHandle; no implicit Daft-to-Ray "
                    "conversion is permitted"
                )
            try:
                iterators = lease.handle.dataset.streaming_split(
                    plan.runtime.worker_count,
                    equal=False,
                )
            except Exception as exc:
                raise AlgorithmInputError(
                    f"Ray Data input sharding failed: {type(exc).__name__}"
                ) from exc
            payloads = tuple(
                WorkerInputPayload(
                    input_name=plan.input_binding.name,
                    binding=plan.input_binding,
                    value=_RayDataIteratorPayload(iterator),
                    partition_index=rank,
                    partition_count=plan.runtime.worker_count,
                )
                for rank, iterator in enumerate(iterators)
            )
            return RuntimeInputBinding(payloads)
        return RuntimeInputBinding(
            WorkerInputPayload(
                input_name=plan.input_binding.name,
                binding=plan.input_binding,
                value=lease.handle,
            )
        )


def _required_columns(binding: InputBinding) -> tuple[str, ...]:
    return binding.feature_names + (
        (binding.label_name,) if binding.label_name is not None else ()
    )


def _column_values(value: object, *, column: str) -> tuple[object, ...]:
    converted = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(converted, (list, tuple)):
        raise AlgorithmInputError(
            f"input column {column!r} did not produce a bounded sequence"
        )
    values = tuple(cast(Sequence[object], converted))
    if any(isinstance(item, (Mapping, list, tuple)) for item in values):
        raise AlgorithmInputError(
            f"input column {column!r} must be one-dimensional scalar data"
        )
    return values


def _prepared_view(
    payload: WorkerInputPayload,
    columns: Mapping[str, Sequence[object]],
) -> PreparedInput:
    view = InMemoryTabularInputView(
        _columns={name: tuple(values) for name, values in columns.items()},
        feature_names=payload.binding.feature_names,
        label_name=payload.binding.label_name,
    )
    return PreparedInput({payload.input_name: view})


def _enforce_materialized_row_limit(
    columns: Mapping[str, Sequence[object]],
) -> None:
    row_count = len(next(iter(columns.values()), ()))
    if row_count > _MAX_MATERIALIZED_ROWS:
        raise AlgorithmInputError(
            "managed algorithm input exceeds the materialization row limit: "
            f"{_MAX_MATERIALIZED_ROWS}"
        )


@DeveloperAPI
def prepare_ray_data_input(payload: WorkerInputPayload) -> PreparedInput:
    """Materialize required columns from a Ray Dataset inside the Worker."""
    batches: Any
    if isinstance(payload.value, RayDataHandle):
        batches = payload.value.dataset.select_columns(
            list(_required_columns(payload.binding))
        ).limit(_MAX_MATERIALIZED_ROWS + 1)
    elif isinstance(payload.value, _RayDataIteratorPayload):
        batches = payload.value.iterator
    else:
        raise AlgorithmInputError(
            "Ray input adapter requires RayDataHandle or a planned Ray Data shard"
        )
    required = _required_columns(payload.binding)
    materialized: dict[str, list[object]] = {name: [] for name in required}
    try:
        for batch in batches.iter_batches(batch_format="numpy"):
            if not isinstance(batch, Mapping):
                raise AlgorithmInputError("Ray Data yielded a non-columnar batch")
            for name in required:
                if name not in batch:
                    raise AlgorithmInputError(
                        f"Ray Data input is missing required column {name!r}"
                    )
                materialized[name].extend(_column_values(batch[name], column=name))
    except AlgorithmInputError:
        raise
    except Exception as exc:
        raise AlgorithmInputError(
            f"Ray Data input preparation failed: {type(exc).__name__}"
        ) from exc
    _enforce_materialized_row_limit(materialized)
    return _prepared_view(payload, materialized)


@DeveloperAPI
def prepare_daft_input(payload: WorkerInputPayload) -> PreparedInput:
    """Materialize required columns from a Daft DataFrame inside the Worker."""
    if not isinstance(payload.value, DaftDataFrameHandle):
        raise AlgorithmInputError("Daft input adapter requires DaftDataFrameHandle")
    required = _required_columns(payload.binding)
    try:
        selected = payload.value.dataframe.select(*required).limit(
            _MAX_MATERIALIZED_ROWS + 1
        )
        raw_columns = selected.to_pydict()
        columns = {
            name: _column_values(raw_columns[name], column=name) for name in required
        }
    except AlgorithmInputError:
        raise
    except Exception as exc:
        raise AlgorithmInputError(
            f"Daft input preparation failed: {type(exc).__name__}"
        ) from exc
    _enforce_materialized_row_limit(columns)
    return _prepared_view(payload, columns)


@DeveloperAPI
def prepare_ingestion_input(payload: WorkerInputPayload) -> PreparedInput:
    """Dispatch by typed handle without an implicit engine conversion."""
    if isinstance(payload.value, (RayDataHandle, _RayDataIteratorPayload)):
        return prepare_ray_data_input(payload)
    if isinstance(payload.value, DaftDataFrameHandle):
        return prepare_daft_input(payload)
    raise AlgorithmInputError("unsupported typed ingestion handle")


__all__ = [
    "IngestionInputInvocation",
    "IngestionInputResolver",
    "IngestionInputRuntimeAdapter",
    "IngestionRequestRef",
    "prepare_daft_input",
    "prepare_ingestion_input",
    "prepare_ray_data_input",
]
