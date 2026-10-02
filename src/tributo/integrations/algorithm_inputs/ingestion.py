"""Bridge the public ingestion Gateway to portable algorithm input ports."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
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
    TabularBatchInputView,
    WorkerInputPayload,
    WorkerInputPayloadSet,
)
from tributo.data import (
    DaftDataFrameHandle,
    IngestionDescriptor,
    IngestionGateway,
    IngestionOpenResult,
    IngestionRequest,
    IngestionRuntimeContext,
    RayDataHandle,
    RayHandleAdaptation,
    adapt_daft_result_to_ray,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

INGESTION_RESOLVER_ID = "tributo.ingestion"
_BRIDGE_DESCRIPTOR_VERSION = 1
_REQUEST_KEY = re.compile(r"^[a-z][a-z0-9_.-]+$")
_SUPPORTED_HANDLE_KINDS = frozenset({"ray_data", "daft_dataframe"})
_MAX_MATERIALIZED_ROWS = 1_000_000
_ADAPTER_MODULE = "tributo.integrations.algorithm_inputs.ingestion"
_GENERIC_ADAPTER_REF = f"{_ADAPTER_MODULE}:prepare_ingestion_input"
_RAY_BATCH_ADAPTER_REF = f"{_ADAPTER_MODULE}:prepare_ray_batch_input"
_RAY_TRAIN_ADAPTER_REF = f"{_ADAPTER_MODULE}:prepare_ray_train_input"
_HANDLE_ADAPTER_REFS = {
    "ray_data": f"{_ADAPTER_MODULE}:prepare_ray_data_input",
    "daft_dataframe": f"{_ADAPTER_MODULE}:prepare_daft_input",
}
HandleKind = Literal["ray_data", "daft_dataframe"]
HandleAdapterId = Literal["tributo.daft_to_ray"]
_DAFT_TO_RAY_ADAPTER_ID: HandleAdapterId = "tributo.daft_to_ray"


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
    handle_adapter_id: HandleAdapterId | None = field(
        default=None,
        kw_only=True,
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
        if self.handle_adapter_id not in (None, _DAFT_TO_RAY_ADAPTER_ID):
            raise AlgorithmInputError(
                "unsupported ingestion handle adapter; expected "
                f"{_DAFT_TO_RAY_ADAPTER_ID!r}"
            )


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
        source_descriptor = self._describe_request(invocation.request)
        ingestion_descriptor = self._describe_adapted_handle(
            source_descriptor,
            invocation.handle_adapter_id,
        )
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
                "source_ingestion_descriptor": source_descriptor.model_dump(
                    mode="json"
                ),
                "handle_adapter_id": invocation.handle_adapter_id,
            },
            compatible_worker_input_adapter_refs=tuple(
                sorted(
                    {
                        _GENERIC_ADAPTER_REF,
                        _HANDLE_ADAPTER_REFS[ingestion_descriptor.handle_kind],
                        *(
                            (_RAY_BATCH_ADAPTER_REF,)
                            if ingestion_descriptor.handle_kind == "ray_data"
                            else ()
                        ),
                        *(
                            (_RAY_TRAIN_ADAPTER_REF,)
                            if ingestion_descriptor.handle_kind == "ray_data"
                            else ()
                        ),
                    }
                )
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
            adaptation = self._adapt_open_result(
                result,
                invocation.handle_adapter_id,
            )
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
        provenance: dict[str, object] = {
            "resolver_id": self.resolver_id,
            "reference": binding.reference,
            "binding_digest": expected.binding_digest,
            "engine_id": result.receipt.engine_id,
            "handle_kind": expected.view_kind,
            "request_digest": result.receipt.request_digest,
            "dataset_ref": result.receipt.dataset_ref,
            "receipt": receipt,
        }
        lease_handle: RayDataHandle | DaftDataFrameHandle = result.handle
        if adaptation is not None:
            lease_handle = adaptation.handle
            provenance.update(
                {
                    "handle_adapter_id": _DAFT_TO_RAY_ADAPTER_ID,
                    "conversion_receipt": adaptation.receipt.model_dump(mode="json"),
                }
            )
        return ResolvedInputLease(
            handle=lease_handle,
            binding=binding,
            provenance=provenance,
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

    @staticmethod
    def _describe_adapted_handle(
        descriptor: IngestionDescriptor,
        adapter_id: HandleAdapterId | None,
    ) -> IngestionDescriptor:
        """Project the source descriptor into the explicitly requested handle."""
        if adapter_id is None:
            return descriptor
        if adapter_id != _DAFT_TO_RAY_ADAPTER_ID:
            raise AlgorithmInputError(
                f"unsupported ingestion handle adapter {adapter_id!r}"
            )
        if descriptor.engine_id != "tributo.daft" or descriptor.handle_kind != (
            "daft_dataframe"
        ):
            raise AlgorithmInputError(
                "tributo.daft_to_ray requires a DaftDataFrameHandle source"
            )
        return descriptor.model_copy(
            update={
                "engine_id": "tributo.ray_data",
                "handle_kind": "ray_data",
            }
        )

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
    def _adapt_open_result(
        result: IngestionOpenResult,
        adapter_id: HandleAdapterId | None,
    ) -> RayHandleAdaptation | None:
        if adapter_id is None:
            return None
        if adapter_id != _DAFT_TO_RAY_ADAPTER_ID:
            raise AlgorithmInputError(
                f"unsupported ingestion handle adapter {adapter_id!r}"
            )
        try:
            return adapt_daft_result_to_ray(result)
        except Exception as exc:
            raise AlgorithmInputError(
                f"Daft-to-Ray handle adaptation failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _validate_open_result(
        result: IngestionOpenResult,
        descriptor: ResolvedInputDescriptor,
    ) -> None:
        ingestion_payload = descriptor.resolver_payload.get(
            "source_ingestion_descriptor",
            descriptor.resolver_payload.get("ingestion_descriptor"),
        )
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
        binding = lease.binding or plan.primary_input_binding
        if plan.runtime.topology is RuntimeTopology.RAY_TRAIN_TORCH:
            if not isinstance(lease.handle, RayDataHandle):
                raise AlgorithmInputError(
                    "Ray Train Torch input requires RayDataHandle; no implicit "
                    "Daft-to-Ray conversion is permitted"
                )
            return RuntimeInputBinding(
                tuple(
                    WorkerInputPayload(
                        input_name=binding.name,
                        binding=binding,
                        value=lease.handle,
                        partition_index=rank,
                        partition_count=plan.runtime.worker_count,
                    )
                    for rank in range(plan.runtime.worker_count)
                )
            )
        if plan.runtime.topology in {
            RuntimeTopology.DATA_PARALLEL,
            RuntimeTopology.RAY_MAP_REDUCE,
            RuntimeTopology.RAY_ITERATIVE_OPTIMIZATION,
        }:
            if not isinstance(lease.handle, RayDataHandle):
                raise AlgorithmInputError(
                    "distributed input requires RayDataHandle; no implicit Daft-to-Ray "
                    "conversion is permitted"
                )
            try:
                expected_total_rows = (
                    lease.handle.dataset.count()
                    if plan.runtime.topology
                    in {
                        RuntimeTopology.RAY_MAP_REDUCE,
                        RuntimeTopology.RAY_ITERATIVE_OPTIMIZATION,
                    }
                    else None
                )
                if expected_total_rows is not None and (
                    not isinstance(expected_total_rows, int)
                    or isinstance(expected_total_rows, bool)
                    or expected_total_rows < 0
                ):
                    raise AlgorithmInputError(
                        "Ray Data count returned an invalid expected row count"
                    )
                partitions: tuple[RayDataHandle | _RayDataIteratorPayload, ...]
                if plan.runtime.topology is RuntimeTopology.RAY_ITERATIVE_OPTIMIZATION:
                    partitions = tuple(
                        RayDataHandle(dataset=dataset)
                        for dataset in lease.handle.dataset.split(
                            plan.runtime.worker_count,
                            equal=False,
                        )
                    )
                else:
                    partitions = tuple(
                        _RayDataIteratorPayload(iterator)
                        for iterator in lease.handle.dataset.streaming_split(
                            plan.runtime.worker_count,
                            equal=False,
                        )
                    )
            except Exception as exc:
                raise AlgorithmInputError(
                    f"Ray Data input sharding failed: {type(exc).__name__}"
                ) from exc
            payloads = tuple(
                WorkerInputPayload(
                    input_name=binding.name,
                    binding=binding,
                    value=partition,
                    partition_index=rank,
                    partition_count=plan.runtime.worker_count,
                    expected_total_rows=expected_total_rows,
                )
                for rank, partition in enumerate(partitions)
            )
            return RuntimeInputBinding(payloads)
        return RuntimeInputBinding(
            WorkerInputPayload(
                input_name=binding.name,
                binding=binding,
                value=lease.handle,
            )
        )


def _required_columns(binding: InputBinding) -> tuple[str, ...]:
    return (
        binding.feature_names
        + ((binding.label_name,) if binding.label_name is not None else ())
        + (
            (binding.sample_weight_name,)
            if binding.sample_weight_name is not None
            else ()
        )
    )


def _prepare_role_payloads(
    payload: WorkerInputPayloadSet,
    adapter: Callable[[WorkerInputPayload], PreparedInput],
) -> PreparedInput:
    prepared: list[PreparedInput] = []
    try:
        for role_payload in payload.payloads:
            prepared.append(adapter(role_payload))
    except Exception:
        for item in reversed(prepared):
            item.close()
        raise
    views = {name: view for item in prepared for name, view in item.views.items()}

    def close_all() -> None:
        for item in reversed(prepared):
            item.close()

    return PreparedInput(views, close_callback=close_all)


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


class _RayTabularBatchInputView:
    """Keep one Ray Data shard streaming for bounded-state algorithms."""

    def __init__(self, payload: WorkerInputPayload, batches: Any) -> None:
        self.feature_names = payload.binding.feature_names
        self.label_name = payload.binding.label_name
        self._required = _required_columns(payload.binding)
        self._batches = batches

    def iter_batches(self) -> Any:
        """Yield validated NumPy-format batches from this Worker shard."""
        try:
            for batch in self._batches.iter_batches(
                batch_format="numpy",
                prefetch_batches=0,
            ):
                if not isinstance(batch, Mapping):
                    raise AlgorithmInputError("Ray Data yielded a non-columnar batch")
                missing = [name for name in self._required if name not in batch]
                if missing:
                    raise AlgorithmInputError(
                        f"Ray Data input is missing required column(s): {missing}"
                    )
                yield {name: batch[name] for name in self._required}
        except AlgorithmInputError:
            raise
        except Exception as exc:
            raise AlgorithmInputError(
                f"Ray Data batch iteration failed: {type(exc).__name__}"
            ) from exc


@DeveloperAPI
def prepare_ray_batch_input(
    payload: WorkerInputPayload | WorkerInputPayloadSet,
) -> PreparedInput:
    """Expose one Ray Dataset shard as a streaming batch view."""
    if isinstance(payload, WorkerInputPayloadSet):
        return _prepare_role_payloads(payload, prepare_ray_batch_input)
    batches: Any
    if isinstance(payload.value, RayDataHandle):
        batches = payload.value.dataset.select_columns(
            list(_required_columns(payload.binding))
        )
    elif isinstance(payload.value, _RayDataIteratorPayload):
        batches = payload.value.iterator
    else:
        raise AlgorithmInputError(
            "Ray batch adapter requires RayDataHandle or a planned Ray Data shard"
        )
    view: TabularBatchInputView = _RayTabularBatchInputView(payload, batches)
    return PreparedInput({payload.input_name: view})


@DeveloperAPI
def prepare_ray_train_input(
    payload: WorkerInputPayload | WorkerInputPayloadSet,
) -> PreparedInput:
    """Expose one canonical Ray Dataset for framework-owned worker sharding."""
    if isinstance(payload, WorkerInputPayloadSet):
        return _prepare_role_payloads(payload, prepare_ray_train_input)
    if not isinstance(payload.value, RayDataHandle):
        raise AlgorithmInputError(
            "Ray Train input adapter requires an unsplit RayDataHandle"
        )
    required = list(_required_columns(payload.binding))
    dataset = payload.value.dataset.select_columns(required)
    return PreparedInput({payload.input_name: dataset})


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
def prepare_ray_data_input(
    payload: WorkerInputPayload | WorkerInputPayloadSet,
) -> PreparedInput:
    """Materialize required columns from a Ray Dataset inside the Worker."""
    if isinstance(payload, WorkerInputPayloadSet):
        return _prepare_role_payloads(payload, prepare_ray_data_input)
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
def prepare_daft_input(
    payload: WorkerInputPayload | WorkerInputPayloadSet,
) -> PreparedInput:
    """Materialize required columns from a Daft DataFrame inside the Worker."""
    if isinstance(payload, WorkerInputPayloadSet):
        return _prepare_role_payloads(payload, prepare_daft_input)
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
def prepare_ingestion_input(
    payload: WorkerInputPayload | WorkerInputPayloadSet,
) -> PreparedInput:
    """Dispatch by typed handle without an implicit engine conversion."""
    if isinstance(payload, WorkerInputPayloadSet):
        return _prepare_role_payloads(payload, prepare_ingestion_input)
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
    "prepare_ray_batch_input",
    "prepare_ray_data_input",
    "prepare_ray_train_input",
]
