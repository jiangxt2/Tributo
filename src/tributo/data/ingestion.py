"""Candidate public contracts for bounded dual-engine ingestion."""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Literal, Mapping, NoReturn, TypeAlias
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.data._source_paths import resolve_file_source_path
from tributo.data.contracts.handles import (
    DaftDataFrameHandle,
    RayDataHandle,
)
from tributo.data.engine_ids import normalize_engine_id
from tributo.data.provider import ResolvedSource
from tributo.data.refs import _credential_paths, digest
from tributo.data.scan_plan import (
    LogicalScanPlan,
    ScanKind,
    SourceCapability,
    logical_scan_plan_to_dict,
)
from tributo.data.source_config import CanonicalSourceInput
from tributo.data.transform_ir import TransformPipeline, transform_ir_digest
from tributo.exceptions import DataSourceError, TributoError
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from tributo.data.engine_binding import EngineBindings

logger = logging.getLogger(__name__)


# ``describe()`` deliberately performs no connectivity, schema metadata, or
# worker probes, so these checks are deferred for every current binding.
_BASE_DEFERRED_VALIDATIONS = (
    "runtime_connectivity",
    "engine_schema",
    "worker_versions",
)


@PublicAPI(stability="alpha")
class ReadOptions(BaseModel):
    """Engine-neutral, result-independent bounded-read hints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_parallelism: int | None = Field(default=None, ge=1)
    target_split_size_bytes: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    concurrency: int | None = Field(default=None, ge=1)
    resource_hints: Mapping[str, float] = Field(default_factory=dict)

    @field_validator("resource_hints")
    @classmethod
    def _validate_resource_hints(
        cls, value: Mapping[str, float]
    ) -> Mapping[str, float]:
        if any(
            not key or not math.isfinite(amount) or amount <= 0
            for key, amount in value.items()
        ):
            raise ValueError(
                "resource_hints require non-empty names and positive finite values"
            )
        return MappingProxyType(dict(value))

    @field_serializer("resource_hints")
    def _serialize_resource_hints(self, value: Mapping[str, float]) -> dict[str, float]:
        return dict(value)

    def requested_hints(self) -> frozenset[ReadHint]:
        """Return the explicitly requested, fail-closed execution hints."""
        hints: set[ReadHint] = set()
        for field_name, hint in _READ_OPTION_HINTS.items():
            value = getattr(self, field_name)
            if value is not None and value != {}:
                hints.add(hint)
        return frozenset(hints)


@PublicAPI(stability="alpha")
class ReadHint(str, Enum):
    """Standard execution hints independently supported by each Binding."""

    TARGET_PARALLELISM = "target_parallelism"
    TARGET_SPLIT_SIZE_BYTES = "target_split_size_bytes"
    BATCH_SIZE = "batch_size"
    CONCURRENCY = "concurrency"
    RESOURCE_HINTS = "resource_hints"


_READ_OPTION_HINTS: dict[str, ReadHint] = {
    "target_parallelism": ReadHint.TARGET_PARALLELISM,
    "target_split_size_bytes": ReadHint.TARGET_SPLIT_SIZE_BYTES,
    "batch_size": ReadHint.BATCH_SIZE,
    "concurrency": ReadHint.CONCURRENCY,
    "resource_hints": ReadHint.RESOURCE_HINTS,
}


@PublicAPI(stability="alpha")
class SchemaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


@PublicAPI(stability="alpha")
class IngestionRequest(BaseModel):
    """Validated request shared by all bounded ingestion engines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: CanonicalSourceInput = Field(repr=False)
    engine: str
    binding_id: str | None = None
    storage_profile: str | None = None
    transforms: TransformPipeline = Field(default_factory=TransformPipeline)
    schema_contract: SchemaContract | None = None
    read_options: ReadOptions = Field(default_factory=ReadOptions)
    trace_context: Mapping[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("source")
    @classmethod
    def _copy_source(cls, value: CanonicalSourceInput) -> CanonicalSourceInput:
        return value.model_copy(deep=True)

    @field_validator("engine")
    @classmethod
    def _normalize_engine(cls, value: str) -> str:
        return normalize_engine_id(value)

    @field_validator("binding_id")
    @classmethod
    def _validate_binding_id(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[a-z][a-z0-9_.-]+", value) is None:
            raise ValueError("binding_id must be a namespaced identifier")
        return value

    @field_validator("storage_profile")
    @classmethod
    def _validate_storage_profile(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("storage_profile must be non-empty")
        return value

    @field_validator("trace_context")
    @classmethod
    def _validate_trace_context(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        leaked = _credential_paths(value)
        if leaked:
            raise ValueError(
                f"trace_context must not contain credential field(s): {sorted(leaked)}"
            )
        return MappingProxyType(dict(value))

    @field_serializer("trace_context")
    def _serialize_trace_context(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    def source_json_for_remote_transport(self) -> str:
        """Serialize a source only when it is safe to cross a job boundary."""
        payload = self.source.model_dump(mode="python", exclude_none=True)
        leaked = _credential_paths(payload, text_exempt_keys=frozenset({"sql"}))
        if leaked:
            raise ValueError(
                "ingestion source contains inline credentials at "
                f"{sorted(leaked)}; configure credentials through runtime "
                "environment variables, IAM, or a storage profile instead"
            )
        return self.source.model_dump_json(exclude_none=True)


@PublicAPI(stability="alpha")
class TransformDecision(BaseModel):
    """Plan-time decision for one ordered Transform step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal: int = Field(ge=0)
    transform_type: str = Field(min_length=1)
    pushdown_level: Literal["exact", "inexact", "none"]
    residual_required: bool
    compiled_result: Literal["pushed", "residual", "pushed_and_residual"]
    diagnostic: str | None = None

    @model_validator(mode="after")
    def _validate_classification(self) -> "TransformDecision":
        expected = {
            "exact": (False, "pushed"),
            "inexact": (True, "pushed_and_residual"),
            "none": (True, "residual"),
        }[self.pushdown_level]
        if (self.residual_required, self.compiled_result) != expected:
            raise ValueError(
                "transform decision is inconsistent with its pushdown level"
            )
        if _credential_paths({"diagnostic": self.diagnostic}):
            raise ValueError("TransformDecision must be credential-free")
        return self


@PublicAPI(stability="alpha")
class DistributionVersionEvidence(BaseModel):
    """Actual driver and worker versions used by one opened ingestion plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_name: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    worker_versions: tuple[str, ...] = ()
    worker_validation_complete: bool = False

    @model_validator(mode="after")
    def _reject_credentials(self) -> "DistributionVersionEvidence":
        if _credential_paths(self.model_dump(mode="python")):
            raise ValueError("DistributionVersionEvidence must be credential-free")
        return self


DistributionProbe: TypeAlias = Callable[
    [tuple[str, ...]], Mapping[str, tuple[str, ...]]
]


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class IngestionRuntimeContext:
    """Ephemeral runtime services and storage selection for ``open()``."""

    distribution_probe: DistributionProbe | None = field(
        default=None, repr=False, compare=False
    )
    require_worker_validation: bool = False

    def __post_init__(self) -> None:
        if self.distribution_probe is not None and not callable(
            self.distribution_probe
        ):
            raise ValueError("distribution_probe must be callable")
        if self.require_worker_validation and self.distribution_probe is None:
            raise ValueError("require_worker_validation requires a distribution_probe")


@PublicAPI(stability="alpha")
class HandleOwnership(str, Enum):
    """Who owns native-handle cleanup callbacks.

    Only ``OWNED`` results may carry callbacks. Borrowed and session-scoped
    handles are released by their external owner.
    """

    OWNED = "owned"
    BORROWED = "borrowed"
    SESSION_SCOPED = "session-scoped"


@PublicAPI(stability="alpha")
class PhysicalSplitSummary(BaseModel):
    """Credential-free summary supplied by the actual native Connector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_count: int | None = Field(default=None, ge=0)
    partition_count: int | None = Field(default=None, ge=0)
    detail: str | None = None

    @model_validator(mode="after")
    def _reject_credentials(self) -> "PhysicalSplitSummary":
        leaked = _credential_paths({"detail": self.detail})
        if leaked:
            raise ValueError("Physical split summary must be credential-free")
        return self


@PublicAPI(stability="alpha")
class IngestionDescriptor(BaseModel):
    """Credential-free static description produced without metadata I/O."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor_version: Literal[2] = 2
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    connector_id: str = Field(min_length=1)
    binding_id: str = Field(min_length=1)
    scan_kind: ScanKind
    handle_kind: Literal["ray_data", "daft_dataframe"]
    schema_contract_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    required_capabilities: tuple[SourceCapability, ...] = ()
    available_capabilities: tuple[SourceCapability, ...] = ()
    deferred_validations: tuple[str, ...] = ()
    binding_distribution: str = Field(min_length=1)
    binding_distribution_version: str = Field(min_length=1)
    capability_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _reject_credentials(self) -> "IngestionDescriptor":
        if _credential_paths(self.model_dump(mode="python")):
            raise ValueError("IngestionDescriptor must be credential-free")
        return self


@PublicAPI(stability="alpha")
class IngestionPlanReceipt(BaseModel):
    """Credential-free record of how an ingestion request was compiled."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal[2] = 2
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_id: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    connector_id: str = Field(min_length=1)
    binding_id: str = Field(min_length=1)
    scan_kind: ScanKind
    logical_plan_version: int = Field(ge=1)
    logical_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_ir_version: int = Field(ge=1)
    transform_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_decisions: tuple[TransformDecision, ...] = ()
    input_schema_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    schema_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata_fetched: bool = False
    binding_distribution: str = Field(min_length=1)
    binding_distribution_version: str = Field(min_length=1)
    reader_api: str = Field(min_length=1)
    transport_id: str = Field(min_length=1)
    physical_splits: PhysicalSplitSummary = Field(default_factory=PhysicalSplitSummary)
    component_versions: tuple[DistributionVersionEvidence, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_credentials(self) -> "IngestionPlanReceipt":
        leaked = _credential_paths(self.model_dump(mode="python"))
        if leaked:
            raise ValueError("IngestionPlanReceipt must be credential-free")
        return self


IngestionHandle: TypeAlias = RayDataHandle | DaftDataFrameHandle


@PublicAPI(stability="alpha")
class IngestionOpenResult:
    """Typed native handle, receipt, and idempotent lifecycle boundary."""

    def __init__(
        self,
        *,
        handle: IngestionHandle,
        receipt: IngestionPlanReceipt,
        ownership: HandleOwnership = HandleOwnership.OWNED,
        close_callback: Callable[[], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(handle, (RayDataHandle, DaftDataFrameHandle)):
            raise TypeError("handle must be RayDataHandle or DaftDataFrameHandle")
        if not isinstance(receipt, IngestionPlanReceipt):
            raise TypeError("receipt must be IngestionPlanReceipt")
        if not isinstance(ownership, HandleOwnership):
            raise TypeError("ownership must be HandleOwnership")
        if close_callback is not None and not callable(close_callback):
            raise TypeError("close_callback must be callable")
        if cancel_callback is not None and not callable(cancel_callback):
            raise TypeError("cancel_callback must be callable")
        if ownership is not HandleOwnership.OWNED and (
            close_callback is not None or cancel_callback is not None
        ):
            raise ValueError("Non-owned handles cannot define lifecycle callbacks")
        self.handle = handle
        self.receipt = receipt
        self.ownership = ownership
        self._close_callback = close_callback
        self._cancel_callback = cancel_callback
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        self._finish(cancel=False)

    def cancel(self) -> None:
        self._finish(cancel=True)

    def _finish(self, *, cancel: bool) -> None:
        callback: Callable[[], None] | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.ownership is HandleOwnership.OWNED:
                callback = (
                    (self._cancel_callback or self._close_callback)
                    if cancel
                    else self._close_callback
                )
        if callback is not None:
            callback()

    def __enter__(self) -> "IngestionOpenResult":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class _PreparedIngestion:
    resolved: ResolvedSource
    plan: LogicalScanPlan
    source_ref: str
    transform_digest: str
    dataset_ref: str
    logical_plan_digest: str
    request_digest: str


@PublicAPI(stability="alpha")
class IngestionGateway:
    """Only downstream facade for describing and opening bounded ingestion."""

    def __init__(
        self,
        bindings: EngineBindings | None = None,
        *,
        project_root_path: Path | None = None,
    ) -> None:
        if bindings is None:
            # Delayed to avoid the ingestion <-> engine-binding import cycle.
            from tributo.data.bindings import default_engine_bindings

            bindings = default_engine_bindings()
        self._bindings = bindings
        self._project_root_path = project_root_path

    def _prepare(self, request: IngestionRequest) -> _PreparedIngestion:
        from tributo.data.provider_registry import resolve_provider

        source = resolve_file_source_path(request.source, self._project_root_path)
        provider = resolve_provider(source)
        resolved = provider.normalize(source)
        resolved = self._attach_storage_profile(request, resolved)
        plan = provider.plan(resolved)
        if plan.provider_id != resolved.provider_id:
            raise DataSourceError(
                "Provider returned a logical plan with a different provider_id"
            )
        source_ref = resolved.ref_id()
        transform_digest = transform_ir_digest(request.transforms)
        plan_digest = digest(logical_scan_plan_to_dict(plan))
        dataset_ref = digest(
            {
                "version": 1,
                "source_ref": source_ref,
                "transform_ir_version": request.transforms.version,
                "transform_digest": transform_digest,
            }
        )
        request_digest = digest(
            {
                "version": 1,
                "dataset_ref": dataset_ref,
                "logical_plan_digest": plan_digest,
                "engine_id": request.engine,
                "binding_id": request.binding_id,
                "read_options": request.read_options.model_dump(mode="json"),
                "schema_contract": (
                    request.schema_contract.model_dump(mode="json")
                    if request.schema_contract is not None
                    else None
                ),
            }
        )
        return _PreparedIngestion(
            resolved=resolved,
            plan=plan,
            source_ref=source_ref,
            transform_digest=transform_digest,
            dataset_ref=dataset_ref,
            logical_plan_digest=plan_digest,
            request_digest=request_digest,
        )

    @staticmethod
    def _attach_storage_profile(
        request: IngestionRequest, resolved: ResolvedSource
    ) -> ResolvedSource:
        """Add resolved storage identity/runtime data without persisting secrets."""
        if request.storage_profile is None:
            return resolved
        is_s3_file = urlsplit(resolved.canonical_uri).scheme.lower() == "s3"
        is_s3_backed_table = resolved.provider_id == "tributo.iceberg"
        if not is_s3_file and not is_s3_backed_table:
            raise DataSourceError(
                "storage_profile requires an S3 file or S3-backed table source"
            )
        profile = StorageProfileResolver().resolve(request.storage_profile)
        identity_options = dict(resolved.identity_options)
        s3_identity = dict(identity_options.get("s3", {}))
        if profile.endpoint and "endpoint" not in s3_identity:
            s3_identity["endpoint"] = _credential_free_endpoint(profile.endpoint)
        if profile.region and "region" not in s3_identity:
            s3_identity["region"] = profile.region
        if s3_identity:
            identity_options["s3"] = s3_identity
        runtime_options = dict(resolved.runtime_options)
        runtime_options["s3_profile"] = profile
        return ResolvedSource(
            provider_id=resolved.provider_id,
            canonical_uri=resolved.canonical_uri,
            identity_options=identity_options,
            runtime_options=runtime_options,
        )

    def describe(self, request: IngestionRequest) -> IngestionDescriptor:
        """Validate static routing without constructing an engine-native plan."""
        prepared = self._prepare(request)
        descriptor, required = self._bindings.describe(
            engine_id=request.engine,
            plan=prepared.plan,
            binding_id=request.binding_id,
            read_options=request.read_options,
        )
        deferred = list(_BASE_DEFERRED_VALIDATIONS)
        if request.schema_contract is not None:
            deferred.append("schema_contract")
        return IngestionDescriptor(
            request_digest=prepared.request_digest,
            source_ref=prepared.source_ref,
            dataset_ref=prepared.dataset_ref,
            logical_plan_digest=prepared.logical_plan_digest,
            engine_id=request.engine,
            provider_id=prepared.plan.provider_id,
            connector_id=prepared.plan.connector_id,
            binding_id=descriptor.key.binding_id,
            scan_kind=prepared.plan.scan_kind,
            handle_kind=(
                "ray_data" if request.engine == "tributo.ray_data" else "daft_dataframe"
            ),
            schema_contract_fingerprint=(
                request.schema_contract.fingerprint
                if request.schema_contract is not None
                else None
            ),
            required_capabilities=tuple(sorted(required, key=lambda item: item.value)),
            available_capabilities=tuple(
                sorted(descriptor.capabilities, key=lambda item: item.value)
            ),
            deferred_validations=tuple(deferred),
            binding_distribution=descriptor.distribution_name,
            binding_distribution_version=descriptor.distribution_version,
            capability_version=descriptor.capability_version,
        )

    def open(
        self,
        request: IngestionRequest,
        runtime_context: IngestionRuntimeContext | None = None,
    ) -> IngestionOpenResult:
        """Compile a lazy native handle without implementing a data Reader."""
        context = runtime_context or IngestionRuntimeContext()
        prepared = self._prepare(request)
        compilation, descriptor, versions = self._bindings.compile(
            engine_id=request.engine,
            plan=prepared.plan,
            binding_id=request.binding_id,
            runtime_options=prepared.resolved.runtime_options,
            transforms=request.transforms,
            read_options=request.read_options,
            source_ref=prepared.source_ref,
            runtime_context=context,
        )
        if request.schema_contract is not None:
            if compilation.schema_fingerprint is None:
                error = DataSourceError(
                    "Selected binding did not provide the schema required by "
                    "schema_contract"
                )
                _raise_after_failed_compilation(
                    ownership=compilation.ownership,
                    close_callback=compilation.close_callback,
                    cancel_callback=compilation.cancel_callback,
                    error=error,
                )
            if compilation.schema_fingerprint != request.schema_contract.fingerprint:
                error = DataSourceError(
                    "Ingestion schema does not match the requested schema contract"
                )
                _raise_after_failed_compilation(
                    ownership=compilation.ownership,
                    close_callback=compilation.close_callback,
                    cancel_callback=compilation.cancel_callback,
                    error=error,
                )
        receipt = IngestionPlanReceipt(
            request_digest=prepared.request_digest,
            engine_id=request.engine,
            engine_version=compilation.engine_version,
            provider_id=prepared.plan.provider_id,
            connector_id=prepared.plan.connector_id,
            binding_id=descriptor.key.binding_id,
            scan_kind=prepared.plan.scan_kind,
            logical_plan_version=prepared.plan.version,
            logical_plan_digest=prepared.logical_plan_digest,
            source_ref=prepared.source_ref,
            dataset_ref=prepared.dataset_ref,
            transform_ir_version=request.transforms.version,
            transform_digest=prepared.transform_digest,
            transform_decisions=compilation.transform_decisions,
            input_schema_fingerprint=compilation.input_schema_fingerprint,
            schema_fingerprint=compilation.schema_fingerprint,
            metadata_fetched=compilation.metadata_fetched,
            binding_distribution=descriptor.distribution_name,
            binding_distribution_version=descriptor.distribution_version,
            reader_api=compilation.reader_api,
            transport_id=compilation.transport_id,
            physical_splits=compilation.physical_splits,
            component_versions=versions,
            diagnostics=compilation.diagnostics,
        )
        return IngestionOpenResult(
            handle=compilation.handle,
            receipt=receipt,
            ownership=compilation.ownership,
            close_callback=compilation.close_callback,
            cancel_callback=compilation.cancel_callback,
        )


@PublicAPI(stability="alpha")
def ray_worker_distribution_probe(
    distributions: tuple[str, ...],
) -> Mapping[str, tuple[str, ...]]:
    """Collect actual distribution versions once on every alive Ray node."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        raise DataSourceError("Ray must be initialized before worker version probing")

    @ray.remote(num_cpus=0)
    def _probe(names: tuple[str, ...]) -> dict[str, str]:
        import importlib.metadata as metadata

        return {name: metadata.version(name) for name in names}

    nodes = [node for node in ray.nodes() if node.get("Alive")]
    refs = [
        _probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], soft=False
            )
        ).remote(distributions)
        for node in nodes
    ]
    results = ray.get(refs)
    return {name: tuple(result[name] for result in results) for name in distributions}


def _credential_free_endpoint(endpoint: str) -> str:
    has_scheme = "://" in endpoint
    parts = urlsplit(endpoint if has_scheme else f"//{endpoint}")
    if parts.hostname is None:
        raise DataSourceError("Storage profile endpoint must be a valid URI")
    netloc = parts.hostname
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    try:
        port = parts.port
    except ValueError:
        raise DataSourceError(
            "Storage profile endpoint must contain a valid port"
        ) from None
    if port is not None:
        netloc = f"{netloc}:{port}"
    sanitized = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return sanitized if has_scheme else sanitized.removeprefix("//")


def _raise_after_failed_compilation(
    *,
    ownership: HandleOwnership,
    close_callback: Callable[[], None] | None,
    cancel_callback: Callable[[], None] | None,
    error: TributoError,
) -> NoReturn:
    """Release one invalid owned compilation and raise its contract error."""
    if ownership is not HandleOwnership.OWNED:
        raise error
    callback = cancel_callback or close_callback
    try:
        if callback is not None:
            callback()
    except Exception as exc:
        # Cleanup failures must not replace or become the hidden context of
        # the credential-free contract error.
        exception_type = type(exc).__name__
        if len(exception_type) > 128 or not exception_type.isidentifier():
            exception_type = "Exception"
        callback_kind = "cancel" if callback is cancel_callback else "close"
        logger.warning(
            "Failed to %s invalid owned ingestion handle; exception_type=%s",
            callback_kind,
            exception_type,
        )
    raise error


@PublicAPI(stability="alpha")
def describe_ingestion(
    request: IngestionRequest,
    *,
    project_root_path: Path | None = None,
) -> IngestionDescriptor:
    return IngestionGateway(project_root_path=project_root_path).describe(request)


@PublicAPI(stability="alpha")
def open_ingestion(
    request: IngestionRequest,
    runtime_context: IngestionRuntimeContext | None = None,
    *,
    project_root_path: Path | None = None,
) -> IngestionOpenResult:
    return IngestionGateway(project_root_path=project_root_path).open(
        request, runtime_context
    )


__all__ = [
    "DaftDataFrameHandle",
    "DistributionVersionEvidence",
    "HandleOwnership",
    "IngestionDescriptor",
    "IngestionGateway",
    "IngestionOpenResult",
    "IngestionPlanReceipt",
    "IngestionRequest",
    "IngestionRuntimeContext",
    "PhysicalSplitSummary",
    "RayDataHandle",
    "ReadHint",
    "ReadOptions",
    "SchemaContract",
    "TransformDecision",
    "describe_ingestion",
    "open_ingestion",
    "ray_worker_distribution_probe",
]
