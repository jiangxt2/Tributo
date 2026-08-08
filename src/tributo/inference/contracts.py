"""Candidate contracts for bundle-aware batch inference.

The models in this module are intentionally framework-neutral.  They may be
serialized into a Ray Job, but never carry a Dataset, model object, SDK client,
temporary path, or plaintext credential.  Concrete Ray and storage adapters
live outside the contract module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal, Protocol, Union, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.data import IngestionDescriptor, IngestionPlanReceipt, IngestionRequest
from tributo.exporting.manifest import ManifestSignature
from tributo.exporting.models import BundleRef
from tributo.util.annotations import PublicAPI


class _FrozenContract(BaseModel):
    """Strict immutable base for serializable inference contracts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


@PublicAPI(stability="alpha")
class BundleModelReference(_FrozenContract):
    """Reference to an already-published Tributo Bundle."""

    kind: Literal["bundle"] = "bundle"
    uri: str = Field(min_length=1)
    role: str = Field(default="inference", min_length=1)
    expected_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    storage_profile: str | None = None
    unsafe: bool = False

    @classmethod
    def from_bundle_ref(
        cls,
        bundle_ref: BundleRef,
        *,
        role: str = "inference",
        storage_profile: str | None = None,
        unsafe: bool = False,
    ) -> "BundleModelReference":
        """Bind an immutable BundleRef to one inference artifact role."""
        return cls(
            uri=bundle_ref.canonical_uri,
            role=role,
            expected_manifest_sha256=bundle_ref.manifest_sha256,
            storage_profile=storage_profile,
            unsafe=unsafe,
        )


@PublicAPI(stability="alpha")
class RegistryModelReference(_FrozenContract):
    """Reference resolved by an explicit external model-registry importer."""

    kind: Literal["registry"] = "registry"
    provider_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    alias: str | None = Field(default=None, min_length=1)
    import_bundle_uri: str = Field(min_length=1)
    storage_profile: str | None = None
    import_storage_profile: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_immutable_selector(self) -> "RegistryModelReference":
        if (self.version is None) == (self.alias is None):
            raise ValueError("exactly one of version or alias must be provided")
        return self


@PublicAPI(stability="alpha")
class ArtifactModelReference(_FrozenContract):
    """Explicit external artifact reference awaiting Bundle normalization."""

    kind: Literal["artifact"] = "artifact"
    provider_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    format_id: str = Field(min_length=1)
    flavor_id: str = Field(min_length=1)
    import_bundle_uri: str = Field(min_length=1)
    architecture_id: str | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    storage_profile: str | None = None
    import_storage_profile: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


ModelReference = Annotated[
    Union[BundleModelReference, RegistryModelReference, ArtifactModelReference],
    Field(discriminator="kind"),
]


@PublicAPI(stability="alpha")
class TensorInputBinding(_FrozenContract):
    """Map one named model input tensor to ordered table columns."""

    tensor_name: str = Field(min_length=1)
    columns: tuple[str, ...] = Field(min_length=1)
    dtype: str | None = Field(default=None, min_length=1)

    @field_validator("columns")
    @classmethod
    def _unique_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not column for column in value):
            raise ValueError("input binding columns must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("input binding columns must be unique")
        return value

    @field_validator("dtype")
    @classmethod
    def _canonical_dtype(cls, value: str | None) -> str | None:
        return _validate_binding_dtype(value)


@PublicAPI(stability="alpha")
class InputBindingSpec(_FrozenContract):
    """Complete table-to-model binding contract."""

    tensors: tuple[TensorInputBinding, ...] = Field(min_length=1)
    passthrough_columns: tuple[str, ...] = ()
    null_policy: Literal["error"] = "error"
    nan_policy: Literal["error", "allow"] = "error"

    @model_validator(mode="after")
    def _check_names(self) -> "InputBindingSpec":
        names = [binding.tensor_name for binding in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError("model input tensor names must be unique")
        if any(not column for column in self.passthrough_columns):
            raise ValueError("passthrough columns must be non-empty")
        if len(set(self.passthrough_columns)) != len(self.passthrough_columns):
            raise ValueError("passthrough columns must be unique")
        return self

    def projected_columns(self) -> tuple[str, ...]:
        """Return deterministic feature-plus-passthrough projection order."""
        return tuple(
            dict.fromkeys(
                column for binding in self.tensors for column in binding.columns
            )
        ) + tuple(
            column
            for column in self.passthrough_columns
            if column
            not in {item for binding in self.tensors for item in binding.columns}
        )


@PublicAPI(stability="alpha")
class TensorOutputBinding(_FrozenContract):
    """Map one named model output tensor to a result column."""

    tensor_name: str = Field(min_length=1)
    column: str = Field(min_length=1)
    semantic: Literal["label", "score", "probability", "embedding", "tensor"]
    dtype: str | None = Field(default=None, min_length=1)
    squeeze_singleton: bool = False

    @field_validator("dtype")
    @classmethod
    def _canonical_dtype(cls, value: str | None) -> str | None:
        return _validate_binding_dtype(value)


@PublicAPI(stability="alpha")
class OutputBindingSpec(_FrozenContract):
    """Named model-output binding and result-column policy."""

    tensors: tuple[TensorOutputBinding, ...] = Field(min_length=1)
    preserve_features: bool = False
    collision_policy: Literal["error"] = "error"

    @model_validator(mode="after")
    def _check_names(self) -> "OutputBindingSpec":
        tensor_names = [binding.tensor_name for binding in self.tensors]
        columns = [binding.column for binding in self.tensors]
        if len(set(tensor_names)) != len(tensor_names):
            raise ValueError("model output tensor names must be unique")
        if len(set(columns)) != len(columns):
            raise ValueError("result columns must be unique")
        return self


@PublicAPI(stability="alpha")
class RayExecutionPolicy(_FrozenContract):
    """Ray Data map-batches execution policy."""

    executor_id: Literal["ray-map-batches-v1"] = "ray-map-batches-v1"
    batch_size: int = Field(default=4096, ge=1)
    concurrency: int = Field(default=4, ge=1)
    num_cpus_per_actor: float = Field(default=1.0, ge=0)
    num_gpus_per_actor: float = Field(default=0.0, ge=0)


@PublicAPI(stability="alpha")
class ParquetResultSinkRequest(_FrozenContract):
    """Configuration for the first ResultSink implementation."""

    sink_id: Literal["parquet-v1"] = "parquet-v1"
    uri: str = Field(min_length=1)
    storage_profile: str | None = None
    compression: str = Field(default="zstd", min_length=1)
    min_rows_per_file: int | None = Field(default=None, ge=1)

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not parsed.scheme:
            return value
        if parsed.scheme.lower() not in {"file", "s3"}:
            raise ValueError("Parquet ResultSink URI must be local, file://, or s3://")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Parquet ResultSink URI must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "Parquet ResultSink URI must not contain query or fragment"
            )
        if parsed.scheme.lower() == "file" and parsed.netloc not in {
            "",
            "localhost",
        }:
            raise ValueError("file ResultSink URI must not name a remote host")
        if parsed.scheme.lower() == "s3" and not parsed.netloc:
            raise ValueError("s3 ResultSink URI must include a bucket")
        return value


@PublicAPI(stability="alpha")
class InferenceRequest(_FrozenContract):
    """User intent for one bounded batch-inference run."""

    schema_version: Literal[1] = 1
    model: ModelReference
    input: IngestionRequest
    input_binding: InputBindingSpec
    output_binding: OutputBindingSpec
    result_sink: ParquetResultSinkRequest
    execution: RayExecutionPolicy = Field(default_factory=RayExecutionPolicy)
    run_id: str | None = Field(default=None, min_length=1)
    parent_run_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _reject_plaintext_credentials(self) -> "InferenceRequest":
        if self.input.engine != "tributo.ray_data":
            raise ValueError(
                "batch inference currently requires ingestion engine 'ray'; "
                "automatic engine selection, fallback, and Daft-to-Ray "
                "conversion are disabled"
            )
        if self.input.trace_context:
            raise ValueError(
                "InferenceRequest.input.trace_context is reserved until a "
                "bounded cross-domain observability contract is defined"
            )
        retained = set(self.input_binding.passthrough_columns)
        if self.output_binding.preserve_features:
            retained.update(
                column
                for binding in self.input_binding.tensors
                for column in binding.columns
            )
        collisions = sorted(
            retained.intersection(
                binding.column for binding in self.output_binding.tensors
            )
        )
        if collisions:
            raise ValueError(
                f"result columns collide with retained input columns: {collisions}"
            )
        paths = _credential_paths(self.model_dump(mode="python"), "request")
        if paths:
            raise ValueError(
                "InferenceRequest must not contain plaintext credentials; use "
                "domain-specific storage profiles or provider environment chains "
                f"(fields: {sorted(paths)})"
            )
        try:
            self.model_dump_json()
        except Exception as exc:
            raise ValueError(
                f"InferenceRequest must be JSON-serializable ({type(exc).__name__})"
            ) from None
        return self


@PublicAPI(stability="alpha")
class ResolvedModelSelection(_FrozenContract):
    """Pinned Bundle identity plus per-run artifact selection."""

    bundle_ref: BundleRef
    role: str
    flavor_id: str
    storage_profile: str | None = None
    source_provenance: str
    unsafe: bool = False


@PublicAPI(stability="alpha")
class ResolvedInputSelection(_FrozenContract):
    """Pinned ingestion request plus its credential-free planning descriptor."""

    request: IngestionRequest
    descriptor: IngestionDescriptor

    @model_validator(mode="after")
    def _validate_pinned_ray_route(self) -> "ResolvedInputSelection":
        if self.request.binding_id != self.descriptor.binding_id:
            raise ValueError(
                "resolved ingestion request must pin the descriptor binding_id"
            )
        if (
            self.request.engine != "tributo.ray_data"
            or self.descriptor.engine_id != "tributo.ray_data"
            or self.descriptor.handle_kind != "ray_data"
        ):
            raise ValueError("resolved inference input must use a Ray Data handle")
        return self


@PublicAPI(stability="alpha")
class ResolvedInference(_FrozenContract):
    """Immutable internal execution object consumed by an Executor."""

    schema_version: Literal[1] = 1
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: ResolvedModelSelection
    input: ResolvedInputSelection
    input_signature: ManifestSignature
    output_signature: ManifestSignature
    input_binding: InputBindingSpec
    output_binding: OutputBindingSpec
    result_sink: ParquetResultSinkRequest
    execution: RayExecutionPolicy
    run_id: str
    attempt_id: str
    submission_id: str
    parent_run_id: str | None = None

    @model_validator(mode="after")
    def _preserve_credential_free_transport(self) -> "ResolvedInference":
        paths = _credential_paths(self.model_dump(mode="python"), "plan")
        if paths:
            raise ValueError(
                "ResolvedInference transport must not contain plaintext "
                f"credentials (fields: {sorted(paths)})"
            )
        return self


@PublicAPI(stability="alpha")
class ResultSinkReceipt(_FrozenContract):
    """Credential-free receipt returned by a ResultSink."""

    sink_id: str
    result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str
    rows_written: int | None = Field(default=None, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("uri")
    @classmethod
    def _credential_free_uri(cls, value: str) -> str:
        if _credential_paths(value, "receipt.uri"):
            raise ValueError("ResultSinkReceipt.uri must be credential-free")
        return value

    @model_validator(mode="after")
    def _credential_free_payload(self) -> "ResultSinkReceipt":
        paths = _credential_paths(self.model_dump(mode="python"), "receipt")
        if paths:
            raise ValueError(
                f"ResultSinkReceipt must be credential-free (fields: {sorted(paths)})"
            )
        return self


@PublicAPI(stability="alpha")
class FailureDiagnostic(_FrozenContract):
    """Sanitized structured failure information."""

    phase: Literal["acquisition", "execution", "materialization", "sink"]
    code: str
    error_type: str
    retryable: bool = False


@PublicAPI(stability="alpha")
class InferenceResult(_FrozenContract):
    """Immutable inference execution result."""

    run_id: str
    attempt_id: str
    submission_id: str
    parent_run_id: str | None = None
    plan_digest: str
    bundle_id: str
    manifest_sha256: str
    role: str
    flavor_id: str
    source_ref_id: str
    ingestion_receipt: IngestionPlanReceipt | None = None
    sink_receipt: ResultSinkReceipt | None = None
    input_rows: int | None = Field(default=None, ge=0)
    output_rows: int | None = Field(default=None, ge=0)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    status: Literal["succeeded", "failed", "cancelled"]
    retryable: bool = False
    failure: FailureDiagnostic | None = None

    @model_validator(mode="after")
    def _check_terminal_payload(self) -> "InferenceResult":
        if self.status == "succeeded":
            if self.sink_receipt is None:
                raise ValueError("succeeded inference requires a sink receipt")
            if self.ingestion_receipt is None:
                raise ValueError("succeeded inference requires an ingestion receipt")
            if self.failure is not None:
                raise ValueError("succeeded inference cannot carry a failure")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed inference requires a failure diagnostic")
        if self.failure is not None and self.retryable != self.failure.retryable:
            raise ValueError(
                "InferenceResult.retryable must match FailureDiagnostic.retryable"
            )
        paths = _credential_paths(self.model_dump(mode="python"), "result")
        if paths:
            raise ValueError(
                f"InferenceResult must be credential-free (fields: {sorted(paths)})"
            )
        return self


@runtime_checkable
@PublicAPI(stability="alpha")
class InferenceExecutor(Protocol):
    """Protocol for executing an immutable ResolvedInference."""

    api_version: ClassVar[int]
    executor_id: ClassVar[str]

    def execute(
        self, plan: ResolvedInference, sink: "ResultSink"
    ) -> InferenceResult: ...


@runtime_checkable
@PublicAPI(stability="alpha")
class ResultSink(Protocol):
    """Minimal output protocol owned by the Inference domain."""

    api_version: ClassVar[int]
    sink_id: ClassVar[str]

    def write(
        self,
        dataset: Any,
        request: ParquetResultSinkRequest,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt: ...


_CREDENTIAL_KEYS = frozenset(
    {
        "password",
        "secret",
        "secretaccesskey",
        "accesskeyid",
        "token",
        "sessiontoken",
        "credential",
        "clientsecret",
        "apikey",
        "apitoken",
        "authorization",
        "cookie",
        "oauthtoken",
        "privatekey",
        "refreshtoken",
        "setcookie",
        "authtoken",
    }
)

_SUPPORTED_BINDING_DTYPES = frozenset(
    {
        "bool",
        "float16",
        "float32",
        "float64",
        "int8",
        "int32",
        "int64",
        "uint8",
    }
)

_SAFE_CREDENTIAL_LIKE_KEYS = frozenset(
    {
        "inputsignature",
        "outputsignature",
    }
)


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_binding_dtype(value: str | None) -> str | None:
    if value is not None and value not in _SUPPORTED_BINDING_DTYPES:
        raise ValueError(
            f"unsupported binding dtype {value!r}; expected one of "
            f"{sorted(_SUPPORTED_BINDING_DTYPES)}"
        )
    return value


def _is_credential_key(value: str) -> bool:
    normalized = _normalized_key(value)
    if normalized in _SAFE_CREDENTIAL_LIKE_KEYS:
        return False
    return (
        normalized in _CREDENTIAL_KEYS
        or normalized.startswith(("xamz", "awsaccesskey", "awssecretaccesskey"))
        or normalized.endswith(
            (
                "password",
                "secret",
                "token",
                "credential",
                "signature",
                "apikey",
                "accesskeyid",
                "secretaccesskey",
            )
        )
    )


def _credential_paths(value: Any, path: str = "input") -> set[str]:
    """Find explicit credential fields without logging their values."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if item not in (None, "") and _is_credential_key(str(key)):
                found.add(child)
            found.update(_credential_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.update(_credential_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            found.add(path)
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_credential_key(key):
                found.add(path)
    return found


__all__ = [
    "ArtifactModelReference",
    "BundleModelReference",
    "FailureDiagnostic",
    "InferenceExecutor",
    "InferenceRequest",
    "InferenceResult",
    "InputBindingSpec",
    "OutputBindingSpec",
    "ParquetResultSinkRequest",
    "RayExecutionPolicy",
    "RegistryModelReference",
    "ResolvedInference",
    "ResolvedInputSelection",
    "ResolvedModelSelection",
    "ResultSink",
    "ResultSinkReceipt",
    "TensorInputBinding",
    "TensorOutputBinding",
]
