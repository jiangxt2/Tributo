"""Credential-safe contracts for the bounded data-writing control plane."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from tributo.data.contracts.handles import DaftDataFrameHandle, RayDataHandle
from tributo.data.contracts.modes import WriteMode
from tributo.data.engine_ids import normalize_engine_id
from tributo.data.refs import _credential_paths, digest
from tributo.data.runtime_credentials import (
    credential_free_runtime_value,
    is_native_runtime_object,
)
from tributo.data.writing.capabilities import WriteCapability
from tributo.exceptions import DataSourceError
from tributo.util.annotations import DeveloperAPI, PublicAPI

_REFERENCE_KEY_NAMES = frozenset({"secret_ref", "credential_ref"})
_REFERENCE_URI_RE = re.compile(r"^(?:env|secret|iam|profile)://[A-Za-z0-9_.:/-]+$")
_WRITE_URI_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")
_WRITE_QUERY_RE = re.compile(
    r"([?&])([^=&\s]*(?:password|secret|token|credential|key)[^=&\s]*)=([^&\s]*)",
    re.IGNORECASE,
)
_WRITE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<key>[a-z0-9_-]*(?:password|secret|token|credential|api[_-]?key|access[_-]?key)[a-z0-9_-]*)"
    r"(?P<separator>\s*[:=]\s*)(?P<value>[^\s;&]+)"
)
_WRITE_LOCAL_PATH_RE = re.compile(r"(?:/Users/|/home/|/tmp/|/private/tmp/)[^\s'\"]*")


def _safe_exception_summary(exc: BaseException) -> str:
    """Retain bounded native context without exposing secrets or host paths."""
    message = " ".join(str(exc).splitlines()).strip()
    if not message:
        return type(exc).__name__
    message = _WRITE_URI_USERINFO_RE.sub(r"\g<scheme><redacted>@", message)
    message = _WRITE_QUERY_RE.sub(r"\1\2=<redacted>", message)
    message = _WRITE_ASSIGNMENT_RE.sub(r"\g<key>\g<separator><redacted>", message)
    message = _WRITE_LOCAL_PATH_RE.sub("<local-path>", message)
    return message[:1000]


def _validate_runtime_options(value: Mapping[str, Any]) -> None:
    """Allow references while rejecting resolved credential material."""

    def visit(nested_value: Any) -> None:
        if is_native_runtime_object(nested_value):
            return
        if isinstance(nested_value, Mapping):
            for key, child in nested_value.items():
                key_text = str(key).lower()
                if key_text in _REFERENCE_KEY_NAMES:
                    if (
                        not isinstance(child, str)
                        or _REFERENCE_URI_RE.fullmatch(child) is None
                    ):
                        raise ValueError(
                            "runtime credential references must use an approved URI"
                        )
                    continue
                if _credential_paths({str(key): None}):
                    raise ValueError(
                        "runtime options must not contain inline credentials"
                    )
                visit(child)
        elif isinstance(nested_value, (list, tuple)):
            for item in nested_value:
                visit(item)
        elif isinstance(nested_value, str) and _credential_paths(
            {"value": nested_value}
        ):
            raise ValueError("runtime options must not contain inline credentials")

    visit(value)


@PublicAPI(stability="alpha")
class WriteRequest(BaseModel):
    """Immutable, credential-free request for one native engine write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: str
    target_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    target: str = Field(min_length=1, repr=False)
    binding_id: str | None = None
    mode: WriteMode
    options: Mapping[str, Any] = Field(default_factory=dict, repr=False)
    runtime_options: Mapping[str, Any] = Field(default_factory=dict, repr=False)

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

    @field_validator("target")
    @classmethod
    def _reject_target_credentials(cls, value: str) -> str:
        if _credential_paths({"target": value}):
            raise ValueError("WriteRequest.target must be credential-free")
        return value

    @field_validator("options")
    @classmethod
    def _copy_and_validate_options(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("write options must be mappings")
        if "binding_id" in value:
            raise ValueError("binding_id must be a top-level WriteRequest field")
        if "mode" in value:
            raise ValueError("mode must be a top-level WriteRequest field")
        if _credential_paths(value):
            raise ValueError("WriteRequest options must not contain inline credentials")
        return MappingProxyType(dict(value))

    @field_validator("runtime_options")
    @classmethod
    def _copy_and_validate_runtime_options(
        cls, value: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("runtime_options must be a mapping")
        _validate_runtime_options(value)
        return MappingProxyType(dict(value))

    @field_serializer("options", "runtime_options")
    def _serialize_options(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], credential_free_runtime_value(value))

    @property
    def credential_free_runtime_options(self) -> Mapping[str, Any]:
        """Return runtime options safe for plans, digests, and diagnostics."""
        return MappingProxyType(credential_free_runtime_value(self.runtime_options))

    @property
    def request_digest(self) -> str:
        """Return a stable digest of the credential-free control-plane request."""
        return digest(
            {
                "version": 1,
                "engine": self.engine,
                "target_kind": self.target_kind,
                "target": self.target,
                "binding_id": self.binding_id,
                "mode": self.mode.value,
                "options": dict(self.options),
                "runtime_options": dict(self.credential_free_runtime_options),
            }
        )


@PublicAPI(stability="alpha")
class DataWriteTargetRequest(BaseModel):
    """Credential-safe target request reusable by result and data writers.

    This contract belongs to the data-writing domain.  Inference may embed it
    in its output-port union, but data target semantics do not belong to the
    inference package.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sink_id: Literal["data-write-v1"] = "data-write-v1"
    target_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    target: str = Field(min_length=1, repr=False)
    binding_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_.-]+$")
    mode: WriteMode = WriteMode.APPEND
    options: Mapping[str, Any] = Field(default_factory=dict, repr=False)
    runtime_options: Mapping[str, Any] = Field(default_factory=dict, repr=False)

    @field_validator("target")
    @classmethod
    def _credential_free_target(cls, value: str) -> str:
        if _credential_paths(value):
            raise ValueError("DataWriteTargetRequest.target must be credential-free")
        return value

    @field_validator("options")
    @classmethod
    def _copy_and_validate_options(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("data write options must be mappings")
        if _credential_paths(value):
            raise ValueError("DataWriteTargetRequest.options must be credential-free")
        return MappingProxyType(dict(value))

    @field_validator("runtime_options")
    @classmethod
    def _copy_and_validate_runtime_options(
        cls, value: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("data write runtime options must be mappings")
        _validate_runtime_options(value)
        return MappingProxyType(dict(value))

    @field_serializer("options", "runtime_options")
    def _serialize_options(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], credential_free_runtime_value(value))

    def as_write_request(self) -> WriteRequest:
        """Lower this target contract into the native data WriteRequest."""
        return WriteRequest(
            engine="ray",
            target_kind=self.target_kind,
            target=self.target,
            binding_id=self.binding_id,
            mode=self.mode,
            options=self.options,
            runtime_options=self.runtime_options,
        )


@PublicAPI(stability="alpha")
class WriteDescriptor(BaseModel):
    """Credential-free capability description for one write binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine_id: str
    target_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    binding_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    api_version: Literal[1] = 1
    engine_version_spec: str = Field(min_length=1)
    binding_distribution: str = Field(min_length=1)
    binding_distribution_version: str = Field(min_length=1)
    dependency_distributions: tuple[str, ...] = ()
    capabilities: WriteCapability = Field(default_factory=WriteCapability)
    installation_hint: str | None = None

    @field_validator("engine_id")
    @classmethod
    def _normalize_engine(cls, value: str) -> str:
        return normalize_engine_id(value)

    @field_validator("dependency_distributions")
    @classmethod
    def _validate_dependencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(
            re.fullmatch(r"[A-Za-z0-9_.-]+", item) is None for item in value
        ):
            raise ValueError("dependency_distributions must be unique identifiers")
        return value

    @field_validator("installation_hint")
    @classmethod
    def _validate_installation_hint(cls, value: str | None) -> str | None:
        if value is not None and _credential_paths({"installation_hint": value}):
            raise ValueError("installation_hint must be credential-free")
        return value


@PublicAPI(stability="alpha")
class WriteReceipt(BaseModel):
    """Credential-free result of a completed native write operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal[1] = 1
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine_id: str
    binding_id: str
    target_kind: str
    target_ref: str = Field(min_length=1, repr=False)
    mode: WriteMode
    committed: bool
    rows_written: int | None = Field(default=None, ge=0)
    bytes_written: int | None = Field(default=None, ge=0)
    native_operation_ref: str | None = None
    diagnostics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict, repr=False)

    @field_validator("engine_id")
    @classmethod
    def _normalize_engine(cls, value: str) -> str:
        return normalize_engine_id(value)

    @field_validator("target_ref", "native_operation_ref")
    @classmethod
    def _reject_reference_credentials(cls, value: str | None) -> str | None:
        if value is not None and _credential_paths({"reference": value}):
            raise ValueError("write references must be credential-free")
        return value

    @field_validator("diagnostics")
    @classmethod
    def _validate_diagnostics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not item or len(item) > 512 or "\n" in item or "\r" in item
            for item in value
        ):
            raise ValueError("write diagnostics must be bounded single-line text")
        if _credential_paths({"diagnostics": value}):
            raise ValueError("write diagnostics must be credential-free")
        return value

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if _credential_paths(value):
            raise ValueError("write metadata must be credential-free")
        return MappingProxyType(dict(value))

    @field_serializer("metadata")
    def _serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(value)


@PublicAPI(stability="alpha")
class WriteError(DataSourceError):
    """Base error for writing control-plane failures."""


@PublicAPI(stability="alpha")
class WriteCapabilityError(WriteError):
    """Raised when a binding cannot satisfy a write request."""


@PublicAPI(stability="alpha")
class WriteBindingError(WriteError):
    """Raised when a native binding fails during execution."""

    def __init__(self, message: str, *, source_error_type: str | None = None) -> None:
        self.source_error_type = source_error_type
        super().__init__(message)


WriteHandle: TypeAlias = RayDataHandle | DaftDataFrameHandle


@DeveloperAPI
@dataclass(frozen=True)
class WriteExecutionContext:
    """Non-secret runtime context, not a security boundary for the request."""

    request_digest: str
    runtime_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.request_digest):
            raise ValueError("request_digest must be SHA-256 hex")
        _validate_runtime_options(self.runtime_options)
