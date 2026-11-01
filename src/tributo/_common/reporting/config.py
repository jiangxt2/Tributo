"""Validated, credential-reference-only reporting configuration."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tributo.exceptions import TributoError
from tributo.util.annotations import PublicAPI

POLICY_ENV = "_TRIBUTO_REPORTING_POLICY"
WORKER_HOOK = "tributo._common.reporting.worker.setup"
ENABLED_ENV = "TRIBUTO_LOG_REPORTING_ENABLED"
CONFIG_ENV = "TRIBUTO_LOG_REPORTING_CONFIG"
MODE_ENV = "TRIBUTO_LOG_REPORTING_MODE"
KNOWN_ENV = (ENABLED_ENV, CONFIG_ENV, MODE_ENV)
MAX_CONFIG_BYTES = 32768
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_REFERENCE = re.compile(r"env://([A-Za-z_][A-Za-z0-9_]{0,127})\Z")
_NAMESPACE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
PositiveSeconds = Annotated[
    float, Field(gt=0, le=300, allow_inf_nan=False, strict=True)
]


@PublicAPI(stability="alpha")
class ReportingError(TributoError):
    """A reporting failure described without sensitive input or exception text."""


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


def _validate_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError("invalid reporting endpoint") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or len(value) > 2048
        or (port is not None and not 0 < port < 65536)
    ):
        raise ValueError("invalid reporting endpoint")
    return value


def resolve_reference(
    reference: str | None,
    *,
    empty: bool = False,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve an explicit trusted environment reference without echoing it."""
    if reference is None:
        return ""
    match = _REFERENCE.fullmatch(reference)
    if not match:
        raise ReportingError("reporting.config.credential_reference")
    value = (environment if environment is not None else os.environ).get(match.group(1))
    if value is None or (not value and not empty) or len(value) > MAX_CONFIG_BYTES:
        raise ReportingError("reporting.config.credential_unavailable")
    return value


@PublicAPI(stability="alpha")
class SenderConfig(_Config):
    """Process-local sender settings; contains references, never resolved secrets."""

    endpoint: str = Field(default="http://127.0.0.1:4318/v1/logs", repr=False)
    headers_ref: str | None = Field(default=None, repr=False)
    ca_file: str | None = Field(default=None, repr=False)
    client_certificate: str | None = Field(default=None, repr=False)
    client_key: str | None = Field(default=None, repr=False)
    proxy: str | None = Field(default=None, repr=False)
    trusted_network: bool = Field(default=False, strict=True)
    namespaces: tuple[str, ...] = ("tributo",)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    service_name: str = "tributo"
    queue_size: int = Field(default=2048, ge=1, le=65536, strict=True)
    batch_size: int = Field(default=128, ge=1, le=4096, strict=True)
    interval_seconds: PositiveSeconds = 1.0
    send_timeout: PositiveSeconds = 5.0
    shutdown_timeout: PositiveSeconds = 10.0
    max_request_bytes: int = Field(
        default=4 * 1024 * 1024, ge=1024, le=16 * 1024 * 1024, strict=True
    )

    @model_validator(mode="after")
    def _validate_sender(self) -> SenderConfig:
        _validate_url(self.endpoint)
        endpoint = urlsplit(self.endpoint)
        try:
            local = ipaddress.ip_address(endpoint.hostname or "").is_loopback
        except ValueError:
            local = endpoint.hostname == "localhost"
        if endpoint.scheme == "http" and not local and not self.trusted_network:
            raise ValueError("remote plaintext reporting requires trusted_network")
        if self.batch_size > self.queue_size:
            raise ValueError("reporting batch exceeds queue")
        if not 1 <= len(self.namespaces) <= 32 or any(
            len(n) > 128
            or not _NAMESPACE.fullmatch(n)
            or n == "root"
            or n.startswith("opentelemetry")
            or n.startswith("tributo.reporting")
            for n in self.namespaces
        ):
            raise ValueError("invalid reporting namespaces")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.service_name):
            raise ValueError("invalid reporting service name")
        if bool(self.client_certificate) != bool(self.client_key):
            raise ValueError("reporting client certificate/key must be paired")
        if self.headers_ref and not _REFERENCE.fullmatch(self.headers_ref):
            raise ValueError("invalid reporting credential reference")
        if self.proxy:
            _validate_url(self.proxy)
        return self


@PublicAPI(stability="alpha")
class DorisProfile(_Config):
    """Owner-only Doris SQL/readback or Stream Load credential references."""

    host: str = Field(repr=False)
    port: int = Field(default=9030, ge=1, le=65535, strict=True)
    database: str
    table: str = "otel_logs"
    username_ref: str = Field(repr=False)
    password_ref: str = Field(repr=False)
    http_endpoint: str | None = Field(default=None, repr=False)
    ca_file: str | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _validate_profile(self) -> DorisProfile:
        if not _IDENTIFIER.fullmatch(self.database) or not _IDENTIFIER.fullmatch(
            self.table
        ):
            raise ValueError("invalid reporting table identifier")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,253}", self.host):
            raise ValueError("invalid reporting database host")
        if not all(
            _REFERENCE.fullmatch(r) for r in (self.username_ref, self.password_ref)
        ):
            raise ValueError("invalid reporting credential reference")
        if self.http_endpoint:
            _validate_url(self.http_endpoint)
        return self


@PublicAPI(stability="alpha")
class CollectorConfig(_Config):
    """A pre-provisioned Collector; startup never downloads or compiles it."""

    binary: str = Field(repr=False)
    sha256: str
    version: str = "0.163.2"
    otlp_bind: str = "127.0.0.1:4318"
    health_bind: str = "127.0.0.1:13133"
    tls_certificate: str | None = Field(default=None, repr=False)
    tls_key: str | None = Field(default=None, repr=False)
    client_ca_file: str | None = Field(default=None, repr=False)
    memory_limit_mib: int = Field(default=128, ge=32, le=2048, strict=True)

    @model_validator(mode="after")
    def _validate_collector(self) -> CollectorConfig:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("invalid Collector checksum")
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("invalid Collector version")
        if bool(self.tls_certificate) != bool(self.tls_key):
            raise ValueError("Collector TLS certificate/key must be paired")
        for address in (self.otlp_bind, self.health_bind):
            _validate_url("http://" + address)
            if not urlsplit("http://" + address).port:
                raise ValueError("Collector bind requires a port")
        return self


@PublicAPI(stability="alpha")
class ReportingConfig(_Config):
    """Explicit opt-in and owner configuration for one reporting execution domain."""

    enabled: bool = Field(default=False, strict=True)
    mode: Literal["managed", "external"] = "managed"
    sender: SenderConfig = Field(default_factory=SenderConfig)
    collector: CollectorConfig | None = None
    read_profile: DorisProfile | None = None
    write_profile: DorisProfile | None = None
    startup_timeout: PositiveSeconds = 30.0

    @model_validator(mode="after")
    def _validate_owner(self) -> ReportingConfig:
        if not self.enabled:
            return self
        if self.read_profile is None:
            raise ValueError("reporting readback profile is required")
        if self.mode == "managed":
            if (
                self.collector is None
                or self.write_profile is None
                or not self.write_profile.http_endpoint
            ):
                raise ValueError("managed reporting needs a binary and write profile")
            if (self.read_profile.database, self.read_profile.table) != (
                self.write_profile.database,
                self.write_profile.table,
            ):
                raise ValueError("reporting profiles must address the same log table")
        return self


def load_config(environment: Mapping[str, str] | None = None) -> ReportingConfig:
    """Read opt-in first; disabled ignores all unrelated reporting configuration."""

    def read(name: str, default: str | None = None) -> str | None:
        if environment is not None and name in environment:
            return environment[name]
        return os.environ.get(name, default)

    enabled = read(ENABLED_ENV, "0")
    if enabled not in {"0", "1"}:
        raise ReportingError("reporting.config.enabled")
    if enabled == "0":
        return ReportingConfig()
    try:
        path = read(CONFIG_ENV)
        if not path or Path(path).stat().st_size > MAX_CONFIG_BYTES:
            raise ValueError
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "enabled" in payload:
            raise ValueError
        mode = read(MODE_ENV, payload.get("mode", "managed"))
        return ReportingConfig.model_validate(
            {**payload, "enabled": True, "mode": mode}
        )
    except (OSError, ValueError, TypeError):
        raise ReportingError("reporting.config.invalid") from None


def validate_sdk_environment() -> None:
    """Reject ambient settings instead of temporarily changing global environment."""
    suffixes = (
        "ENDPOINT",
        "HEADERS",
        "TIMEOUT",
        "COMPRESSION",
        "CERTIFICATE",
        "CLIENT_CERTIFICATE",
        "CLIENT_KEY",
    )
    keys = [
        p + s
        for p in ("OTEL_EXPORTER_OTLP_", "OTEL_EXPORTER_OTLP_LOGS_")
        for s in suffixes
    ]
    keys += [
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER",
    ]
    if any(os.environ.get(key, "") != "" for key in keys):
        raise ReportingError("reporting.config.otel_exporter_conflict")
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() not in {"", "false"}:
        raise ReportingError("reporting.config.sdk_disabled_conflict")
