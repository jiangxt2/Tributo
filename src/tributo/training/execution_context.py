"""Broker-neutral worker control reconstruction for training execution.

This module is deliberately separate from :mod:`tributo.integrations.broker`.
The Broker API describes driver-side delivery handling; worker cancellation
and progress are an optional training execution SPI and must not expand that
public transport contract.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tributo.util.annotations import DeveloperAPI

EXECUTION_CONTEXT_ENV = "TRIBUTO_EXECUTION_CONTEXT"
EXECUTION_CONTEXT_SCHEMA = "tributo.execution-context"
EXECUTION_CONTEXT_VERSION = 1


@DeveloperAPI
class TrainingCancelledError(RuntimeError):
    """Raised inside a training worker after a confirmed cooperative cancel."""


@DeveloperAPI
@runtime_checkable
class CancellationChecker(Protocol):
    """Optional worker-side cancellation control."""

    def is_cancelled(self, job_id: str) -> bool:
        """Return whether ``job_id`` has been cancelled."""
        ...


@DeveloperAPI
@runtime_checkable
class TrainingEventReporter(Protocol):
    """Optional worker/driver progress sink reconstructed from a factory."""

    def report_phase(self, job_id: str, phase: str) -> None:
        """Report a non-terminal training phase."""
        ...

    def report_metrics(
        self, job_id: str, metrics: Mapping[str, Any], progress: float | None = None
    ) -> None:
        """Report one live metric sample."""
        ...


@DeveloperAPI
@dataclass(frozen=True)
class TrainingControlSpec:
    """Serializable reference to an independently installed control factory."""

    factory_ref: str
    job_id: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.factory_ref, str) or ":" not in self.factory_ref:
            raise ValueError("factory_ref must use the 'module:callable' form")
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        _validate_options(self.options, "options")

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible control metadata."""
        return {
            "factory_ref": self.factory_ref,
            "job_id": self.job_id,
            "options": dict(self.options),
        }


@DeveloperAPI
@dataclass(frozen=True)
class ExecutionContext:
    """Serializable factories for optional worker-side training controls."""

    schema: str = EXECUTION_CONTEXT_SCHEMA
    version: int = EXECUTION_CONTEXT_VERSION
    cancellation: TrainingControlSpec | None = None
    event_reporter: TrainingControlSpec | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the context as JSON-compatible data."""
        result: dict[str, Any] = {"schema": self.schema, "version": self.version}
        if self.cancellation is not None:
            result["cancellation"] = self.cancellation.as_dict()
        if self.event_reporter is not None:
            result["event_reporter"] = self.event_reporter.as_dict()
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ExecutionContext:
        """Parse context data without importing a referenced factory."""
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("execution context must be an object")
        allowed = {"schema", "version", "cancellation", "event_reporter"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown execution context fields: {sorted(map(str, unknown))}"
            )
        schema = value.get("schema", EXECUTION_CONTEXT_SCHEMA)
        version = value.get("version", EXECUTION_CONTEXT_VERSION)
        if schema != EXECUTION_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported execution context schema: {schema!r}")
        if version != EXECUTION_CONTEXT_VERSION:
            raise ValueError(f"unsupported execution context version: {version!r}")
        return cls(
            cancellation=_parse_spec(value.get("cancellation"), "cancellation"),
            event_reporter=_parse_spec(value.get("event_reporter"), "event_reporter"),
        )

    @classmethod
    def from_environment(cls) -> ExecutionContext:
        """Parse context from the Ray worker environment."""
        raw = os.environ.get(EXECUTION_CONTEXT_ENV)
        if not raw:
            return cls()
        return cls.from_mapping(json.loads(raw))

    def build_cancellation_checker(self) -> CancellationChecker | None:
        """Rebuild and structurally validate the configured checker."""
        if self.cancellation is None:
            return None
        candidate = _invoke_factory(self.cancellation)
        if not isinstance(candidate, CancellationChecker):
            raise TypeError(
                "cancellation factory did not return a cancellation checker"
            )
        return candidate

    def build_event_reporter(self) -> TrainingEventReporter | None:
        """Rebuild and structurally validate the configured reporter."""
        if self.event_reporter is None:
            return None
        candidate = _invoke_factory(self.event_reporter)
        if not isinstance(candidate, TrainingEventReporter):
            raise TypeError("event reporter factory did not return a training reporter")
        return candidate


def _parse_spec(value: Any, section: str) -> TrainingControlSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"execution context {section} must be an object")
    allowed = {"factory_ref", "job_id", "options"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {section} fields: {sorted(map(str, unknown))}")
    options = value.get("options", {})
    if not isinstance(options, Mapping):
        raise TypeError(f"execution context {section}.options must be an object")
    factory_ref = value.get("factory_ref")
    job_id = value.get("job_id")
    if not isinstance(factory_ref, str):
        raise TypeError(f"execution context {section}.factory_ref must be a string")
    if not isinstance(job_id, str):
        raise TypeError(f"execution context {section}.job_id must be a string")
    return TrainingControlSpec(factory_ref, job_id, dict(options))


def _invoke_factory(spec: TrainingControlSpec) -> object:
    module_name, separator, object_path = spec.factory_ref.partition(":")
    if not separator or not module_name or not object_path:
        raise ValueError("factory_ref must use the 'module:callable' form")
    candidate: object = importlib.import_module(module_name)
    for component in object_path.split("."):
        candidate = getattr(candidate, component)
    if not callable(candidate):
        raise TypeError(
            f"execution control factory is not callable: {spec.factory_ref}"
        )
    return candidate(job_id=spec.job_id, options=dict(spec.options))


def _normalize_key(key: str) -> str:
    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake)
    return snake.replace("-", "_").lower()


def _is_reference_key(key: str) -> bool:
    return key.endswith(("_env", "_env_var", "_ref", "_reference"))


def _is_sensitive_key(key: str) -> bool:
    exact = {
        "api_key",
        "authorization",
        "credential",
        "dsn",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_access_key",
        "token",
        "access_token",
    }
    endings = (
        "_api_key",
        "_credential",
        "_password",
        "_passwd",
        "_private_key",
        "_secret",
        "_secret_access_key",
        "_token",
    )
    prefixes = (
        "access_token_",
        "authorization_",
        "password_",
        "passwd_",
        "private_key_",
        "secret_access_key_",
    )
    return key in exact or key.endswith(endings) or key.startswith(prefixes)


def _validate_options(value: Any, path: str) -> None:
    """Validate options as JSON-safe metadata containing no inline secrets."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(
                    f"execution context option key at {path} must be a string"
                )
            key = _normalize_key(raw_key)
            if _is_sensitive_key(key) and not _is_reference_key(key):
                if child not in (None, ""):
                    raise ValueError(
                        f"inline secret or credential is forbidden at {path}.{raw_key}"
                    )
            _validate_options(child, f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_options(child, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if "-----BEGIN" in value.upper() and "PRIVATE KEY-----" in value.upper():
            raise ValueError(f"inline private-key secret is forbidden at {path}")
        if "://" in value:
            from urllib.parse import urlsplit

            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError(f"URI userinfo credential is forbidden at {path}")
        return
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"execution context option at {path} is not JSON-safe")


__all__ = [
    "CancellationChecker",
    "ExecutionContext",
    "TrainingCancelledError",
    "TrainingControlSpec",
    "TrainingEventReporter",
]
