"""Engine-neutral target planning for bounded data writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from tributo.data.contracts.modes import WriteMode
from tributo.data.engine_ids import normalize_engine_id
from tributo.data.refs import _credential_paths
from tributo.data.runtime_credentials import credential_free_runtime_value
from tributo.data.writing.contracts import WriteRequest
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class LogicalWritePlan:
    """Credential-free target plan consumed by one engine write binding."""

    plan_version: int
    provider_id: str
    request_digest: str
    engine_id: str
    target_kind: str
    target: str
    mode: WriteMode
    options: Mapping[str, Any]
    runtime_options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.plan_version != 1:
            raise ValueError("LogicalWritePlan.plan_version must be 1")
        if not self.provider_id:
            raise ValueError("LogicalWritePlan.provider_id must not be empty")
        object.__setattr__(self, "engine_id", normalize_engine_id(self.engine_id))
        if re.fullmatch(r"[0-9a-f]{64}", self.request_digest) is None:
            raise ValueError("LogicalWritePlan.request_digest must be SHA-256 hex")
        if not self.target:
            raise ValueError("LogicalWritePlan.target must not be empty")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
        object.__setattr__(
            self,
            "runtime_options",
            MappingProxyType(credential_free_runtime_value(self.runtime_options)),
        )
        if _credential_paths(
            {
                "provider_id": self.provider_id,
                "target": self.target,
                "options": self.options,
                "runtime_options": self.runtime_options,
            }
        ):
            raise ValueError("LogicalWritePlan must be credential-free")


@runtime_checkable
@DeveloperAPI
class WriteTargetProvider(Protocol):
    """Normalize one logical target without executing a write."""

    @property
    def provider_id(self) -> str:
        """Return a stable, credential-free provider identifier."""

    def plan(self, request: WriteRequest) -> LogicalWritePlan:
        """Return an immutable logical plan for ``request``."""


@DeveloperAPI
class GenericWriteTargetProvider:
    """Default target planner for built-in format target kinds."""

    def __init__(self, target_kind: str) -> None:
        self._target_kind = target_kind

    @property
    def provider_id(self) -> str:
        return f"tributo.target.{self._target_kind}"

    def plan(self, request: WriteRequest) -> LogicalWritePlan:
        if request.target_kind != self._target_kind:
            raise ValueError("target provider does not match request target kind")
        return LogicalWritePlan(
            plan_version=1,
            provider_id=self.provider_id,
            request_digest=request.request_digest,
            engine_id=request.engine,
            target_kind=request.target_kind,
            target=request.target,
            mode=request.mode,
            options=request.options,
            runtime_options=request.runtime_options,
        )


__all__ = [
    "GenericWriteTargetProvider",
    "LogicalWritePlan",
    "WriteTargetProvider",
]
