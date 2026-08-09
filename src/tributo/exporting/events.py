"""Immutable operation events emitted after durable bundle publication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.util.annotations import PublicAPI


def _event_id(bundle_id: str, manifest_sha256: str) -> str:
    payload = json.dumps(
        {
            "schema_version": 1,
            "event_kind": "bundle.published",
            "bundle_id": bundle_id,
            "manifest_sha256": manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"evt-{hashlib.sha256(payload).hexdigest()}"


@PublicAPI(stability="beta")
class OperationEvent(BaseModel):
    """Versioned, storage-neutral notification of a committed bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    event_kind: Literal["bundle.published"] = "bundle.published"
    event_id: str = Field(..., min_length=1)
    occurred_at: datetime
    bundle_id: str = Field(..., min_length=1)
    canonical_uri: str = Field(..., min_length=1)
    manifest_sha256: str = Field(..., min_length=64, max_length=64)
    source_kind: str | None = None
    correlation_ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _require_utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("correlation_ids")
    @classmethod
    def _allow_stable_correlation_ids_only(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        allowed = {"run_id", "request_id", "execution_id"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported correlation ID keys: {unknown!r}")
        if any(not item for item in value.values()):
            raise ValueError("correlation ID values must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_event_id(self) -> OperationEvent:
        expected = _event_id(self.bundle_id, self.manifest_sha256)
        if self.event_id != expected:
            raise ValueError("event_id does not match the immutable publication fact")
        return self

    @classmethod
    def bundle_published(
        cls,
        *,
        occurred_at: datetime,
        bundle_id: str,
        canonical_uri: str,
        manifest_sha256: str,
        source_kind: str | None = None,
        correlation_ids: dict[str, str] | None = None,
    ) -> OperationEvent:
        """Build the stable event for a committed manifest."""
        return cls(
            event_id=_event_id(bundle_id, manifest_sha256),
            occurred_at=occurred_at,
            bundle_id=bundle_id,
            canonical_uri=canonical_uri,
            manifest_sha256=manifest_sha256,
            source_kind=source_kind,
            correlation_ids=correlation_ids or {},
        )
