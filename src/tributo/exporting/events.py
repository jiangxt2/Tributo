"""Versioned operation events derived from committed bundle facts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.util.annotations import PublicAPI


def _event_id(
    bundle_id: str,
    canonical_uri: str,
    manifest_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "schema_version": 1,
            "event_kind": "bundle.published",
            "bundle_id": bundle_id,
            "canonical_uri": canonical_uri,
            "manifest_sha256": manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@PublicAPI(stability="beta")
class OperationEvent(BaseModel):
    """Immutable event envelope for a committed operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    event_kind: Literal["bundle.published"] = "bundle.published"
    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurred_at: datetime
    bundle_id: str = Field(..., min_length=1)
    canonical_uri: str = Field(..., min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
        expected = _event_id(
            self.bundle_id,
            self.canonical_uri,
            self.manifest_sha256,
        )
        if self.event_id != expected:
            raise ValueError("event_id does not match the immutable publication fact")
        return self

    @classmethod
    def bundle_published(
        cls,
        *,
        manifest: dict[str, Any] | None = None,
        manifest_sha256: str,
        occurred_at: datetime | None = None,
        bundle_id: str | None = None,
        canonical_uri: str | None = None,
        source_kind: str | None = None,
        correlation_ids: dict[str, str] | None = None,
    ) -> OperationEvent:
        """Derive a stable ``bundle.published`` event from committed facts."""
        if manifest is not None:
            bundle_id = str(manifest["bundle_id"])
            canonical_uri = str(manifest["canonical_uri"])
            occurred_at = datetime.fromisoformat(str(manifest["created_at"]))
            source_info = manifest.get("source_info") or {}
            source_kind = source_info.get("source_kind")
        if occurred_at is None or bundle_id is None or canonical_uri is None:
            raise ValueError(
                "manifest or occurred_at, bundle_id, and canonical_uri are required"
            )
        return cls(
            event_id=_event_id(bundle_id, canonical_uri, manifest_sha256),
            occurred_at=occurred_at,
            bundle_id=bundle_id,
            canonical_uri=canonical_uri,
            manifest_sha256=manifest_sha256,
            source_kind=source_kind,
            correlation_ids=dict(correlation_ids or {}),
        )


__all__ = ["OperationEvent"]
