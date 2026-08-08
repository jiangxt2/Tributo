"""Versioned operation events derived from committed bundle facts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class OperationEvent(BaseModel):
    """Immutable event envelope for a committed operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    event_kind: Literal["bundle.published"]
    event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurred_at: datetime
    bundle_id: str
    canonical_uri: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: str | None = None
    correlation_ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @classmethod
    def bundle_published(
        cls,
        *,
        manifest: dict[str, Any],
        manifest_sha256: str,
        correlation_ids: dict[str, str] | None = None,
    ) -> OperationEvent:
        """Derive a stable ``bundle.published`` event from committed facts."""
        event_kind: Literal["bundle.published"] = "bundle.published"
        bundle_id = str(manifest["bundle_id"])
        canonical_uri = str(manifest["canonical_uri"])
        source_info = manifest.get("source_info") or {}
        occurred_at = datetime.fromisoformat(str(manifest["created_at"]))
        stable = {
            "schema_version": 1,
            "event_kind": event_kind,
            "bundle_id": bundle_id,
            "canonical_uri": canonical_uri,
            "manifest_sha256": manifest_sha256,
        }
        payload = json.dumps(
            stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return cls(
            event_kind=event_kind,
            event_id=hashlib.sha256(payload).hexdigest(),
            occurred_at=occurred_at,
            bundle_id=bundle_id,
            canonical_uri=canonical_uri,
            manifest_sha256=manifest_sha256,
            source_kind=source_info.get("source_kind"),
            correlation_ids=dict(correlation_ids or {}),
        )


__all__ = ["OperationEvent"]
