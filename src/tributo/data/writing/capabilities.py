"""Capability contract for native-engine write bindings."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from tributo.data.contracts.modes import WriteMode
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class WriteCapability(BaseModel):
    """Credential-free capabilities that a write binding can guarantee."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supported_modes: frozenset[WriteMode] = frozenset()
    supported_options: frozenset[str] = frozenset()
    distributed: bool = False
    native_metrics: bool = False
    requires_existing_target: bool = False
    can_create_target: bool = False
    # Empty-input behavior is target- and engine-specific.  Do not advertise
    # support until a binding has a tested native contract for it.
    supports_empty_input: bool = False

    @field_validator("supported_modes", mode="before")
    @classmethod
    def _normalize_modes(cls, value: Any) -> frozenset[WriteMode]:
        return frozenset(WriteMode(item) for item in value)

    @field_validator("supported_options")
    @classmethod
    def _validate_options(cls, value: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(r"[a-z][a-z0-9_.-]*", item) is None for item in value):
            raise ValueError("supported_options must contain identifiers")
        return value
