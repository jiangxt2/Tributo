"""Lightweight validation contract for external algorithm metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@runtime_checkable
class AlgorithmContractValidator(Protocol):
    """Validate one canonical JSON object without importing algorithm code."""

    api_version: int
    schema_digest: str

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a validated canonical JSON object or raise an exception."""


__all__ = ["AlgorithmContractValidator"]
