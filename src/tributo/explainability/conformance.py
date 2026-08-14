"""Structural conformance checks for third-party explainability adapters."""

from __future__ import annotations

from typing import Any

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
def validate_adapter_conformance(adapter_cls: type[Any]) -> None:
    """Fail closed when an adapter does not implement the public SPI."""
    adapter_id = getattr(adapter_cls, "adapter_id", None)
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("explainer adapter_id must be a non-empty string")
    if getattr(adapter_cls, "api_version", None) != 1:
        raise ValueError(
            f"explainer adapter {adapter_id!r} has unsupported api_version "
            f"{getattr(adapter_cls, 'api_version', None)!r}; expected 1"
        )
    adapter_version = getattr(adapter_cls, "adapter_version", None)
    if not isinstance(adapter_version, str) or not adapter_version:
        raise ValueError(
            f"explainer adapter {adapter_id!r} must declare adapter_version"
        )
    for method_name in (
        "supports",
        "prepare",
        "explain_batch",
        "summarize",
    ):
        if not callable(getattr(adapter_cls, method_name, None)):
            raise ValueError(
                f"explainer adapter {adapter_id!r} is missing {method_name}"
            )


__all__ = ["validate_adapter_conformance"]
