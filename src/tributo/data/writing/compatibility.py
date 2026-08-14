"""Compatibility-only adapters for legacy DataConnector write calls."""

from __future__ import annotations

from typing import Any

from tributo.data.contracts.handles import RayDataHandle
from tributo.data.contracts.modes import WriteMode
from tributo.data.writing.contracts import WriteRequest


def ray_connector_write_request(
    *,
    dataset: Any,
    target_kind: str,
    target: str,
    options: dict[str, Any] | None = None,
    runtime_options: dict[str, Any] | None = None,
    mode: WriteMode | str = WriteMode.OVERWRITE,
) -> tuple[WriteRequest, RayDataHandle]:
    """Convert legacy Ray write parameters without executing or owning Dataset."""
    normalized_options = dict(options or {})
    binding_id = normalized_options.pop("binding_id", None)
    raw_mode = normalized_options.pop("mode", mode)
    try:
        mode = raw_mode if isinstance(raw_mode, WriteMode) else WriteMode(str(raw_mode))
    except (TypeError, ValueError):
        raise ValueError("legacy write mode must be 'append' or 'overwrite'") from None
    return (
        WriteRequest(
            engine="ray",
            target_kind=target_kind,
            target=target,
            binding_id=binding_id,
            mode=mode,
            options=normalized_options,
            runtime_options=dict(runtime_options or {}),
        ),
        RayDataHandle(dataset),
    )


def execute_ray_connector_write(
    *,
    dataset: Any,
    target_kind: str,
    target: str,
    options: dict[str, Any] | None = None,
    runtime_options: dict[str, Any] | None = None,
    mode: WriteMode | str = WriteMode.OVERWRITE,
) -> None:
    """Execute one legacy Ray write through the native write Gateway."""
    from tributo.data.writing.builtins import default_write_gateway

    request, handle = ray_connector_write_request(
        dataset=dataset,
        target_kind=target_kind,
        target=target,
        options=options,
        runtime_options=runtime_options,
        mode=mode,
    )
    default_write_gateway().execute(request, handle)
