"""Stable engine identifiers shared by bounded reads and writes."""

from __future__ import annotations

from tributo.util.annotations import DeveloperAPI

ENGINE_ALIASES = {
    "ray": "tributo.ray_data",
    "tributo.ray_data": "tributo.ray_data",
    "daft": "tributo.daft",
    "tributo.daft": "tributo.daft",
}


@DeveloperAPI
def normalize_engine_id(value: str) -> str:
    """Normalize a public engine alias or reject an unknown engine."""
    normalized = ENGINE_ALIASES.get(value)
    if normalized is None:
        raise ValueError(
            "engine must be one of: ray, tributo.ray_data, daft, tributo.daft"
        )
    return normalized


__all__ = ["normalize_engine_id"]
