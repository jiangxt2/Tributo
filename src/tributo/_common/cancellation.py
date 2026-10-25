"""Cancellation classification shared by business and logging boundaries."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError

from ray.exceptions import RayTaskError, TaskCancelledError


def is_cancellation(error: BaseException) -> bool:
    """Follow explicit causes and Ray wrappers, never names or implicit context."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(
            current, (asyncio.CancelledError, CancelledError, TaskCancelledError)
        ):
            return True
        candidates: list[object] = [current.__cause__]
        if isinstance(current, RayTaskError):
            candidates.append(current.cause)
        pending.extend(
            value for value in candidates if isinstance(value, BaseException)
        )
    return False
