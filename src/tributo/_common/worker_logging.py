"""Import-safe Ray setup hook and serializable callable composition."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from tributo._common.logging_policy import (
    USER_HOOK_ENV,
    LoggingPolicy,
    resolve_policy,
)

_dispatched: tuple[int, str, str | None] | None = None


def setup() -> None:
    """Ray 2.59 resolves this dotted function name and calls it once per worker."""
    global _dispatched
    from tributo._common.logging import apply_logging_policy

    policy = resolve_policy()
    original = os.environ.get(USER_HOOK_ENV)
    key = (os.getpid(), policy.serialize(), original)
    if _dispatched == key:
        return
    apply_logging_policy(policy)
    if original is not None:
        module_name, separator, name = original.rpartition(".")
        if not separator:
            raise ValueError("Ray setup hook must name an importable function")
        hook = getattr(importlib.import_module(module_name), name)
        if not callable(hook):
            raise ValueError("Ray setup hook must name an importable function")
        hook()
    _dispatched = key


@dataclass(frozen=True)
class _CombinedSetup:
    policy: LoggingPolicy
    original: Callable[[], None]
    __name__: ClassVar[str] = "tributo_composed_setup"

    def __call__(self) -> None:
        from tributo._common.logging import apply_logging_policy

        apply_logging_policy(self.policy)
        self.original()


def compose_setup(
    policy: LoggingPolicy, original: Callable[[], None]
) -> Callable[[], None]:
    if isinstance(original, _CombinedSetup):
        if original.policy != policy:
            raise ValueError("Conflicting application logging setup")
        return original
    return _CombinedSetup(policy, original)
