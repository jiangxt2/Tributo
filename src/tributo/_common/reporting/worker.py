"""Ray setup module. Ordinary imports without a producer policy are inert."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from tributo._common.reporting.session import bootstrap_producer


@dataclass(frozen=True)
class CombinedHook:
    """Preserve a trusted local Ray callable after reporting initialization."""

    hook: Callable[[], Any]
    __name__: ClassVar[str] = "tributo_reporting_combined_setup"

    def __call__(self) -> None:
        bootstrap_producer(ray_worker=True)
        self.hook()


def setup() -> None:
    """Ray's string hook is a callable import path, not a module side effect."""
    bootstrap_producer(ray_worker=True)
    original = os.environ.get("_TRIBUTO_REPORTING_USER_HOOK")
    if original:
        module, separator, attribute = original.rpartition(".")
        if not separator or original == __name__ + ".setup":
            raise ValueError("reporting.policy.user_hook")
        getattr(importlib.import_module(module), attribute)()
