"""Bounded caller waits independent of the optional telemetry SDK."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


def bounded_call(function: Callable[[], Any], timeout: float) -> bool:
    """Attempt cleanup even after a deadline, but never wait beyond the budget."""
    done = threading.Event()
    success = [False]

    def run() -> None:
        try:
            success[0] = function() is not False
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=run, name="tributo-reporting-close", daemon=True).start()
    return done.wait(max(0, timeout)) and success[0]
