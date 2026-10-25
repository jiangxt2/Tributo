"""Safe local events and process-local budgets; no business execution probes."""

from __future__ import annotations

import itertools
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from tributo._common.cancellation import is_cancellation
from tributo._common.logging import safe_fields

_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "tributo_log_context", default=None
)
_phases = itertools.count(1)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def emit_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any
) -> None:
    """Logging failures cannot change a result or replace a business exception."""
    if not logger.isEnabledFor(level):
        return
    values: dict[str, Any] = {
        key: os.environ[env]
        for key, env in (
            ("run_id", "TRIBUTO_RUN_ID"),
            ("attempt_id", "TRIBUTO_ATTEMPT_ID"),
            ("submission_id", "TRIBUTO_SUBMISSION_ID"),
        )
        if env in os.environ
    }
    values.update(_context.get() or {})
    values.update(fields)
    values["event"] = event
    values.setdefault("operation_id", uuid.uuid4().hex)
    try:
        ray = sys.modules.get("ray")
        if ray is not None and ray.is_initialized():
            context = ray.get_runtime_context()
            for key, method in (
                ("ray_job_id", "get_job_id"),
                ("worker_id", "get_worker_id"),
                ("actor_id", "get_actor_id"),
            ):
                getter = getattr(context, method, None)
                if callable(getter):
                    value = getter()
                    if isinstance(value, str) and value:
                        values.setdefault(key, value)
    except Exception:
        # Runtime metadata is optional, including during Ray teardown.
        pass
    try:
        logger.log(level, event, extra=safe_fields(values))
    except Exception:
        # Isolated logging boundary: business execution is outside this try.
        return


@dataclass
class StageEvent:
    status: str = "succeeded"


@contextmanager
def log_stage(
    logger: logging.Logger, phase: str, **fields: Any
) -> Iterator[StageEvent]:
    """Observe a real call, preserving results, cancellation and exceptions."""
    context = dict(_context.get() or {})
    context.update(fields)
    context.setdefault("operation_id", uuid.uuid4().hex)
    context["phase"] = phase
    context["phase_index"] = next(_phases)
    token = _context.set(context)
    started = time.monotonic()
    outcome = StageEvent()
    emit_event(logger, "tributo.stage.started", level=logging.DEBUG, status="started")
    try:
        yield outcome
    except BaseException as exc:
        cancelled = is_cancellation(exc) or not isinstance(exc, Exception)
        outcome.status = "cancelled" if cancelled else "failed"
        emit_event(
            logger,
            "tributo.stage.finished",
            level=logging.INFO if cancelled else logging.ERROR,
            status=outcome.status,
            error_type=type(exc).__name__,
            duration_ms=(time.monotonic() - started) * 1000,
        )
        raise
    else:
        emit_event(
            logger,
            "tributo.stage.finished",
            level=logging.ERROR if outcome.status == "failed" else logging.INFO,
            status=outcome.status,
            duration_ms=(time.monotonic() - started) * 1000,
        )
    finally:
        _context.reset(token)


def observed_stage(
    phase: str, *, result_status: Callable[[Any], str] | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Attach one stage boundary while preserving the callable's API metadata."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def observed(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with log_stage(logging.getLogger(function.__module__), phase) as outcome:
                result = function(*args, **kwargs)
                if result_status is not None:
                    outcome.status = result_status(result)
                return result

        return observed

    return decorate


@dataclass
class ProgressBudget:
    """Fixed physical-execution budget. Disabled detail never reads a clock."""

    count: int = 0
    emitted: int = 0
    last_time: float = 0.0

    def allow(self, *, enabled: bool, logger: logging.Logger) -> bool:
        if not enabled or not logger.isEnabledFor(logging.DEBUG):
            return False
        self.count += 1
        if self.emitted >= 100 or (self.count != 1 and self.count % 100):
            return False
        now = time.monotonic()
        if self.count != 1 and now - self.last_time < 1:
            return False
        self.last_time = now
        self.emitted += 1
        return True
