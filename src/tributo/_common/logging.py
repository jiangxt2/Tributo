"""Application-owned logging with bounded fields and safe coexistence."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from tributo._common.logging_policy import (
    LOG_ENV_KEYS,
    POLICY_ENV,
    LoggingPolicy,
    resolve_policy,
)
from tributo.util.annotations import DeveloperAPI

_HANDLER_NAME = "tributo.console"
_lock = threading.RLock()
_applied: tuple[int, LoggingPolicy] | None = None
_last_inbound: tuple[int, LoggingPolicy] | None = None
_explicit_policy: tuple[int, LoggingPolicy] | None = None
_warned: set[str] = set()
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_FIELDS = (
    "event",
    "phase",
    "status",
    "operation_id",
    "run_id",
    "attempt_id",
    "submission_id",
    "ray_job_id",
    "request_id",
    "trace_id",
    "execution_id",
    "stage_id",
    "node_id",
    "trial_id",
    "phase_index",
    "rank",
    "world_size",
    "duration_ms",
    "worker_id",
    "task_id",
    "actor_id",
    "provider_id",
    "binding_id",
    "request_digest",
    "plan_id",
    "split_id",
    "bundle_id",
    "model_id",
    "model_version",
    "mode",
    "format",
    "namespace_count",
    "metadata_fetched",
    "batch_index",
    "step_index",
    "rows",
    "rows_written",
    "bytes_written",
    "committed",
    "result_kind",
    "source_status",
    "sink_id",
    "resource_kind",
    "error_type",
    "mismatch_category",
    "field_count",
    "suppressed_progress",
    "fields_truncated",
    "outcome",
)
_CREDENTIAL = re.compile(
    r"(?i)([a-z0-9_-]*(?:password|secret|token|api_key|access_key)[a-z0-9_-]*"
    r"\s*[:=]\s*)([^\s,;]+)"
)
_URI_USER = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")


class _ConsoleHandler(logging.StreamHandler):
    """Identity establishes ownership; a caller's matching name is insufficient."""


def safe_fields(values: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Accept only bounded simple values; never stringify arbitrary objects."""
    result: dict[str, str | int | float | bool] = {}
    truncated = False
    for key in _FIELDS:
        value = values.get(key)
        if value is None:
            continue
        if type(value) is bool:
            result[key] = value
        elif type(value) is int and -(2**63) <= value < 2**63:
            result[key] = value
        elif type(value) is float and math.isfinite(value):
            result[key] = value
        elif type(value) is str:
            limit = 64 if key == "event" else 128
            if len(value) > limit:
                truncated = True
            elif _IDENTIFIER.fullmatch(value):
                result[key] = value
    if truncated:
        result["fields_truncated"] = True
    return result


def _message(record: logging.LogRecord) -> str:
    if record.__dict__.get("event") == "tributo.serving.request.finished":
        values = safe_fields(record.__dict__)
        return "inference_audit " + " ".join(
            f"{key}={values.get(key)}"
            for key in (
                "request_id",
                "trace_id",
                "bundle_id",
                "model_version",
                "status",
                "duration_ms",
            )
        )
    if record.__dict__.get("event"):
        return str(safe_fields(record.__dict__).get("event", "tributo.event"))
    message = record.getMessage()
    message = _URI_USER.sub(r"\1<redacted>@", message)
    return (
        _CREDENTIAL.sub(r"\1<redacted>", message)
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


class JsonFormatter(logging.Formatter):
    """Format one line with a shared allowlist, never a raw exception chain."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _message(record),
            **_record_fields(record),
        }
        return json.dumps(log_entry, ensure_ascii=False, allow_nan=False)


class TextFormatter(logging.Formatter):
    """Render the same safe context as JSON in a stable, readable order."""

    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(
            f"{key}={value}" for key, value in _record_fields(record).items()
        )
        return (
            f"{self.formatTime(record)} {record.levelname} {record.name} {_message(record)}"
            + (f" {fields}" if fields else "")
        )


def _record_fields(record: logging.LogRecord) -> dict[str, str | int | float | bool]:
    values = dict(record.__dict__)
    if record.exc_info and record.exc_info[1] is not None:
        values["error_type"] = type(record.exc_info[1]).__name__
    return safe_fields(values)


def _handlers(logger: logging.Logger) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    current: logging.Logger | None = logger
    while current is not None:
        handlers.extend(current.handlers)
        if not current.propagate:
            break
        current = current.parent
    return handlers


def apply_logging_policy(
    policy: LoggingPolicy, *, stream: Any = None, force: bool = False
) -> None:
    """Apply once per process; later user hook changes survive wrapper entry."""
    global _applied, _last_inbound
    key = (os.getpid(), policy)
    with _lock:
        if not force and _applied is not None and _last_inbound == key:
            return
        changed = _applied != key
        if not force and _applied == key:
            return
        if _applied is not None and _applied[0] == os.getpid():
            next_names = {name for name, _ in policy.logger_levels}
            for name, old_level in _applied[1].logger_levels:
                if name not in next_names:
                    old_logger = logging.getLogger(name)
                    if old_logger.level == logging.getLevelName(old_level):
                        old_logger.setLevel(logging.NOTSET)
        roots = {"tributo", *(name.split(".")[0] for name, _ in policy.logger_levels)}
        for namespace in sorted(roots):
            logger = logging.getLogger(namespace)
            logger.setLevel(policy.level if namespace == "tributo" else "INFO")
            owned = [h for h in logger.handlers if isinstance(h, _ConsoleHandler)]
            external = [
                h
                for h in _handlers(logger)
                if not isinstance(h, (_ConsoleHandler, logging.NullHandler))
            ]
            if owned:
                handler = owned[0]
                handler.setLevel(logging.DEBUG)
                handler.setFormatter(
                    JsonFormatter() if policy.log_format == "json" else TextFormatter()
                )
                if stream is not None and isinstance(handler, logging.StreamHandler):
                    handler.setStream(stream)
            elif not external:
                handler = _ConsoleHandler(stream if stream is not None else sys.stderr)
                handler.name = _HANDLER_NAME
                handler.setLevel(logging.DEBUG)
                handler.setFormatter(
                    JsonFormatter() if policy.log_format == "json" else TextFormatter()
                )
                logger.addHandler(handler)
                logger.propagate = False
        for name, level in policy.logger_levels:
            logging.getLogger(name).setLevel(level)
        for name in roots | {name for name, _ in policy.logger_levels}:
            logger = logging.getLogger(name)
            handlers = _handlers(logger)
            if (
                logger.isEnabledFor(logging.DEBUG)
                and handlers
                and all(h.level > logging.DEBUG for h in handlers)
                and name not in _warned
            ):
                try:
                    logger.warning(
                        "Application DEBUG is filtered by existing handler thresholds"
                    )
                except Exception:
                    pass
                _warned.add(name)
        _applied = key
        if not force:
            _last_inbound = key
        from tributo._common.logging_events import emit_event

        if not changed:
            return
        emit_event(
            logging.getLogger("tributo"),
            "tributo.logging.configured",
            level=logging.DEBUG,
            phase="logging",
            status="configured",
            mode="owned"
            if any(
                isinstance(h, _ConsoleHandler)
                for h in logging.getLogger("tributo").handlers
            )
            else "external",
            format=policy.log_format,
            namespace_count=len(roots),
        )


@DeveloperAPI
def configure_logging(
    level: str | None = None,
    log_format: str | None = None,
    stream: Any = None,
) -> None:
    """Configure logging while preserving caller-owned output handlers.

    Explicit parameters override known environment options in this process.
    The default format is JSON; owned CLI entry points may default to text.
    """
    global _explicit_policy
    # Override raw leaf options before parsing them. An invalid lower-priority
    # shell option must not defeat an explicit SDK argument. Transport snapshots
    # still undergo full schema validation before any explicit override.
    raw = {name: os.environ[name] for name in LOG_ENV_KEYS if name in os.environ}
    if POLICY_ENV in os.environ:
        policy = resolve_policy()
    else:
        if level is not None:
            raw["TRIBUTO_LOG_LEVEL"] = level
        if log_format is not None:
            raw["TRIBUTO_LOG_FORMAT"] = log_format
        policy = resolve_policy(raw)
    if level is not None:
        policy = replace(policy, level=level.upper())
    if log_format is not None:
        policy = replace(policy, log_format=log_format)
    policy = LoggingPolicy.deserialize(policy.serialize())
    apply_logging_policy(policy, stream=stream, force=True)
    _explicit_policy = (os.getpid(), policy)


def bootstrap_logging(*, default_format: str = "json") -> LoggingPolicy:
    """Initialize an owned entry point before its first business operation."""
    policy = (
        _explicit_policy[1]
        if _explicit_policy is not None
        and _applied is not None
        and _explicit_policy == _applied
        and _applied[0] == os.getpid()
        else resolve_policy(default_format=default_format)
    )
    apply_logging_policy(policy)
    return policy


def active_logging_policy() -> LoggingPolicy:
    """Use explicit current-process configuration for owned child workers."""
    if _applied is not None and _applied[0] == os.getpid():
        return _applied[1]
    return resolve_policy()
