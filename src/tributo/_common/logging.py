"""Structured logging utilities.

Provides a JSON formatter for structured logging compatible with
ELK/Loki aggregation. Uses only the standard library (no structlog dependency).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from tributo.util.annotations import DeveloperAPI

# LogRecord attributes that are reserved by the logging module and should
# not be emitted as extra fields in JSON output.
_RESERVED_LOG_ATTRS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "created",
        "relativeCreated",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "pathname",
        "filename",
        "module",
        "levelno",
        "levelname",
        "msecs",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines.

    Each log record is serialized to a single-line JSON object with
    standard fields (timestamp, level, logger, message) plus any
    extra fields attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields (job_id, model_name, duration_ms, etc.)
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                log_entry[key] = value

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


@DeveloperAPI
def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    stream: Any = None,
) -> None:
    """Configure the ``tributo`` package logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format — ``"json"`` for structured JSON lines,
            ``"text"`` for human-readable plain text.
        stream: Output stream (defaults to stderr).
    """
    level_upper = level.upper()
    if not hasattr(logging, level_upper):
        raise ValueError(f"Invalid log level: {level!r}")

    root = logging.getLogger("tributo")
    root.setLevel(getattr(logging, level_upper))

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    root.addHandler(handler)
