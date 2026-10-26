"""Process-local OTel bridge with safe records and bounded lifecycle waits."""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk._logs import LoggerProvider, LogRecordLimits
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from tributo._common.reporting.config import (
    ReportingError,
    SenderConfig,
    validate_sdk_environment,
)
from tributo._common.reporting.lifecycle import bounded_call
from tributo._common.reporting.transport import DeliveryOutcome, ValidatingExporter

CONTEXT: ContextVar[dict[str, str | int | float | bool] | None] = ContextVar(
    "tributo_reporting_context", default=None
)
_BRIDGING: ContextVar[bool] = ContextVar("tributo_reporting_bridging", default=False)
_SAFE_STRING = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_EVENT = re.compile(r"[a-z][a-z0-9_.]{0,63}\Z")
_QUEUE_OWNER: ContextVar[Sender | None] = ContextVar(
    "tributo_queue_owner", default=None
)
_FIELDS = frozenset(
    {
        "event",
        "phase",
        "status",
        "operation_id",
        "execution_id",
        "run_id",
        "attempt_id",
        "submission_id",
        "ray_job_id",
        "ray_task_id",
        "ray_worker_id",
        "ray_actor_id",
        "request_id",
        "trace_id",
        "trial_id",
        "stage_id",
        "node_id",
        "rank",
        "world_size",
        "duration_ms",
        "provider_id",
        "binding_id",
        "bundle_id",
        "result_kind",
        "committed",
        "rows_written",
        "error_type",
        "probe_id",
    }
)
_DIAG = logging.getLogger("tributo.reporting.local")
_DIAG.propagate = False
_DIAG.setLevel(logging.INFO)
if not _DIAG.handlers:
    _DIAG.addHandler(logging.StreamHandler())


def safe_attributes(values: dict[str, Any]) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {}
    for key in _FIELDS:
        value = values.get(key)
        if isinstance(value, (bool, int)):
            result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
        elif isinstance(value, str) and _SAFE_STRING.fullmatch(value):
            result[key] = value
    return result


@contextmanager
def context_scope(
    _replace_fields: frozenset[str] = frozenset(), /, **attributes: Any
) -> Iterator[None]:
    inherited = {
        key: value
        for key, value in (CONTEXT.get() or {}).items()
        if key not in _replace_fields
    }
    token = CONTEXT.set({**inherited, **safe_attributes(attributes)})
    try:
        yield
    finally:
        CONTEXT.reset(token)


class _QueueDiagnostics(logging.Handler):
    _tributo_reporting_handler = True

    def __init__(self, sender: Sender) -> None:
        super().__init__(logging.WARNING)
        self.sender = sender

    def emit(self, record: logging.LogRecord) -> None:
        if (
            _QUEUE_OWNER.get() is self.sender
            and record.msg == "Queue full, dropping %s."
        ):
            with self.sender._lock:
                self.sender.overflow_events += 1
            self.sender.diagnose("sender.queue_full")


class _Bridge(logging.Handler):
    _tributo_reporting_handler = True

    def __init__(self, sender: Sender) -> None:
        super().__init__(logging.getLevelNamesMapping()[sender.config.level])
        self.sender = sender

    def emit(self, record: logging.LogRecord) -> None:
        if _BRIDGING.get() or record.name.startswith(
            ("tributo.reporting", "opentelemetry")
        ):
            return
        token = _BRIDGING.set(True)
        try:
            context = CONTEXT.get()
            attributes = safe_attributes(
                {
                    **(self.sender.attributes if context is None else {}),
                    **record.__dict__,
                    **(context or {}),
                }
            )
            event = attributes.get("event")
            if not isinstance(event, str) or not _EVENT.fullmatch(event):
                event = "log.record"
                attributes["event"] = event
            attributes["redacted"] = True
            if _SAFE_STRING.fullmatch(record.name):
                attributes["logger.name"] = record.name
            if record.exc_info and record.exc_info[1] is not None:
                attributes["error_type"] = type(record.exc_info[1]).__name__[:128]
            ray = sys.modules.get("ray")
            if ray is not None and ray.is_initialized():
                try:
                    ctx = ray.get_runtime_context()
                    methods = [
                        ("ray_job_id", "get_job_id"),
                        ("ray_worker_id", "get_worker_id"),
                        ("node_id", "get_node_id"),
                    ]
                    if self.sender.ray_worker:
                        methods.extend(
                            [
                                ("ray_task_id", "get_task_id"),
                                ("ray_actor_id", "get_actor_id"),
                            ]
                        )
                    for field, method in methods:
                        try:
                            attributes.update(
                                safe_attributes({field: getattr(ctx, method)()})
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
            severity = (
                SeverityNumber.FATAL
                if record.levelno >= 50
                else SeverityNumber.ERROR
                if record.levelno >= 40
                else SeverityNumber.WARN
                if record.levelno >= 30
                else SeverityNumber.INFO
                if record.levelno >= 20
                else SeverityNumber.DEBUG
            )
            self.sender.emit(event, attributes, severity)
        except Exception:
            self.sender.diagnose("sender.record_failed")
        finally:
            _BRIDGING.reset(token)


class Sender:
    """Own one provider/processor; never replace application global providers."""

    def __init__(
        self,
        config: SenderConfig,
        diagnose: Callable[[str], None],
        *,
        environment: Mapping[str, str] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        validate_sdk_environment()
        self.config = config
        self.attributes = safe_attributes(dict(attributes or {}))
        self.diagnose = diagnose
        self._pid = os.getpid()
        self.ray_worker = False
        self._lock = threading.Lock()
        self._handlers: list[tuple[logging.Logger, _Bridge]] = []
        self._closed = False
        self.offered = 0
        self.accepted = 0
        self.rejected = 0
        self.failures = 0
        self.protocol_failures = 0
        self.overflow_events = 0
        self._exporter = ValidatingExporter(
            config, self._observe, environment=environment
        )
        self._provider = LoggerProvider(
            resource=Resource(
                {
                    "service.name": config.service_name,
                    "service.instance.id": uuid.uuid4().hex,
                    "process.pid": self._pid,
                }
            ),
            shutdown_on_exit=False,
            meter_provider=NoOpMeterProvider(),
            log_record_limits=LogRecordLimits(
                max_attributes=64,
                max_attribute_length=256,
                max_log_record_attributes=64,
                max_log_record_attribute_length=256,
            ),
        )
        self._processor = BatchLogRecordProcessor(
            self._exporter,
            schedule_delay_millis=config.interval_seconds * 1000,
            max_export_batch_size=config.batch_size,
            max_queue_size=config.queue_size,
            export_timeout_millis=config.send_timeout * 1000,
            meter_provider=NoOpMeterProvider(),
        )
        self._provider.add_log_record_processor(self._processor)
        self._logger = self._provider.get_logger("tributo.lifecycle", "1")
        self._queue_logger = logging.getLogger("opentelemetry.sdk._shared_internal")
        self._queue_diagnostics = _QueueDiagnostics(self)
        self._queue_logger.addHandler(self._queue_diagnostics)

    def attach(self) -> None:
        if self._handlers:
            return
        parents: set[logging.Logger] = set()
        for name in sorted(set(self.config.namespaces), key=len):
            logger = logging.getLogger(name)
            ancestor: logging.Logger | None = logger
            while (
                ancestor is not None and ancestor not in parents and ancestor.propagate
            ):
                ancestor = ancestor.parent
            if ancestor in parents:
                continue
            parents.add(logger)
            handler = _Bridge(self)
            logger.addHandler(handler)
            self._handlers.append((logger, handler))

    def emit(
        self,
        body: str,
        attributes: dict[str, Any],
        severity: SeverityNumber = SeverityNumber.INFO,
    ) -> None:
        if self._closed or os.getpid() != self._pid:
            return
        with self._lock:
            self.offered += 1
        token = _QUEUE_OWNER.set(self)
        try:
            self._logger.emit(
                LogRecord(
                    timestamp=time.time_ns(),
                    observed_timestamp=time.time_ns(),
                    severity_number=severity,
                    severity_text=severity.name,
                    body=body,
                    attributes=attributes,
                )
            )
        finally:
            _QUEUE_OWNER.reset(token)

    def _observe(self, outcome: DeliveryOutcome) -> None:
        with self._lock:
            if outcome.accepted is not None:
                self.accepted += outcome.accepted
            if outcome.rejected is not None:
                self.rejected += outcome.rejected
            if outcome.kind in {"protocol_error", "partial", "transport_error"}:
                self.failures += 1
            if outcome.kind == "protocol_error":
                self.protocol_failures += 1
        if outcome.kind not in {"accepted"}:
            self.diagnose("sender." + outcome.kind)

    def flush(self, timeout: float | None = None) -> bool:
        return bounded_call(
            self._provider.force_flush,
            self.config.shutdown_timeout if timeout is None else timeout,
        )

    def close(self, timeout: float | None = None) -> bool:
        if os.getpid() != self._pid or self._closed:
            return True
        for logger, handler in self._handlers:
            logger.removeHandler(handler)
            handler.close()
        self._handlers.clear()
        self._queue_logger.removeHandler(self._queue_diagnostics)
        self._queue_diagnostics.close()
        budget = self.config.shutdown_timeout if timeout is None else max(0, timeout)
        deadline = time.monotonic() + budget
        self._closed = True
        flushed = self.flush(budget)
        remaining = max(0, deadline - time.monotonic())
        stopped = bounded_call(self._provider.shutdown, remaining)
        if not flushed or not stopped:
            self.diagnose("sender.drain_incomplete")
        return flushed and stopped

    def check_delivery(self, failures_before: int, deadline: float) -> None:
        if (
            not self.flush(max(0.01, deadline - time.monotonic()))
            or self.failures != failures_before
        ):
            raise ReportingError("reporting.probe.export_failed")
