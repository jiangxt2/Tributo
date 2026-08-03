"""Request correlation and W3C trace-context helpers for E3 serving."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TRACEPARENT_PATTERN = re.compile(
    r"^(?P<version>[0-9a-fA-F]{2})-"
    r"(?P<trace_id>[0-9a-fA-F]{32})-"
    r"(?P<parent_id>[0-9a-fA-F]{16})-"
    r"(?P<flags>[0-9a-fA-F]{2})$"
)

_metrics_lock = Lock()
_metrics: dict[str, float | int] = {
    "inference_requests_total": 0,
    "inference_requests_ok": 0,
    "inference_requests_invalid_argument": 0,
    "inference_requests_error": 0,
    "inference_duration_ms_total": 0.0,
}


def _header_value(value: Any) -> str | None:
    """Convert an HTTP/gRPC metadata value to a text header value."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore")
    return str(value)


def _safe_request_id(value: Any) -> str:
    candidate = _header_value(value)
    if candidate and _ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _trace_context(value: Any) -> tuple[str, str]:
    """Return ``(traceparent, trace_id)`` using a valid W3C context."""
    candidate = _header_value(value)
    if candidate:
        match = _TRACEPARENT_PATTERN.fullmatch(candidate)
        if match:
            version = match.group("version").lower()
            trace_id = match.group("trace_id").lower()
            parent_id = match.group("parent_id").lower()
            flags = match.group("flags").lower()
            if version != "ff" and trace_id != "0" * 32 and parent_id != "0" * 16:
                return f"{version}-{trace_id}-{parent_id}-{flags}", trace_id

    trace_id = uuid.uuid4().hex
    parent_id = uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{parent_id}-01", trace_id


@dataclass(frozen=True, slots=True)
class InferenceContext:
    """Correlation metadata shared by HTTP, gRPC, and audit logs."""

    request_id: str
    traceparent: str
    trace_id: str

    @classmethod
    def from_http(cls, request: Any) -> "InferenceContext":
        """Extract ``X-Request-ID`` and W3C ``traceparent`` from HTTP."""
        headers = getattr(request, "headers", {})
        traceparent, trace_id = _trace_context(headers.get("traceparent"))
        return cls(
            request_id=_safe_request_id(headers.get("x-request-id")),
            traceparent=traceparent,
            trace_id=trace_id,
        )

    @classmethod
    def from_grpc(cls, context: Any) -> "InferenceContext":
        """Extract correlation metadata from gRPC invocation metadata."""
        values: dict[str, str] = {}
        metadata_fn = getattr(context, "invocation_metadata", None)
        if callable(metadata_fn):
            for item in metadata_fn() or ():
                key = _header_value(getattr(item, "key", None))
                value = _header_value(getattr(item, "value", None))
                if key is None and isinstance(item, (tuple, list)) and len(item) == 2:
                    key = _header_value(item[0])
                    value = _header_value(item[1])
                if key and value is not None:
                    values[key.lower()] = value
        traceparent, trace_id = _trace_context(values.get("traceparent"))
        return cls(
            request_id=_safe_request_id(values.get("x-request-id")),
            traceparent=traceparent,
            trace_id=trace_id,
        )

    def response_fields(
        self,
        *,
        bundle_id: str | None,
        model_version: str | None,
    ) -> dict[str, str | None]:
        """Return non-sensitive correlation fields for a response body."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "traceparent": self.traceparent,
            "bundle_id": bundle_id,
            "model_version": model_version,
        }

    def response_headers(
        self,
        *,
        bundle_id: str | None,
        model_version: str | None,
    ) -> dict[str, str]:
        """Return HTTP headers for correlation and version association."""
        headers = {
            "X-Request-ID": self.request_id,
            "traceparent": self.traceparent,
        }
        if bundle_id is not None:
            headers["X-Bundle-ID"] = bundle_id
        if model_version is not None:
            headers["X-Model-Version"] = model_version
        return headers


def log_inference_audit(
    logger: Any,
    context: InferenceContext,
    *,
    bundle_id: str | None,
    model_version: str | None,
    status: str,
    duration_ms: float | None = None,
) -> None:
    """Write a structured, non-sensitive inference audit record."""
    record_inference_metrics(status=status, duration_ms=duration_ms)
    logger.info(
        "inference_audit request_id=%s trace_id=%s bundle_id=%s "
        "model_version=%s status=%s duration_ms=%s",
        context.request_id,
        context.trace_id,
        bundle_id,
        model_version,
        status,
        None if duration_ms is None else round(duration_ms, 2),
    )


def record_inference_metrics(*, status: str, duration_ms: float | None) -> None:
    """Record process-local inference counters and aggregate latency.

    Deployments can export the snapshot to their metrics backend without
    making an optional metrics SDK a serving dependency.
    """
    with _metrics_lock:
        _metrics["inference_requests_total"] += 1
        status_key = f"inference_requests_{status}"
        if status_key in _metrics:
            _metrics[status_key] += 1
        if duration_ms is not None:
            _metrics["inference_duration_ms_total"] += max(duration_ms, 0.0)


def inference_metrics_snapshot() -> dict[str, float | int]:
    """Return a consistent snapshot of the process-local serving metrics."""
    with _metrics_lock:
        return dict(_metrics)
