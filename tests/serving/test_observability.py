"""Tests for E3 request correlation, trace propagation, and metrics."""

from __future__ import annotations

import logging

from tributo.serving.observability import (
    InferenceContext,
    inference_metrics_snapshot,
    log_inference_audit,
)


class _HttpRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _GrpcContext:
    def __init__(self, metadata: tuple[tuple[str, str], ...]) -> None:
        self._metadata = metadata

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata


def test_http_context_preserves_valid_w3c_traceparent() -> None:
    traceparent = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"

    context = InferenceContext.from_http(
        _HttpRequest(
            {
                "x-request-id": "request-1",
                "traceparent": traceparent,
            }
        )
    )

    assert context.request_id == "request-1"
    assert context.trace_id == "a" * 32
    assert context.traceparent == traceparent


def test_invalid_headers_are_replaced_with_safe_generated_context() -> None:
    context = InferenceContext.from_http(
        _HttpRequest(
            {
                "x-request-id": "bad value with spaces",
                "traceparent": "not-a-traceparent",
            }
        )
    )

    version, trace_id, parent_id, flags = context.traceparent.split("-")
    assert context.request_id != "bad value with spaces"
    assert version == "00"
    assert trace_id == context.trace_id
    assert len(trace_id) == 32
    assert len(parent_id) == 16
    assert flags == "01"


def test_grpc_context_reads_tuple_metadata() -> None:
    traceparent = "00-" + "c" * 32 + "-" + "d" * 16 + "-01"

    context = InferenceContext.from_grpc(
        _GrpcContext(
            (
                ("x-request-id", "grpc-request"),
                ("traceparent", traceparent),
            )
        )
    )

    assert context.request_id == "grpc-request"
    assert context.trace_id == "c" * 32


def test_audit_logging_updates_metrics(caplog) -> None:
    before = inference_metrics_snapshot()
    context = InferenceContext.from_http(_HttpRequest({}))

    with caplog.at_level(logging.INFO):
        log_inference_audit(
            logging.getLogger("tributo.test.observability"),
            context,
            bundle_id="bundle-1",
            model_version="model-1",
            status="ok",
            duration_ms=2.5,
        )

    after = inference_metrics_snapshot()
    assert after["inference_requests_total"] == before["inference_requests_total"] + 1
    assert after["inference_requests_ok"] == before["inference_requests_ok"] + 1
    assert (
        after["inference_duration_ms_total"]
        >= before["inference_duration_ms_total"] + 2.5
    )
    assert "inference_audit" in caplog.text
    assert "bundle-1" in caplog.text
