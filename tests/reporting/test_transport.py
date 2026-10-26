"""Exercise real SDK serialization, Session validation and actual request counts."""

from __future__ import annotations

import json
import logging

import pytest
from opentelemetry._logs import LogRecord
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceResponse,
)
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExportResult
from opentelemetry.sdk.resources import Resource

from tests.reporting.conftest import Endpoint
from tributo._common.reporting.config import SenderConfig
from tributo._common.reporting.transport import (
    MAX_RESPONSE_BYTES,
    DeliveryOutcome,
    ValidatingExporter,
)


def test_runtime_producer_keeps_sender_after_initial_protocol_failure(
    otlp_endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )

    from tributo._common.reporting import session as sessions

    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    otlp_endpoint.responses = [(200, "text/html", b"private-server-error")]
    sender = SenderConfig(
        endpoint=otlp_endpoint.url,
        namespaces=("tributo_runtime_test",),
        interval_seconds=0.1,
    )
    policy = json.dumps(
        {"version": 1, "role": "producer", "sender": sender.model_dump(mode="json")}
    )
    producer = sessions.bootstrap_producer(policy)
    assert producer and producer.state == "degraded" and producer._sender
    logger = logging.getLogger("tributo_runtime_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    try:
        assert not producer._closed
        logger.info("not exported", extra={"event": "tributo.runtime.recovered"})
        assert producer._sender.flush(2.0)
        assert len(otlp_endpoint.requests) == 2
        message = ExportLogsServiceRequest.FromString(otlp_endpoint.requests[-1][1])
        assert (
            message.resource_logs[0].scope_logs[0].log_records[0].body.string_value
            == "tributo.runtime.recovered"
        )
        assert producer._sender.accepted == 1
    finally:
        producer.close()


def _batch() -> list[ReadableLogRecord]:
    return [
        ReadableLogRecord(
            LogRecord(body="safe.event"), Resource({"service.name": "test"})
        )
    ]


@pytest.mark.parametrize("status", [201, 204, 301, 302, 307, 308])
def test_non_protocol_success_never_redirects_or_retries(
    otlp_endpoint: Endpoint, status: int
) -> None:
    otlp_endpoint.responses = [(status, "application/x-protobuf", b"")]
    outcomes: list[DeliveryOutcome] = []
    exporter = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url), outcomes.append
    )
    try:
        assert exporter.export(_batch()) is LogRecordExportResult.FAILURE
        assert len(otlp_endpoint.requests) == 1
        assert outcomes[0].kind == "protocol_error"
        assert outcomes[0].accepted is None
        assert outcomes[0].status_code == status
    finally:
        exporter.shutdown()


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("text/html", b"<html>secret</html>"),
        ("application/json", b'{"error":"secret"}'),
        ("application/x-protobuf", b'{"error":"secret"}'),
        ("application/x-protobuf", b"<html>secret</html>"),
        ("application/x-protobuf", b"\xff"),
        ("application/x-protobuf", b"x" * (MAX_RESPONSE_BYTES + 1)),
    ],
)
def test_invalid_200_is_unknown_and_body_is_not_logged(
    otlp_endpoint: Endpoint,
    caplog: pytest.LogCaptureFixture,
    content_type: str,
    body: bytes,
) -> None:
    otlp_endpoint.responses = [(200, content_type, body)]
    outcomes: list[DeliveryOutcome] = []
    exporter = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url), outcomes.append
    )
    try:
        with caplog.at_level(logging.WARNING):
            assert exporter.export(_batch()) is LogRecordExportResult.FAILURE
        assert len(otlp_endpoint.requests) == 1
        assert outcomes[0].kind == "protocol_error"
        assert outcomes[0].accepted is None and outcomes[0].rejected is None
        assert "secret" not in caplog.text
    finally:
        exporter.shutdown()


@pytest.mark.parametrize(
    ("rejected", "message", "kind", "success"),
    [
        (0, "", "accepted", True),
        (0, "untrusted server warning", "warning", True),
        (1, "untrusted rejection", "partial", False),
        (-1, "invalid", "protocol_error", False),
        (2, "too many", "protocol_error", False),
    ],
)
def test_partial_success_counts_and_no_whole_batch_replay(
    otlp_endpoint: Endpoint,
    caplog: pytest.LogCaptureFixture,
    rejected: int,
    message: str,
    kind: str,
    success: bool,
) -> None:
    reply = ExportLogsServiceResponse()
    reply.partial_success.rejected_log_records = rejected
    reply.partial_success.error_message = message
    otlp_endpoint.responses = [
        (200, "application/x-protobuf", reply.SerializeToString())
    ]
    outcomes: list[DeliveryOutcome] = []
    exporter = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url), outcomes.append
    )
    try:
        result = exporter.export(_batch())
        assert (result is LogRecordExportResult.SUCCESS) is success
        assert len(otlp_endpoint.requests) == 1
        assert outcomes[0].kind == kind
        if 0 <= rejected <= 1:
            assert outcomes[0].accepted == 1 - rejected
            assert outcomes[0].rejected == rejected
        else:
            assert outcomes[0].accepted is None
        assert "untrusted" not in caplog.text
    finally:
        exporter.shutdown()


def test_valid_empty_response_is_success(otlp_endpoint: Endpoint) -> None:
    outcomes: list[DeliveryOutcome] = []
    exporter = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url), outcomes.append
    )
    try:
        assert exporter.export(_batch()) is LogRecordExportResult.SUCCESS
        assert outcomes[0].accepted == 1 and outcomes[0].rejected == 0
    finally:
        exporter.shutdown()


@pytest.mark.parametrize("status", [429, 503])
def test_retryable_status_uses_one_bounded_retry_layer(
    otlp_endpoint: Endpoint, status: int
) -> None:
    otlp_endpoint.responses = [(status, "text/plain", b"untrusted error")]
    outcomes: list[DeliveryOutcome] = []
    exporter = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url, send_timeout=2.0), outcomes.append
    )
    try:
        assert exporter.export(_batch()) is LogRecordExportResult.SUCCESS
        assert len(otlp_endpoint.requests) == 2
        assert outcomes[0].accepted == 1
    finally:
        exporter.shutdown()


def test_explicit_headers_and_disabled_environment_inheritance(
    otlp_endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_REPORT_HEADERS", '{"Authorization":"test-token"}')
    monkeypatch.setenv("HTTP_PROXY", "http://unreachable.invalid:1")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/does/not/exist")
    exporter = ValidatingExporter(
        SenderConfig(
            endpoint=otlp_endpoint.url, headers_ref="env://TEST_REPORT_HEADERS"
        ),
        lambda outcome: None,
    )
    try:
        assert exporter.export(_batch()) is LogRecordExportResult.SUCCESS
        headers = {
            key.lower(): value for key, value in otlp_endpoint.requests[0][0].items()
        }
        assert headers["authorization"] == "test-token"
        assert exporter.session.trust_env is False
    finally:
        exporter.shutdown()
