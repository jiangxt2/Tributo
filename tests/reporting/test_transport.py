"""Exercise real SDK serialization, Session validation and actual request counts."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry._logs import LogRecord
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceResponse,
)
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExportResult
from opentelemetry.sdk.resources import Resource

from tests.reporting.conftest import Endpoint
from tributo._common.reporting.config import ReportingError, SenderConfig
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


def _reply(rejected: int, message: str) -> bytes:
    response = ExportLogsServiceResponse()
    response.partial_success.rejected_log_records = rejected
    response.partial_success.error_message = message
    return response.SerializeToString()


@pytest.fixture
def exporter(otlp_endpoint: Endpoint):
    outcomes: list[DeliveryOutcome] = []
    value = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url, send_timeout=2.0), outcomes.append
    )
    try:
        yield value, outcomes
    finally:
        value.shutdown()


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        *[
            ((s, "application/x-protobuf", b""), ("protocol_error", None, None, 1))
            for s in (201, 204, 301, 302, 307, 308)
        ],
        *[
            ((200, media, body), ("protocol_error", None, None, 1))
            for media, body in (
                ("text/html", b"<html>secret</html>"),
                ("application/json", b'{"error":"secret"}'),
                ("application/x-protobuf", b'{"error":"secret"}'),
                ("application/x-protobuf", b"<html>secret</html>"),
                ("application/x-protobuf", b"\xff"),
                ("application/x-protobuf", b"x" * (MAX_RESPONSE_BYTES + 1)),
            )
        ],
        *[
            ((200, "application/x-protobuf", _reply(n, message)), (kind, yes, no, 1))
            for n, message, kind, yes, no in (
                (0, "", "accepted", 1, 0),
                (0, "untrusted warning", "warning", 1, 0),
                (1, "untrusted rejection", "partial", 0, 1),
                (-1, "invalid", "protocol_error", None, None),
                (2, "too many", "protocol_error", None, None),
            )
        ],
        ((200, "application/x-protobuf", b""), ("accepted", 1, 0, 1)),
        *[
            ((s, "text/plain", b"untrusted error"), ("accepted", 1, 0, 2))
            for s in (429, 503)
        ],
    ],
    ids=[
        *[f"status-{s}" for s in (201, 204, 301, 302, 307, 308)],
        "html",
        "json",
        "json-as-protobuf",
        "html-as-protobuf",
        "invalid-protobuf",
        "oversized",
        "full",
        "warning",
        "partial",
        "negative-count",
        "over-count",
        "empty",
        "retry-429",
        "retry-503",
    ],
)
def test_otlp_response_and_retry_contract(
    otlp_endpoint, exporter, caplog, reply, expected
):
    delegate, outcomes = exporter
    otlp_endpoint.responses = [reply]
    batch = [ReadableLogRecord(LogRecord(body="safe.event"), Resource({}))]
    with caplog.at_level(logging.WARNING):
        result = delegate.export(batch)
    kind, accepted, rejected, count = expected
    assert (result is LogRecordExportResult.SUCCESS) == (
        kind in {"accepted", "warning"}
    )
    assert len(otlp_endpoint.requests) == count
    outcome = outcomes[0]
    assert (outcome.kind, outcome.accepted, outcome.rejected) == expected[:3]
    assert outcome.status_code == (200 if count == 2 else reply[0])
    assert "secret" not in caplog.text and "untrusted" not in caplog.text


def test_explicit_headers_and_disabled_environment_inheritance(
    otlp_endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_REPORT_HEADERS", '{"Authorization":"test-token"}')
    monkeypatch.setenv("HTTP_PROXY", "http://unreachable.invalid:1")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/does/not/exist")
    monkeypatch.setenv("SSL_CERT_FILE", "/does/not/exist")
    exporter = ValidatingExporter(
        SenderConfig(
            endpoint=otlp_endpoint.url, headers_ref="env://TEST_REPORT_HEADERS"
        ),
        lambda outcome: None,
    )
    try:
        batch = [ReadableLogRecord(LogRecord(body="safe.event"), Resource({}))]
        assert exporter.export(batch) is LogRecordExportResult.SUCCESS
        headers = {
            key.lower(): value for key, value in otlp_endpoint.requests[0][0].items()
        }
        assert headers["authorization"] == "test-token"
        assert exporter.session.trust_env is False
    finally:
        exporter.shutdown()


@pytest.mark.parametrize("slow", ["headers", "body"])
def test_total_response_deadline_cancels_io_and_releases_export_thread(
    otlp_endpoint, slow
):
    outcomes = []
    exporter = ValidatingExporter(
        SenderConfig(endpoint=otlp_endpoint.url, send_timeout=0.25), outcomes.append
    )
    otlp_endpoint.slow = slow
    otlp_endpoint.responses = [
        (200, "application/x-protobuf", _reply(0, "slow-warning"))
    ]
    batch = [ReadableLogRecord(LogRecord(body="safe.event"), Resource({}))]
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            began = time.monotonic()
            assert (
                executor.submit(exporter.export, batch).result(timeout=0.8)
                is LogRecordExportResult.FAILURE
            )
            assert time.monotonic() - began < 0.8
            assert otlp_endpoint.disconnected.wait(0.8)
            assert len(otlp_endpoint.requests) == 1
            assert (
                outcomes[0].kind == "transport_error" and outcomes[0].accepted is None
            )
            otlp_endpoint.slow = None
            assert (
                executor.submit(exporter.export, batch).result(timeout=0.8)
                is LogRecordExportResult.SUCCESS
            )
            assert len(otlp_endpoint.requests) == 2 and outcomes[-1].accepted == 1
    finally:
        exporter.shutdown()


@pytest.mark.parametrize(
    "tls",
    [
        {"ca_file": "/private/test-secret-ca.pem"},
        {
            "client_certificate": "/private/test-secret-cert.pem",
            "client_key": "/private/test-secret-key.pem",
        },
    ],
)
def test_tls_configuration_failure_keeps_paths_private(tls):
    with pytest.raises(ReportingError, match="^reporting.config.tls$"):
        ValidatingExporter(
            SenderConfig(endpoint="https://localhost:1/v1/logs", **tls),
            lambda outcome: None,
        )
