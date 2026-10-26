"""Strict OTLP response validation through the exporter's public Session hook."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import requests
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceResponse,
)
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

from tributo._common.reporting.config import (
    ReportingError,
    SenderConfig,
    resolve_reference,
    validate_sdk_environment,
)

MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ProtocolResponseError(ValueError):
    """A safe, non-connection exception: SDK retries must not replay this batch."""


@dataclass
class DeliveryOutcome:
    """Adjacent-receiver outcome; none of these counts proves a Doris commit."""

    count: int
    kind: str = "transport_error"
    accepted: int | None = None
    rejected: int | None = None
    status_code: int | None = None


_OUTCOME: ContextVar[DeliveryOutcome | None] = ContextVar(
    "tributo_delivery", default=None
)


def _headers(
    config: SenderConfig, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    if config.headers_ref is None:
        return {}
    try:
        value = json.loads(
            resolve_reference(config.headers_ref, environment=environment)
        )
        if not isinstance(value, dict) or len(value) > 16:
            raise ValueError
        if any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", k) is None
            or len(k) > 128
            or len(v) > 4096
            or any(c in k + v for c in "\r\n")
            or k.lower()
            in {"host", "content-length", "content-type", "content-encoding"}
            for k, v in value.items()
        ):
            raise ValueError
        for header_value in value.values():
            header_value.encode("latin-1")
        return value
    except (ValueError, TypeError):
        raise ReportingError("reporting.config.headers") from None


class ControlledSession(requests.Session):
    """An owned Session that never follows redirects or reads implicit credentials."""

    def __init__(self, config: SenderConfig) -> None:
        super().__init__()
        self.trust_env = False
        self._config = config
        if config.proxy:
            self.proxies = {"http": config.proxy, "https": config.proxy}

    def request(
        self, method: str, url: str | bytes, *args: Any, **kwargs: Any
    ) -> requests.Response:
        outcome = _OUTCOME.get()
        if (
            args
            or outcome is None
            or url != self._config.endpoint
            or method.upper() != "POST"
        ):
            raise ProtocolResponseError("reporting.protocol.request")
        data = kwargs.get("data", b"")
        if not isinstance(data, bytes) or len(data) > self._config.max_request_bytes:
            outcome.kind = "protocol_error"
            raise ProtocolResponseError("reporting.protocol.request_size")
        kwargs["allow_redirects"] = False
        kwargs["stream"] = True
        deadline = time.monotonic() + self._config.send_timeout
        try:
            response = super().request(method, url, **kwargs)
        except requests.RequestException as error:
            # Preserve retry classification without disclosing the original URL,
            # request headers, proxy details, or underlying exception representation.
            if isinstance(error, (requests.ConnectionError, requests.Timeout)):
                raise requests.ConnectionError(
                    "reporting.transport.unavailable"
                ) from None
            outcome.kind = "protocol_error"
            raise ProtocolResponseError("reporting.protocol.http_request") from None
        except Exception:
            outcome.kind = "protocol_error"
            raise ProtocolResponseError("reporting.protocol.http_request") from None
        try:
            outcome.status_code = response.status_code
            # Do not let arbitrary server-provided reason text reach SDK logging.
            response.reason = str(response.status_code)
            if response.status_code != 200:
                if response.status_code < 400:
                    outcome.kind = "protocol_error"
                    raise ProtocolResponseError("reporting.protocol.status")
                return response
            outcome.kind = "protocol_error"
            media = (
                response.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if media != "application/x-protobuf":
                raise ProtocolResponseError("reporting.protocol.content_type")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                if (
                    time.monotonic() >= deadline
                    or len(body) + len(chunk) > MAX_RESPONSE_BYTES
                ):
                    raise ProtocolResponseError("reporting.protocol.response_limit")
                body.extend(chunk)
            reply = ExportLogsServiceResponse()
            try:
                reply.ParseFromString(bytes(body))
            except Exception:
                raise ProtocolResponseError("reporting.protocol.protobuf") from None
            rejected = reply.partial_success.rejected_log_records
            if rejected < 0 or rejected > outcome.count:
                raise ProtocolResponseError("reporting.protocol.rejected_count")
            outcome.accepted = outcome.count - rejected
            outcome.rejected = rejected
            if rejected:
                outcome.kind = "partial"
                raise ProtocolResponseError("reporting.protocol.partial_rejection")
            outcome.kind = (
                "warning" if reply.partial_success.error_message else "accepted"
            )
            return response
        except requests.RequestException as error:
            outcome.kind = "transport_error"
            if isinstance(error, (requests.ConnectionError, requests.Timeout)):
                raise requests.ConnectionError(
                    "reporting.transport.unavailable"
                ) from None
            outcome.kind = "protocol_error"
            raise ProtocolResponseError("reporting.protocol.http_response") from None
        finally:
            response.close()


class ValidatingExporter(LogRecordExporter):
    """Compose the upstream exporter and preserve per-batch partial outcomes."""

    def __init__(
        self,
        config: SenderConfig,
        observe: Callable[[DeliveryOutcome], None],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        validate_sdk_environment()
        self._lock = threading.Lock()
        self._observe = observe
        self.session = ControlledSession(config)
        self._delegate = OTLPLogExporter(
            endpoint=config.endpoint,
            certificate_file=config.ca_file,
            client_certificate_file=config.client_certificate,
            client_key_file=config.client_key,
            headers=_headers(config, environment),
            timeout=config.send_timeout,
            compression=Compression.NoCompression,
            session=self.session,
            max_request_size=config.max_request_bytes,
            meter_provider=NoOpMeterProvider(),
        )

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        with self._lock:
            outcome = DeliveryOutcome(len(batch))
            token = _OUTCOME.set(outcome)
            try:
                result = self._delegate.export(batch)
                return result
            except Exception:
                # ProtocolResponseError is classified as non-retryable by the
                # selected SDK. Unexpected SDK errors must also stay off business paths.
                return LogRecordExportResult.FAILURE
            finally:
                _OUTCOME.reset(token)
                self._observe(outcome)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        self._delegate.shutdown()
