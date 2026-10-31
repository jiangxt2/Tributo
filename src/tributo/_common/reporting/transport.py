"""Strict OTLP response validation through the exporter's public Session hook."""

from __future__ import annotations

import asyncio
import io
import json
import re
import ssl
import threading
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp
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
        self._tls: ssl.SSLContext | bool = True
        if any(
            urlsplit(url).scheme == "https"
            for url in (config.endpoint, config.proxy or "")
        ):
            try:
                self._tls = ssl.create_default_context(
                    cafile=config.ca_file or requests.certs.where()
                )
                if config.client_certificate:
                    self._tls.load_cert_chain(
                        config.client_certificate, config.client_key
                    )
            except (OSError, ValueError):
                self.close()
                raise ReportingError("reporting.config.tls") from None

    async def _post(
        self,
        data: bytes,
        headers: Mapping[str, str],
        timeout: float,
        outcome: DeliveryOutcome,
    ) -> tuple[requests.Response, bytes]:
        # Run in the SDK's synchronous export thread. The async timeout cancels
        # the socket read itself; no detached response-reader thread survives it.
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout, ceil_threshold=float("inf")),
            trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),
        ) as client:
            async with client.post(
                self._config.endpoint,
                data=data,
                headers=headers,
                proxy=self._config.proxy,
                ssl=self._tls,
                allow_redirects=False,
            ) as incoming:
                outcome.status_code = incoming.status
                response = requests.Response()
                response.status_code = incoming.status
                # Never forward a server-provided reason string into SDK logs.
                response.reason = str(incoming.status)
                response.headers.update(incoming.headers)
                body = bytearray()
                if incoming.status == 200:
                    outcome.kind = "protocol_error"
                    media = (
                        response.headers.get("Content-Type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if media != "application/x-protobuf":
                        raise ProtocolResponseError("reporting.protocol.content_type")
                    async for chunk in incoming.content.iter_chunked(65536):
                        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise ProtocolResponseError(
                                "reporting.protocol.response_limit"
                            )
                        body.extend(chunk)
                response.raw = io.BytesIO(body)
        await asyncio.sleep(0)
        return response, bytes(body)

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
        headers = self.headers.copy()
        headers.update(kwargs.get("headers") or {})
        try:
            # The SDK passes its remaining export budget on every retry.
            timeout = min(
                self._config.send_timeout,
                float(kwargs.get("timeout", self._config.send_timeout)),
            )
            if timeout <= 0:
                raise TimeoutError
            response, body = asyncio.run(self._post(data, headers, timeout, outcome))
        except (aiohttp.ClientConnectionError, TimeoutError):
            outcome.kind = "transport_error"
            raise requests.ConnectionError("reporting.transport.unavailable") from None
        except ProtocolResponseError:
            raise
        except Exception:
            outcome.kind = "protocol_error"
            raise ProtocolResponseError("reporting.protocol.http_request") from None
        try:
            if response.status_code != 200:
                if response.status_code < 400:
                    outcome.kind = "protocol_error"
                    raise ProtocolResponseError("reporting.protocol.status")
                return response
            outcome.kind = "protocol_error"
            reply = ExportLogsServiceResponse()
            try:
                reply.ParseFromString(body)
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
