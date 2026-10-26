"""Owned loopback OTLP endpoint; this is protocol evidence, not a Doris fake."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class Endpoint:
    url: str = ""
    responses: list[tuple[int, str, bytes]] = field(default_factory=list)
    requests: list[tuple[dict[str, str], bytes]] = field(default_factory=list)


@pytest.fixture
def otlp_endpoint() -> Iterator[Endpoint]:
    endpoint = Endpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            endpoint.requests.append((dict(self.headers), body))
            status, content_type, reply = (
                endpoint.responses.pop(0)
                if endpoint.responses
                else (200, "application/x-protobuf", b"")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(reply)))
            self.send_header("Retry-After", "0")
            self.send_header("Location", endpoint.url + "/redirected")
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    endpoint.url = f"http://127.0.0.1:{server.server_port}/v1/logs"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(autouse=True)
def clean_reporting_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from tributo._common.reporting.config import KNOWN_ENV, POLICY_ENV

    for name in (*KNOWN_ENV, POLICY_ENV, "OTEL_SDK_DISABLED"):
        monkeypatch.delenv(name, raising=False)
