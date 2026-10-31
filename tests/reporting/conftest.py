"""Owned loopback OTLP endpoint; this is protocol evidence, not a Doris fake."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class Endpoint:
    url: str = ""
    responses: list[tuple[int, str, bytes]] = field(default_factory=list)
    requests: list[tuple[dict[str, str], bytes]] = field(default_factory=list)
    slow: str | None = None
    disconnected: threading.Event = field(default_factory=threading.Event)


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
            try:
                if endpoint.slow == "headers":
                    reply = (
                        f"HTTP/1.0 {status} OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(reply)}\r\n\r\n".encode()
                        + reply
                    )
                else:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(reply)))
                    self.send_header("Retry-After", "0")
                    self.send_header("Location", endpoint.url + "/redirected")
                    self.end_headers()
                if endpoint.slow:
                    for byte in reply:
                        time.sleep(0.05)
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                else:
                    self.wfile.write(reply)
            except (BrokenPipeError, ConnectionResetError):
                endpoint.disconnected.set()

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


def reporting_kernel(policy, emit):
    """Same load/predict/close workload for direct and real Ray actor checks."""
    from types import SimpleNamespace

    import numpy as np

    from tests.inference.test_bundle_predictor import _bindings, _Runtime
    from tributo.inference.kernel import KernelBatchPredictor

    class Kernel(_Runtime):
        def predict(self, inputs):
            emit("predict")
            return super().predict(inputs)

        def close(self):
            emit("close")
            super().close()

    def create():
        emit("load")
        return Kernel()

    factory = SimpleNamespace(factory_id="test.reporting-policy", create=create)
    batch = {
        key: np.array([value])
        for key, value in (
            ("feature_a", 1.0),
            ("feature_b", 2.0),
            ("entity_id", 7),
        )
    }
    return KernelBatchPredictor(factory, *_bindings(), _reporting_policy=policy), batch


async def reporting_serving_request(kind, identity, emit, *, invalid=False):
    """Real serving entry points with one small model double, reused by Ray IT."""
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from tests.serving.test_grpc_deployment import _RpcContext
    from tests.serving.test_model_deployment import _http_request
    from tributo.serving.grpc_deployment import gRPCInferenceService
    from tributo.serving.model_deployment import ONNXModel
    from tributo.serving.proto import inference_pb2

    def predict(outputs, inputs):
        emit()
        return [np.array([0.25, 0.75])]

    cls = ONNXModel if kind == "http" else gRPCInferenceService
    service = cls.__new__(cls)
    service._runtime = None
    service._session = SimpleNamespace(run=predict)
    service._input_names = ("input",)
    service._input_dtypes = service._input_shapes = ()
    service.input_name = "input"
    service.model_path = "test-model"
    service._bundle_id = service._model_version = "test-bundle"
    traceparent = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
    metadata = (("x-request-id", identity), ("traceparent", traceparent))
    if kind == "http":
        return await service(
            _http_request(
                b'{"features":[[0,1]]}',
                headers=[(key.encode(), value.encode()) for key, value in metadata],
            )
        )
    request = inference_pb2.PredictRequest()
    if not invalid:
        request.inputs.add(
            schema_version=1,
            name="input",
            shape=[2, 1],
            datatype="float32",
            data=[0, 1],
        )
    context = _RpcContext(metadata)
    if kind == "stream":
        return service.StreamPredict(request, context)
    if kind == "grpc":
        return await service.Predict(request, context)

    async def requests():
        await asyncio.sleep(0)
        yield request

    return await service.BatchPredict(requests(), context)
