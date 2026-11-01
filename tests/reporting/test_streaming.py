"""Actual async iterator boundaries and request-local reporting correlation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from starlette.requests import Request

from tributo._common.reporting.sender import CONTEXT
from tributo.serving import streaming_deployment as module


class Stream(module.StreamingInferenceService):
    async def _generate_stream(
        self, input_data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        try:
            await asyncio.sleep(0)
            if input_data.get("fail"):
                raise ValueError("test-private-failure")
            yield str((CONTEXT.get() or {}).get("request_id"))
            await asyncio.sleep(0)
            yield str((CONTEXT.get() or {}).get("request_id"))
        finally:
            await asyncio.sleep(0)
            self.closed_context = dict(CONTEXT.get() or {})


def request(identity: str, *, fail: bool = False, invalid: bool = False) -> Request:
    async def receive() -> dict[str, Any]:
        return {
            "type": "http.request",
            "body": b"invalid" if invalid else b'{"fail":true}' if fail else b"{}",
        }

    return Request(
        {"type": "http", "headers": [(b"x-request-id", identity.encode())]}, receive
    )


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> list[logging.LogRecord]:
    from types import SimpleNamespace

    from tributo._common.reporting import session as sessions
    from tributo.observability import SenderConfig
    from tributo.serving import grpc_deployment

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            record.correlated_request_id = (CONTEXT.get() or {}).get("request_id")
            record.correlated_trace_id = (CONTEXT.get() or {}).get("trace_id")
            record.reported_context = dict(CONTEXT.get() or {})
            records.append(record)

    owner = sessions.ReportingSession(producer=SenderConfig())
    monkeypatch.setattr(
        owner, "_sender", SimpleNamespace(attributes={"operation_id": "service-op"})
    )
    monkeypatch.setattr(sessions, "_ACTIVE", owner)
    for target in (module, grpc_deployment):
        monkeypatch.setattr(target.logger, "level", logging.DEBUG)
    handler = Capture()
    for target in (module, grpc_deployment):
        target.logger.addHandler(handler)
    yield records
    for target in (module, grpc_deployment):
        target.logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_concurrent_streams_finish_on_iteration_without_context_leak(
    enabled: list[logging.LogRecord],
) -> None:
    service = Stream()
    responses = [await service(request(identity)) for identity in ("first", "second")]
    assert not enabled  # constructing a response is not stream completion

    async def consume(index: int) -> list[str]:
        return [value async for value in responses[index].body_iterator]

    first, second = await asyncio.gather(consume(0), consume(1))
    assert first == ["data: first\n\n", "data: first\n\n", "data: [DONE]\n\n"]
    assert second == ["data: second\n\n", "data: second\n\n", "data: [DONE]\n\n"]
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert {(r.request_id, r.status) for r in ends} == {
        ("first", "succeeded"),
        ("second", "succeeded"),
    }
    assert CONTEXT.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["complete", "close", "cancel", "invalid", "failure"])
async def test_grpc_stream_terminal_preserves_computation_audit_and_metrics(
    enabled, mode
):
    from tests.reporting.conftest import reporting_serving_request
    from tributo._common.reporting.sender import context_scope
    from tributo.serving.observability import inference_metrics_snapshot

    before = inference_metrics_snapshot()
    error = RuntimeError("test model failure")

    def predict():
        if mode == "failure":
            raise error

    def terminals():
        return [
            r
            for r in enabled
            if getattr(r, "event", "") == "tributo.serving.stream.finished"
        ]

    stream = await reporting_serving_request(
        "stream", "stream-request", predict, invalid=mode == "invalid"
    )
    if mode == "failure":
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)
        assert caught.value is error
    elif mode == "invalid":
        assert [item async for item in stream] == []
    else:
        assert (await asyncio.create_task(anext(stream))).predictions == [0.25]
        assert not terminals() and CONTEXT.get() is None
        with context_scope(request_id="closing-caller", trace_id="c" * 32):
            if mode == "complete":
                assert len([item async for item in stream]) == 1
            elif mode == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.create_task(stream.athrow(asyncio.CancelledError()))
            else:
                await asyncio.create_task(stream.aclose())
            assert CONTEXT.get()["request_id"] == "closing-caller"
    assert len(terminals()) == 1
    terminal = terminals()[0]
    expected = {
        "complete": "succeeded",
        "invalid": "invalid_argument",
        "failure": "failed",
    }
    assert terminal.status == expected.get(mode, "cancelled")
    assert (
        terminal.correlated_request_id == "stream-request"
        and terminal.correlated_trace_id == "1" * 32
    )
    audits = [
        r
        for r in enabled
        if getattr(r, "event", "") == "tributo.serving.inference.audit"
    ]
    assert len(audits) == 1
    after = inference_metrics_snapshot()
    assert after["inference_requests_total"] == before["inference_requests_total"] + 1
    assert (
        after["inference_requests_" + audits[0].status]
        == before["inference_requests_" + audits[0].status] + 1
    )
    assert CONTEXT.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sse", "grpc"])
@pytest.mark.parametrize("execution", [None, "original-execution"])
async def test_cross_task_close_restores_each_callers_context(
    enabled: list[logging.LogRecord], kind, execution
) -> None:
    from tests.reporting.conftest import reporting_serving_request
    from tributo._common.reporting.sender import context_scope

    identity = {"run_id": "original-run", "execution_id": execution}
    with context_scope(**identity):
        stream = (
            (await Stream()(request("stream-a"))).body_iterator
            if kind == "sse"
            else await reporting_serving_request("stream", "stream-a", lambda: None)
        )
        await asyncio.create_task(anext(stream))
    assert CONTEXT.get() is None
    with context_scope(
        request_id="closing-task",
        run_id="foreign-run",
        execution_id="foreign-execution",
    ):
        caller = dict(CONTEXT.get())
        await asyncio.create_task(stream.aclose())
        assert CONTEXT.get() == caller
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert len(ends) == 1 and ends[0].status == "cancelled"
    assert ends[0].correlated_request_id == "stream-a"
    assert ends[0].reported_context["run_id"] == "original-run"
    assert ends[0].reported_context.get("execution_id") == execution


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spec", "failure_at"),
    [
        ("2.0", "http.response.body"),
        ("2.4", "http.response.body"),
        ("2.4", "http.response.start"),
    ],
)
async def test_asgi_send_disconnect_and_foreign_task_cleanup(
    enabled: list[logging.LogRecord], spec: str, failure_at: str
) -> None:
    from starlette.requests import ClientDisconnect

    from tributo._common.reporting.sender import context_scope

    service = Stream()
    with context_scope(run_id="original-run", execution_id="original-execution"):
        response = await service(request("send-failure"))
    sent = asyncio.Event()

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == failure_at:
            assert CONTEXT.get() == caller
            sent.set()
            if spec == "2.4":
                raise OSError("test client disconnected")
            await asyncio.Event().wait()

    async def receive() -> dict[str, Any]:
        await sent.wait()
        return {"type": "http.disconnect"}

    with context_scope(
        request_id="foreign-request",
        run_id="foreign-run",
        execution_id="foreign-execution",
    ):
        caller = dict(CONTEXT.get())
        call = response({"type": "http", "asgi": {"spec_version": spec}}, receive, send)
        if spec == "2.4":
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(call, 2)
        else:
            await asyncio.wait_for(call, 2)
        assert CONTEXT.get() == caller
    if failure_at == "http.response.body":
        assert service.closed_context["run_id"] == "original-run"
        assert service.closed_context["execution_id"] == "original-execution"
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert len(ends) == 1 and ends[0].status == "cancelled"
    assert ends[0].correlated_request_id == "send-failure"
    assert ends[0].reported_context["run_id"] == "original-run"
    assert ends[0].reported_context["execution_id"] == "original-execution"


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [ValueError, asyncio.CancelledError])
async def test_user_cleanup_runs_in_request_scope_without_masking_disconnect(
    enabled: list[logging.LogRecord],
    cleanup_error,
) -> None:
    seen = []

    class CleanupStream(module.StreamingInferenceService):
        async def _generate_stream(
            self, data: dict[str, Any]
        ) -> AsyncGenerator[str, None]:
            try:
                yield "token"
            finally:
                seen.append((CONTEXT.get() or {}).get("request_id"))
                raise cleanup_error("test cleanup error")

    response = await CleanupStream()(request("cleanup-request"))
    await asyncio.create_task(anext(response.body_iterator))
    await asyncio.create_task(response.body_iterator.aclose())
    assert seen == ["cleanup-request"] and CONTEXT.get() is None
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert len(ends) == 1 and ends[0].status == "cancelled"


@pytest.mark.asyncio
async def test_stream_close_failure_and_invalid_json_have_one_correct_terminal(
    enabled: list[logging.LogRecord],
) -> None:
    service = Stream()
    response = await service(request("cancel"))
    await anext(response.body_iterator)
    await response.body_iterator.aclose()
    failed = await service(request("failure", fail=True))
    result = [value async for value in failed.body_iterator]
    assert result[-1] == "data: [DONE]\n\n"
    invalid = await service(request("invalid", invalid=True))
    assert invalid.status_code == 400
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert [(r.request_id, r.status) for r in ends] == [
        ("cancel", "cancelled"),
        ("failure", "failed"),
        ("invalid", "invalid_argument"),
    ]
    assert CONTEXT.get() is None
