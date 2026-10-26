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
        await asyncio.sleep(0)
        if input_data.get("fail"):
            raise ValueError("test-private-failure")
        yield str((CONTEXT.get() or {}).get("request_id"))
        await asyncio.sleep(0)
        yield str((CONTEXT.get() or {}).get("request_id"))


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
    from contextlib import contextmanager

    from tributo._common.reporting.sender import context_scope

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            record.correlated_request_id = (CONTEXT.get() or {}).get("request_id")
            records.append(record)

    @contextmanager
    def scope(**values: Any) -> Any:
        with context_scope(**values):
            yield

    monkeypatch.setattr(module, "active_session", lambda: object())
    monkeypatch.setattr(module, "reporting_context", scope)
    monkeypatch.setattr(module, "bootstrap_producer", lambda: None)
    monkeypatch.setattr(module.logger, "level", logging.DEBUG)
    handler = Capture()
    module.logger.addHandler(handler)
    yield records
    module.logger.removeHandler(handler)


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
async def test_cross_task_close_restores_each_callers_context(
    enabled: list[logging.LogRecord],
) -> None:
    from tributo._common.reporting.sender import context_scope

    response = await Stream()(request("stream-a"))
    assert (
        await asyncio.create_task(anext(response.body_iterator)) == "data: stream-a\n\n"
    )
    assert CONTEXT.get() is None
    with context_scope(request_id="closing-task"):
        await asyncio.create_task(response.body_iterator.aclose())
        assert CONTEXT.get()["request_id"] == "closing-task"
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert len(ends) == 1 and ends[0].status == "cancelled"
    assert ends[0].request_id == ends[0].correlated_request_id == "stream-a"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ["2.0", "2.4"])
async def test_asgi_send_disconnect_and_foreign_task_cleanup(
    enabled: list[logging.LogRecord], spec: str
) -> None:
    from starlette.requests import ClientDisconnect

    from tributo._common.reporting.sender import context_scope

    response = await Stream()(request("send-failure"))
    sent = asyncio.Event()

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            assert CONTEXT.get() is None
            sent.set()
            if spec == "2.4":
                raise OSError("test client disconnected")
            await asyncio.Event().wait()

    async def receive() -> dict[str, Any]:
        await sent.wait()
        return {"type": "http.disconnect"}

    call = response({"type": "http", "asgi": {"spec_version": spec}}, receive, send)
    if spec == "2.4":
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(call, 2)
    else:
        await asyncio.wait_for(call, 2)
    with context_scope(request_id="foreign-request"):
        await asyncio.create_task(response.body_iterator.aclose())
        assert CONTEXT.get()["request_id"] == "foreign-request"
    ends = [r for r in enabled if getattr(r, "event", "").endswith("finished")]
    assert len(ends) == 1 and ends[0].status == "cancelled"
    assert ends[0].correlated_request_id == "send-failure"


@pytest.mark.asyncio
async def test_user_cleanup_runs_in_request_scope_without_masking_disconnect(
    enabled: list[logging.LogRecord],
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
                raise ValueError("test cleanup error")

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
