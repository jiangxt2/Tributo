"""Direct assertions on the real lifecycle logging boundaries."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from tests.inference.test_executor import _plan, _receipt
from tests.training.exporters.test_manager import _make_plan, _make_source
from tributo._common.logging_policy import POLICY_ENV, LoggingPolicy
from tributo.data.writing.contracts import WriteReceipt
from tributo.data.writing.gateway import WriteGateway
from tributo.exporting.executor import ExportManager
from tributo.exporting.models import BundleOutputConfig, ExportTarget
from tributo.inference.kernel import KernelBatchPredictor
from tributo.serving.streaming_deployment import StreamingInferenceService

pytestmark = pytest.mark.usefixtures("isolated_logging")


@pytest.mark.parametrize(
    "committed,expected", [(True, "succeeded"), (False, "not_committed")]
)
def test_write_terminal_uses_actual_receipt_commit_state(committed, expected, caplog):
    receipt = WriteReceipt(
        request_digest="a" * 64,
        engine_id="ray",
        binding_id="test.binding",
        target_kind="parquet",
        target_ref="unused",
        mode="append",
        committed=committed,
    )
    gateway = WriteGateway.__new__(WriteGateway)
    with (
        patch.object(gateway, "_execute", return_value=receipt),
        caplog.at_level(logging.INFO, logger="tributo.data.writing.gateway"),
    ):
        assert gateway.execute(object(), object()) is receipt
    events = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.write.finished"
    ]
    assert len(events) == 1
    assert events[0].status == expected
    assert events[0].committed is committed
    assert not hasattr(events[0], "rows_written")


def test_write_failure_without_receipt_is_logged_and_exception_is_preserved(caplog):
    gateway = WriteGateway.__new__(WriteGateway)
    original = RuntimeError("password=private")
    with (
        patch.object(gateway, "_execute", side_effect=original),
        caplog.at_level(logging.INFO, logger="tributo.data.writing.gateway"),
    ):
        with pytest.raises(RuntimeError) as error:
            gateway.execute(object(), object())
    assert error.value is original
    events = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.write.finished"
    ]
    assert len(events) == 1 and events[0].status == "failed"
    assert events[0].error_type == "RuntimeError"
    assert "private" not in caplog.text
    assert not hasattr(events[0], "committed")


def test_write_cancellation_is_not_an_error(caplog):
    gateway = WriteGateway.__new__(WriteGateway)
    original = CancelledError()
    with (
        patch.object(gateway, "_execute", side_effect=original),
        caplog.at_level(logging.INFO),
    ):
        with pytest.raises(CancelledError) as caught:
            gateway.execute(object(), object())
    assert caught.value is original
    event = next(
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.write.finished"
    )
    assert event.status == "cancelled" and event.levelno == logging.INFO


def test_export_partial_and_blocked_are_not_reported_as_success(tmp_path, caplog):
    config = BundleOutputConfig(
        bundle_uri=str(tmp_path / "bundle"),
        targets=[
            ExportTarget(name="primary", format="onnx"),
            ExportTarget(
                name="optional", format="onnx", exporter_id="failing-v1", required=False
            ),
            ExportTarget(
                name="dependent",
                format="onnx",
                depends_on=("optional",),
                required=False,
            ),
        ],
    )
    plan, exports, validators = _make_plan(config)
    with caplog.at_level(logging.DEBUG, logger="tributo.exporting.executor"):
        result = ExportManager(exports, validators).execute(
            plan, _make_source(), tmp_path / "staging", "execution-1"
        )
    assert result.status == "partial"
    events = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.stage.finished"
    ]
    assert next(r for r in events if r.phase == "export").status == "partial"
    assert (
        next(r for r in events if getattr(r, "node_id", None) == "dependent").status
        == "blocked"
    )
    assert not any(
        getattr(r, "node_id", None) == "dependent"
        and getattr(r, "event", None) == "tributo.stage.started"
        for r in caplog.records
    )
    node_events = [record for record in events if record.phase == "export_node"]
    assert len({record.phase_index for record in node_events}) == 3
    assert all(not hasattr(record, "duration_ms") for record in node_events)


def test_inference_failure_has_safe_type_and_existing_identity(caplog):
    from tributo.inference.executor import _failed_result, prepared_inference_plan

    with caplog.at_level(logging.ERROR, logger="tributo.inference.executor"):
        result = _failed_result(
            prepared_inference_plan(_plan()),
            phase="execution",
            exc=ValueError("private"),
        )
    assert result.status == "failed"
    event = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tributo.stage.finished"
    )
    assert event.levelno == logging.ERROR
    assert event.run_id == "run-1" and event.error_type == "ValueError"
    assert "private" not in caplog.text


@pytest.mark.parametrize("valid_receipt", [False, True])
def test_sink_events_bind_plan_identity_and_preserve_cleanup(
    caplog, monkeypatch, valid_receipt
):
    from unittest.mock import MagicMock

    from tributo.inference.executor import (
        RayMapBatchesExecutor,
        prepared_inference_plan,
    )

    plan = prepared_inference_plan(_plan())
    monkeypatch.setenv("TRIBUTO_RUN_ID", "parent-run")
    monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", "parent-attempt")
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", "parent-submission")
    from tributo.inference.contracts import ResultSinkReceipt

    receipt = (
        ResultSinkReceipt(
            sink_id=plan.output_port_id,
            result_id="a" * 64,
            uri="/output",
            rows_written=1,
        )
        if valid_receipt
        else None
    )
    opened = SimpleNamespace(
        dataset=SimpleNamespace(map_batches=MagicMock(return_value=object())),
        receipt=_receipt() if valid_receipt else None,
        close=MagicMock(),
        cancel=MagicMock(),
    )
    sink = SimpleNamespace(
        sink_id=plan.output_port_id, write=MagicMock(return_value=receipt)
    )
    executor = RayMapBatchesExecutor(
        kernel_factory=SimpleNamespace(factory_id="test.factory")
    )
    with caplog.at_level(logging.DEBUG, logger="tributo.inference.executor"):
        if valid_receipt:
            assert executor.execute_prepared(plan, sink, opened).status == "succeeded"
        else:
            with pytest.raises(AttributeError):
                executor.execute_prepared(plan, sink, opened)
    opened.close.assert_called_once()
    opened.cancel.assert_not_called()
    writes = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tributo.write.finished"
    ]
    assert len(writes) == 1 and writes[0].status == (
        "succeeded" if valid_receipt else "failed"
    )
    events = [r for r in caplog.records if getattr(r, "event", None)]
    assert events and all(
        r.run_id == plan.run_id
        and r.attempt_id == plan.attempt_id
        and r.submission_id == plan.submission_id
        for r in events
    )


def test_kernel_configures_before_create_and_emits_model_ready_once(
    monkeypatch, caplog
):
    plan = _plan()
    calls = []

    class Kernel:
        def predict(self, inputs):
            return {"score": inputs["x"]}

        def close(self):
            calls.append("close")

    class Factory:
        factory_id = "test.logging"

        def create(self):
            assert logging.getLogger("tributo").level == logging.DEBUG
            calls.append("create")
            return Kernel()

    policy = LoggingPolicy(level="DEBUG", batches=True)
    monkeypatch.setenv(POLICY_ENV, policy.serialize())
    with caplog.at_level(logging.DEBUG, logger="tributo.inference.kernel"):
        predictor = KernelBatchPredictor(
            Factory(),
            plan.input_binding,
            plan.output_binding,
            _logging_policy=policy.serialize(),
        )
        for _ in range(3):
            predictor({"feature": np.array([1.0])})
        predictor.close()
        predictor.close()
    assert calls == ["create", "close"]
    assert (
        sum(getattr(r, "event", None) == "tributo.model.ready" for r in caplog.records)
        == 1
    )
    progress = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.batch.progress"
    ]
    assert len(progress) == 1 and progress[0].batch_index == 1


def test_kernel_loading_failure_has_no_ready_event(monkeypatch, caplog):
    plan = _plan()

    class Factory:
        factory_id = "test.failure"

        def create(self):
            raise RuntimeError("load failed")

    monkeypatch.setenv(POLICY_ENV, LoggingPolicy().serialize())
    with caplog.at_level(logging.DEBUG, logger="tributo.inference.kernel"):
        with pytest.raises(RuntimeError, match="load failed"):
            KernelBatchPredictor(Factory(), plan.input_binding, plan.output_binding)
    assert not any(
        getattr(r, "event", None) == "tributo.model.ready" for r in caplog.records
    )


class _Stream(StreamingInferenceService):
    async def _generate_stream(self, data):
        if data.get("fail_before"):
            raise RuntimeError("private")
        if data.get("cancel"):
            raise asyncio.CancelledError()
        yield "first"
        if data.get("fail_after"):
            raise RuntimeError("private")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data,status",
    [
        ({}, "succeeded"),
        ({"fail_before": True}, "failed"),
        ({"fail_after": True}, "failed"),
        ({"cancel": True}, "cancelled"),
    ],
)
async def test_sse_terminal_follows_iteration_and_never_repeats(data, status, caplog):
    request = SimpleNamespace(
        headers={"x-request-id": "request-1"}, json=AsyncMock(return_value=data)
    )
    with caplog.at_level(logging.DEBUG, logger="tributo.serving.streaming_deployment"):
        response = await _Stream()(request)
        assert not any(
            getattr(r, "event", None) == "tributo.serving.stream.finished"
            for r in caplog.records
        )
        try:
            async for _ in response.body_iterator:
                pass
        except asyncio.CancelledError:
            assert data.get("cancel")
    events = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.finished"
    ]
    assert len(events) == 1 and events[0].status == status
    assert events[0].request_id == "request-1"
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_sse_explicit_close_and_input_rejection_have_unique_terminal(caplog):
    with caplog.at_level(logging.DEBUG, logger="tributo.serving.streaming_deployment"):
        stream = _Stream()
        response = await stream(
            SimpleNamespace(
                headers={"x-request-id": "closed"}, json=AsyncMock(return_value={})
            )
        )
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        await response.body_iterator.aclose()
        rejected = await stream(
            SimpleNamespace(
                headers={"x-request-id": "rejected"},
                json=AsyncMock(side_effect=ValueError("invalid")),
            )
        )
        assert rejected.status_code == 400
    events = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.finished"
    ]
    assert {(r.request_id, r.status) for r in events} == {
        ("closed", "cancelled"),
        ("rejected", "invalid_argument"),
    }
    assert len(events) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cancel", "disconnect"])
async def test_sse_input_read_cancellation_preserves_original(kind, caplog):
    from starlette.requests import ClientDisconnect

    original = asyncio.CancelledError() if kind == "cancel" else ClientDisconnect()
    request = SimpleNamespace(headers={}, json=AsyncMock(side_effect=original))
    with caplog.at_level(logging.DEBUG), pytest.raises(type(original)) as caught:
        await _Stream()(request)
    assert caught.value is original
    terminals = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.finished"
    ]
    assert len(terminals) == 1 and terminals[0].status == "cancelled"


@pytest.mark.asyncio
async def test_sse_close_before_first_advance_has_one_terminal(caplog):
    with caplog.at_level(logging.DEBUG):
        response = await _Stream()(
            SimpleNamespace(headers={}, json=AsyncMock(return_value={}))
        )
        await response.body_iterator.aclose()
        await response.body_iterator.aclose()
    terminals = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.finished"
    ]
    assert len(terminals) == 1 and terminals[0].status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.0", "2.4"])
async def test_sse_transport_disconnect_while_suspended_in_send(version, caplog):
    from starlette.requests import ClientDisconnect

    sent = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            sent.set()
            if version == "2.4":
                raise OSError("disconnected")
            await asyncio.Event().wait()

    async def receive():
        await sent.wait()
        return {"type": "http.disconnect"}

    with caplog.at_level(logging.DEBUG):
        response = await _Stream()(
            SimpleNamespace(headers={}, json=AsyncMock(return_value={}))
        )
        if version == "2.4":
            with pytest.raises(ClientDisconnect):
                await response(
                    {"type": "http", "asgi": {"spec_version": version}}, receive, send
                )
        else:
            await asyncio.wait_for(
                response(
                    {"type": "http", "asgi": {"spec_version": version}}, receive, send
                ),
                timeout=2,
            )
        await response.body_iterator.aclose()
    terminals = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.finished"
    ]
    assert len(terminals) == 1 and terminals[0].status == "cancelled"


@pytest.mark.asyncio
async def test_concurrent_sse_requests_keep_context_and_terminal_isolated(caplog):
    service = _Stream()

    async def consume(identity, data):
        response = await service(
            SimpleNamespace(
                headers={"x-request-id": identity}, json=AsyncMock(return_value=data)
            )
        )
        return [chunk async for chunk in response.body_iterator]

    with caplog.at_level(logging.DEBUG, logger="tributo.serving.streaming_deployment"):
        first, second = await asyncio.gather(
            consume("first", {"fail_after": True}), consume("second", {})
        )
    assert "data: [DONE]\n\n" == first[-1] == second[-1]
    terminals = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.finished"
    ]
    starts = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "tributo.serving.stream.started"
    ]
    assert {(r.request_id, r.status) for r in terminals} == {
        ("first", "failed"),
        ("second", "succeeded"),
    }
    assert len(terminals) == 2
    assert {r.request_id: r.operation_id for r in starts} == {
        r.request_id: r.operation_id for r in terminals
    }


@pytest.mark.parametrize("cancelled", [False, True])
def test_adapter_worker_observes_rank_failure_and_preserves_exception(
    caplog, cancelled
):
    import ray.train

    from tributo.integrations.algorithm_runtimes import ray_train_torch

    original = CancelledError() if cancelled else RuntimeError("private")
    context = SimpleNamespace(get_world_rank=lambda: 1, get_world_size=lambda: 2)
    with (
        patch.object(ray.train, "get_context", return_value=context),
        patch.object(
            ray_train_torch, "_adapter_worker", side_effect=original
        ) as worker,
        caplog.at_level(logging.DEBUG, logger=ray_train_torch.__name__),
        pytest.raises(type(original)) as caught,
    ):
        ray_train_torch.ray_torch_adapter_train_loop_per_worker(
            {"_core_logging_policy": LoggingPolicy().serialize()}
        )
    assert caught.value is original
    worker.assert_called_once()
    terminals = [
        r
        for r in caplog.records
        if getattr(r, "phase", None) == "training_worker"
        and getattr(r, "event", None) == "tributo.stage.finished"
    ]
    assert len(terminals) == 1
    assert terminals[0].rank == 1 and terminals[0].world_size == 2
    assert terminals[0].mode == "adapter"
    assert terminals[0].status == ("cancelled" if cancelled else "failed")
    assert "private" not in caplog.text
