"""Session ownership tests; SQL readback is mocked only in this unit suite."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from opentelemetry._logs import get_logger_provider
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)

from tests.reporting.conftest import Endpoint
from tributo._common.reporting import session as sessions
from tributo._common.reporting.sender import CONTEXT, Sender, context_scope
from tributo._common.reporting.session import ReportingSession, active_session
from tributo.observability import (
    DorisProfile,
    ReportingConfig,
    ReportingError,
    SenderConfig,
    reporting_session,
)


def test_bridge_reentrant_library_logs_do_not_recurse(
    config: ReportingConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)

    def job_id() -> str:
        logger.info("library emits while identity is read")
        return "job-id"

    context = SimpleNamespace(
        get_job_id=job_id,
        get_worker_id=lambda: "worker-id",
        get_node_id=lambda: "node-id",
    )
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: True, get_runtime_context=lambda: context
        ),
    )
    with reporting_session(config):
        session = active_session()
        assert session and session._sender
        before = session._sender.offered
        logger.info("outer event")
        assert session._sender.offered == before + 1


@pytest.mark.parametrize("propagate", [True, False])
def test_explicit_descendant_routing_preserves_propagation_without_duplicates(
    config: ReportingConfig, monkeypatch: pytest.MonkeyPatch, propagate: bool
) -> None:
    logger = logging.getLogger("tributo_test.isolated")
    monkeypatch.setattr(logger, "level", logging.INFO)
    monkeypatch.setattr(logger, "propagate", propagate)
    changed = config.model_copy(
        update={
            "sender": config.sender.model_copy(
                update={"namespaces": ("tributo_test", "tributo_test.isolated")}
            )
        }
    )
    with reporting_session(changed):
        session = active_session()
        assert session and session._sender
        before = session._sender.offered
        logger.info("one record")
        assert session._sender.offered == before + 1
        assert logger.propagate is propagate


def test_producer_ack_is_not_owner_doris_checked(
    config: ReportingConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_readback(
        self: ReportingSession, probe: str, deadline: float
    ) -> None:
        raise AssertionError("producer must not perform owner SQL")

    monkeypatch.setattr(ReportingSession, "_readback", unexpected_readback)
    producer = ReportingSession(producer=config.sender)
    try:
        producer.start()
        assert producer.state == "ready"
        assert producer.snapshot()["diagnostics"]["producer.checked"] == 1
        assert "reporting.checked" not in producer.snapshot()["diagnostics"]
    finally:
        producer.close()


@pytest.fixture
def otlp_stub(otlp_endpoint: Endpoint) -> Endpoint:
    return otlp_endpoint


@pytest.fixture
def config(otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch) -> ReportingConfig:
    monkeypatch.setenv("TEST_READ_USER", "reader")
    monkeypatch.setenv("TEST_READ_PASSWORD", "unit-test-only")
    monkeypatch.setattr(
        ReportingSession, "_readback", lambda self, probe, deadline: None
    )
    return ReportingConfig(
        enabled=True,
        mode="external",
        sender=SenderConfig(
            endpoint=otlp_stub.url,
            namespaces=("tributo_test",),
            shutdown_timeout=2.0,
        ),
        read_profile=DorisProfile(
            host="unreachable.invalid",
            database="logs",
            username_ref="env://TEST_READ_USER",
            password_ref="env://TEST_READ_PASSWORD",
        ),
    )


def _records(endpoint: Endpoint) -> list[object]:
    result = []
    for _, body in endpoint.requests:
        request = ExportLogsServiceRequest()
        request.ParseFromString(body)
        result.extend(
            record
            for resource in request.resource_logs
            for scope in resource.scope_logs
            for record in scope.log_records
        )
    return result


def _policy(config: SenderConfig, name: str, *, attempt: bool = False) -> str:
    context = {"operation_id": "op-" + name, "run_id": "run-" + name}
    if attempt:
        context["attempt_id"] = "attempt-" + name
    return json.dumps(
        {
            "version": 1,
            "role": "producer",
            "sender": config.model_dump(mode="json"),
            "context": context,
        }
    )


def _wire_attributes(endpoint: Endpoint) -> list[dict[str, object]]:
    return [
        {
            entry.key: entry.value.string_value
            if entry.value.HasField("string_value")
            else entry.value.int_value
            for entry in record.attributes
        }
        for record in _records(endpoint)
        if record.body.string_value.startswith("tributo.scope.")
    ]


@pytest.fixture
def producer(config: ReportingConfig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", "untrusted-node-id")
    a, b = _policy(config.sender, "a", attempt=True), _policy(config.sender, "b")
    owner = sessions.bootstrap_producer(a)
    assert owner and owner._sender
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    try:
        yield owner, a, b, logger
    finally:
        owner.close()


def test_reused_sender_uses_scoped_policy_and_restores_missing_ids(
    producer, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, a, b, logger = producer
    with pytest.raises(ReportingError, match="context_requires_scope"):
        sessions.bootstrap_producer(b)
    with sessions.producer_scope(b) as borrowed:
        assert borrowed is owner
        with sessions.session_scope():
            logger.info("hidden", extra={"event": "tributo.scope.b"})
            child_policy = json.loads(sessions.current_policy())
            assert child_policy["context"] == {
                "operation_id": "op-b",
                "run_id": "run-b",
            }
    assert CONTEXT.get() is None
    with sessions.producer_scope(a):
        logger.info("hidden", extra={"event": "tributo.scope.a"})
    logger.info("hidden", extra={"event": "tributo.scope.default"})
    assert owner._sender.flush(2.0)
    rows = _wire_attributes(otlp_stub)
    assert [row["run_id"] for row in rows] == ["run-b", "run-a", "run-a"]
    assert "attempt_id" not in rows[0] and "submission_id" not in rows[0]
    assert rows[1]["attempt_id"] == "attempt-a"
    assert owner._context["run_id"] == "run-a"


@pytest.mark.asyncio
async def test_reused_sender_isolates_concurrent_policy_contexts(
    producer, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:

    owner, a, b, logger = producer

    async def work(policy: str, name: str) -> None:
        with sessions.producer_scope(policy):
            for _ in range(3):
                await asyncio.sleep(0)
                logger.info("hidden", extra={"event": "tributo.scope." + name})

    await asyncio.gather(work(a, "a"), work(b, "b"))
    assert owner and owner._sender and owner._sender.flush(2.0)
    assert CONTEXT.get() is None
    rows = _wire_attributes(otlp_stub)
    assert len(rows) == 6
    assert all(row["run_id"] == "run-" + str(row["event"])[-1] for row in rows)


def test_kernel_load_predict_close_bind_explicit_policy(
    producer, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, a, b, logger = producer
    from tests.reporting.conftest import reporting_kernel

    def emit(name):
        logger.info("hidden", extra={"event": "tributo.scope." + name})

    kernel, batch = reporting_kernel(b, emit)
    output = kernel(batch)
    assert output["prediction"].tolist() == [0]
    kernel.close()
    assert owner and owner._sender and owner._sender.flush(2.0)
    rows = _wire_attributes(otlp_stub)
    assert len(rows) == 3 and all(row["run_id"] == "run-b" for row in rows)
    assert all("attempt_id" not in row for row in rows)
    assert CONTEXT.get() is None
    late_kernel, _ = reporting_kernel(b, emit)
    owner.close()
    monkeypatch.setattr(sessions, "_ACTIVE", None)
    # Cleanup after the domain closes never initializes another sender.
    late_kernel.close()
    assert late_kernel._kernel.close_calls == 1
    assert sessions.active_session() is None


def test_train_wrapper_reuses_sender_with_invocation_identity(
    producer, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:

    owner, a, b, logger = producer
    import ray
    from ray import train

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        train,
        "get_context",
        lambda: SimpleNamespace(get_world_rank=lambda: 1, get_world_size=lambda: 2),
    )

    @sessions.reporting_worker
    def work(values: object) -> None:
        logger.info("hidden", extra={"event": "tributo.scope.train"})

    work({"_core_reporting_policy": b})
    assert owner and owner._sender and owner._sender.flush(2.0)
    row = _wire_attributes(otlp_stub)[0]
    assert row["run_id"] == "run-b" and row["rank"] == 1 and row["world_size"] == 2
    assert CONTEXT.get() is None


def test_tune_trial_binds_captured_policy_and_restores(
    producer,
    otlp_stub: Endpoint,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    owner, a, b, logger = producer
    from dataclasses import replace

    from ray import tune

    from tests.training.test_tune_runner import MockTrainer
    from tests.training.test_tune_runner import trainer_spec as make_trainer_spec
    from tributo.training import tune_runner as module
    from tributo.training.tune_config import TuneSearchConfig
    from tributo.training.tune_space import SearchSpaceSpec

    class Trainer(MockTrainer):
        def training_loop(self) -> object:
            logger.info("hidden", extra={"event": "tributo.scope.tune"})
            return super().training_loop()

    spec = replace(make_trainer_spec.__wrapped__(), trainer_cls=Trainer)
    monkeypatch.setattr(module, "_build_trial_run_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tune, "get_context", lambda: SimpleNamespace(get_trial_id=lambda: "trial-b")
    )
    reported = []
    monkeypatch.setattr(tune, "report", lambda metrics: reported.append(metrics))
    with sessions.producer_scope(b):
        runner = module.TuneRunner(
            spec,
            TuneSearchConfig(metric="loss", mode="min", num_samples=1),
            SearchSpaceSpec(parameters=()),
            {},
        )
        trial = runner._build_trainable({}, str(tmp_path))
    trial({})
    assert reported == [{"loss": 0.5}]
    assert owner and owner._sender and owner._sender.flush(2.0)
    row = _wire_attributes(otlp_stub)[0]
    assert row["run_id"] == "run-b" and row["trial_id"] == "trial-b"
    assert CONTEXT.get() is None


def test_runtime_failure_does_not_replace_business_exception(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    original = ValueError("business failure")
    with pytest.raises(ValueError) as caught:
        with reporting_session(config) as session:
            otlp_stub.responses.append((200, "text/html", b"invalid gateway"))
            logger.error("safe message")
            assert session._sender is not None and session._sender.flush()
            assert session.state == "degraded"
            raise original
    assert caught.value is original
    assert session.snapshot()["failed_batches"] == 1


def test_frozen_sender_ignores_later_ambient_changes(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    with reporting_session(config) as session:
        original_env = dict(os.environ)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Injected=sensitive")
        logger.warning("untrusted text")
    assert all("Injected" not in headers for headers, _ in otlp_stub.requests)
    assert session.state == "closed"
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == "Injected=sensitive"
    assert all(
        os.environ.get(k) == v
        for k, v in original_env.items()
        if k != "OTEL_EXPORTER_OTLP_HEADERS"
    )


def test_real_sdk_queue_overflow_and_flush_wait_are_bounded(
    otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opentelemetry.sdk._logs.export import LogRecordExportResult

    diagnostics = []
    sender = Sender(
        SenderConfig(
            endpoint=otlp_stub.url, queue_size=2, batch_size=1, shutdown_timeout=2.0
        ),
        diagnostics.append,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocking_export(batch: object) -> LogRecordExportResult:
        entered.set()
        release.wait(2)
        return LogRecordExportResult.SUCCESS

    monkeypatch.setattr(sender._exporter, "export", blocking_export)
    try:
        sender.emit("test", {})
        assert entered.wait(1)
        for _ in range(10):
            sender.emit("test", {})
        assert sender.overflow_events > 0
        assert "sender.queue_full" in diagnostics
        began = time.monotonic()
        assert sender.flush(0.03) is False
        assert time.monotonic() - began < 0.5
    finally:
        release.set()
        assert sender.close()


def test_logging_console_and_reporting_handler_coexist(
    config, otlp_stub, monkeypatch, isolated_logging
):
    from tributo._common.logging import configure_logging

    logger = logging.getLogger("tributo")
    monkeypatch.setattr(logger, "propagate", False)
    config = config.model_copy(
        update={"sender": config.sender.model_copy(update={"namespaces": ("tributo",)})}
    )
    output = io.StringIO()
    with reporting_session(config) as owner:
        configure_logging("DEBUG", stream=output)
        logger.info("message", extra={"event": "tributo.console.and.reporting"})
        assert owner._sender and owner._sender.flush(2)
        assert "tributo.console.and.reporting" in output.getvalue()
        assert (
            sum(
                r.body.string_value == "tributo.console.and.reporting"
                for r in _records(otlp_stub)
            )
            == 1
        )


def test_nested_parallel_owner_preserves_logging_and_global_provider(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    global_provider = get_logger_provider()
    entered = threading.Barrier(2)

    def work(_):
        with reporting_session(config) as owner:
            with reporting_session(config) as nested:
                assert nested is owner and active_session() is owner
                with context_scope(run_id="run-one"):
                    logger.info(
                        "raw SQL and secret must not export",
                        extra={
                            "password": "sensitive",
                            "event": "tributo.stage.finished",
                            "status": "succeeded",
                        },
                    )
                entered.wait(timeout=5)
            assert owner.state == "ready"
        return owner

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.map(work, range(2))
        assert first is second and first.state == "closed"
        assert active_session() is None and handler in logger.handlers
        assert "raw SQL" in output.getvalue()
        assert get_logger_provider() is global_provider
        wire = " ".join(str(r) for r in _records(otlp_stub))
        assert "sensitive" not in wire and "raw SQL" not in wire
        assert "run-one" in wire and "tributo.stage.finished" in wire
        assert "committed" not in first.snapshot()
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize("disabled", [False, True])
def test_incompatible_nested_config_preserves_active_owner(config, disabled):
    changed = (
        ReportingConfig()
        if disabled
        else config.model_copy(
            update={"sender": config.sender.model_copy(update={"level": "DEBUG"})}
        )
    )
    with reporting_session(config) as owner:
        with pytest.raises(ReportingError, match="reporting.session.conflict"):
            with reporting_session(changed):
                pass
        assert active_session() is owner and owner.state == "ready"
