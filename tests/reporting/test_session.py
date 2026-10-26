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
import requests
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


def test_bounded_flush_wait_does_not_hold_business_thread(
    config: ReportingConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    with reporting_session(config):
        session = active_session()
        assert session and session._sender
        monkeypatch.setattr(
            session._sender._provider, "force_flush", lambda: release.wait(5)
        )
        began = time.monotonic()
        try:
            assert session._sender.flush(0.03) is False
            assert time.monotonic() - began < 0.5
        finally:
            release.set()


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


def test_parallel_compatible_sessions_share_one_sender(config: ReportingConfig) -> None:
    entered = threading.Barrier(2)
    leaving = threading.Barrier(2)
    handles = []
    errors = []

    def work() -> None:
        try:
            with reporting_session(config) as owner:
                handles.append(owner)
                entered.wait(timeout=5)
                leaving.wait(timeout=5)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert not errors and len(handles) == 2
    assert handles[0] is handles[1] and handles[0].state == "closed"
    assert active_session() is None


@pytest.fixture
def otlp_stub(monkeypatch: pytest.MonkeyPatch) -> Endpoint:
    """Ownership unit tests stub I/O; real HTTP semantics belong to test_transport."""
    endpoint = Endpoint(url="http://127.0.0.1:1/v1/logs")

    def request(
        session: requests.Session, method: str, url: str, **kwargs: object
    ) -> requests.Response:
        data = kwargs.get("data")
        assert isinstance(data, bytes)
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        endpoint.requests.append((dict(headers), data))
        status, content_type, body = (
            endpoint.responses.pop(0)
            if endpoint.responses
            else (200, "application/x-protobuf", b"")
        )
        response = requests.Response()
        response.status_code = status
        response.headers["Content-Type"] = content_type
        response.raw = io.BytesIO(body)
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    return endpoint


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


def test_reused_sender_uses_scoped_policy_and_restores_missing_ids(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", "untrusted-node-id")
    a, b = _policy(config.sender, "a", attempt=True), _policy(config.sender, "b")
    owner = sessions.bootstrap_producer(a)
    assert owner and owner._sender
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    try:
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
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_reused_sender_isolates_concurrent_policy_contexts(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    a, b = _policy(config.sender, "a"), _policy(config.sender, "b")
    owner = sessions.bootstrap_producer(a)
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)

    async def work(policy: str, name: str) -> None:
        with sessions.producer_scope(policy):
            for _ in range(3):
                await asyncio.sleep(0)
                logger.info("hidden", extra={"event": "tributo.scope." + name})

    try:
        await asyncio.gather(work(a, "a"), work(b, "b"))
        assert owner and owner._sender and owner._sender.flush(2.0)
        assert CONTEXT.get() is None
        rows = _wire_attributes(otlp_stub)
        assert len(rows) == 6
        assert all(row["run_id"] == "run-" + str(row["event"])[-1] for row in rows)
    finally:
        assert owner
        owner.close()


def test_kernel_load_predict_close_bind_explicit_policy(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from tests.inference.test_bundle_predictor import _bindings, _Runtime
    from tributo.inference.kernel import KernelBatchPredictor

    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    a, b = _policy(config.sender, "a", attempt=True), _policy(config.sender, "b")
    owner = sessions.bootstrap_producer(a)
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)

    class Kernel(_Runtime):
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            logger.info("hidden", extra={"event": "tributo.scope.predict"})
            return super().predict(inputs)

        def close(self) -> None:
            logger.info("hidden", extra={"event": "tributo.scope.close"})
            super().close()

    class Factory:
        factory_id = "test.scope"

        def create(self) -> Kernel:
            logger.info("hidden", extra={"event": "tributo.scope.load"})
            return Kernel()

    try:
        kernel = KernelBatchPredictor(Factory(), *_bindings(), _reporting_policy=b)
        output = kernel(
            {
                "feature_a": np.array([1.0]),
                "feature_b": np.array([2.0]),
                "entity_id": np.array([7]),
            }
        )
        assert output["prediction"].tolist() == [0]
        kernel.close()
        assert owner and owner._sender and owner._sender.flush(2.0)
        rows = _wire_attributes(otlp_stub)
        assert len(rows) == 3 and all(row["run_id"] == "run-b" for row in rows)
        assert all("attempt_id" not in row for row in rows)
        assert CONTEXT.get() is None
        late_kernel = KernelBatchPredictor(Factory(), *_bindings(), _reporting_policy=b)
        owner.close()
        monkeypatch.setattr(sessions, "_ACTIVE", None)
        # Cleanup after the domain closes never initializes another sender.
        late_kernel.close()
        assert late_kernel._kernel.close_calls == 1
        assert sessions.active_session() is None
    finally:
        assert owner
        owner.close()


def test_train_wrapper_reuses_sender_with_invocation_identity(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import ray
    from ray import train

    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    a, b = _policy(config.sender, "a"), _policy(config.sender, "b")
    owner = sessions.bootstrap_producer(a)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        train,
        "get_context",
        lambda: SimpleNamespace(get_world_rank=lambda: 1, get_world_size=lambda: 2),
    )
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)

    @sessions.reporting_worker
    def work(values: object) -> None:
        logger.info("hidden", extra={"event": "tributo.scope.train"})

    try:
        work({"_core_reporting_policy": b})
        assert owner and owner._sender and owner._sender.flush(2.0)
        row = _wire_attributes(otlp_stub)[0]
        assert row["run_id"] == "run-b" and row["rank"] == 1 and row["world_size"] == 2
        assert CONTEXT.get() is None
    finally:
        assert owner
        owner.close()


def test_tune_trial_binds_captured_policy_and_restores(
    config: ReportingConfig,
    otlp_stub: Endpoint,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from ray import tune

    from tests.training.test_tune_runner import MockTrainer
    from tests.training.test_tune_runner import trainer_spec as make_trainer_spec
    from tributo.training import tune_runner as module
    from tributo.training.tune_config import TuneSearchConfig
    from tributo.training.tune_space import SearchSpaceSpec

    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setattr(sessions, "_REFERENCES", 0)
    a, b = _policy(config.sender, "a"), _policy(config.sender, "b")
    owner = sessions.bootstrap_producer(a)
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)

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
    try:
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
    finally:
        assert owner
        owner.close()


def test_nested_owner_preserves_global_provider_and_caller_handlers(
    config: ReportingConfig, otlp_stub: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("tributo_test")
    monkeypatch.setattr(logger, "level", logging.INFO)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    global_provider = get_logger_provider()
    try:
        with reporting_session(config) as owner:
            with reporting_session(config) as nested:
                assert nested is owner
                with context_scope(run_id="run-one"):
                    logger.info(
                        "raw SQL and secret must not export",
                        extra={
                            "password": "sensitive",
                            "event": "tributo.stage.finished",
                            "status": "succeeded",
                        },
                    )
                assert active_session() is owner
            assert owner.state == "ready"
        assert owner.state == "closed" and active_session() is None
        assert handler in logger.handlers
        assert "raw SQL" in output.getvalue()
        assert get_logger_provider() is global_provider
        wire = " ".join(str(r) for r in _records(otlp_stub))
        assert "sensitive" not in wire and "raw SQL" not in wire
        assert "run-one" in wire and "tributo.stage.finished" in wire
        assert "committed" not in owner.snapshot()
    finally:
        logger.removeHandler(handler)


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


def test_conflicting_nested_configuration_rejected(config: ReportingConfig) -> None:
    changed = config.model_copy(
        update={"sender": config.sender.model_copy(update={"level": "DEBUG"})}
    )
    with reporting_session(config):
        with pytest.raises(ReportingError, match="session.conflict"):
            with reporting_session(changed):
                pass


def test_disabled_nested_configuration_rejects_active_domain(
    config: ReportingConfig,
) -> None:
    with reporting_session(config) as owner:
        with pytest.raises(ReportingError, match="reporting.session.conflict"):
            with reporting_session(ReportingConfig()):
                raise AssertionError(
                    "must not present an inactive handle for the active domain"
                )
        assert active_session() is owner and owner.state == "ready"


def test_real_sdk_queue_overflow_is_visible(
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
    finally:
        release.set()
        assert sender.close()
