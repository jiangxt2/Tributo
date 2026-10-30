"""No result mocks: real SDK, pre-provisioned Collector, Doris and Ray processes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import secrets
import socket
import time
from pathlib import Path
from typing import Any

import pymysql
import pytest

from tributo._common.reporting.collector import ManagedCollector
from tributo._common.reporting.session import active_session
from tributo.observability import (
    CollectorConfig,
    DorisProfile,
    ReportingConfig,
    ReportingError,
    SenderConfig,
    reporting_session,
)

pytestmark = pytest.mark.integration
_WORK = Path("/workspace/tributo-work")
_ARTIFACTS = _WORK / "artifacts"
_DATABASE = "tributo_reporting"


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        return int(listener.getsockname()[1])


def _sql(statement: str, parameters: tuple[Any, ...] | None = None) -> list[Any]:
    with pymysql.connect(
        host="fe",
        port=9030,
        user="root",
        connect_timeout=3,
        read_timeout=5,
        autocommit=True,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(statement, parameters)
            return list(cursor.fetchall())


@pytest.fixture(scope="module", autouse=True)
def infrastructure() -> None:
    assert os.environ.get("TRIBUTO_DOCKER_REPORTING_TEST") == "1", (
        "Use python -m tools.reporting_it"
    )
    assert importlib.metadata.version("ray") == "2.59.0"
    assert importlib.metadata.version("opentelemetry-sdk") == "1.45.1"
    binary = Path("/opt/reporting/otelcol-contrib")
    with binary.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == os.environ["REPORTING_COLLECTOR_SHA"]
        )
    deadline = time.monotonic() + 180
    while True:
        try:
            with pymysql.connect(
                host="fe", port=9030, user="root", connect_timeout=3
            ) as connection:
                with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                    cursor.execute("SHOW BACKENDS")
                    assert any(
                        str(row["Alive"]).lower() == "true" for row in cursor.fetchall()
                    )
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    _sql(f"CREATE DATABASE {_DATABASE}")
    _sql(f"""CREATE TABLE {_DATABASE}.otel_logs (
        timestamp DATETIME(6), service_name VARCHAR(200), service_instance_id VARCHAR(200),
        trace_id VARCHAR(200), span_id STRING, severity_number INT, severity_text STRING,
        body STRING, resource_attributes VARIANT, log_attributes VARIANT,
        scope_name STRING, scope_version STRING
    ) DUPLICATE KEY(timestamp, service_name) DISTRIBUTED BY HASH(service_name) BUCKETS 1
    PROPERTIES ('replication_num'='1')""")
    for role in ("reader", "writer"):
        password = secrets.token_hex(24)
        # Test credentials are random, scoped to this disposable database and
        # assigned in-process, never command-line or runtime_env payload values.
        os.environ["REPORTING_IT_SECRET_" + role.upper() + "_USER"] = (
            "reporting_" + role
        )
        os.environ["REPORTING_IT_SECRET_" + role.upper() + "_PASSWORD"] = password
        _sql("CREATE USER %s@'%%' IDENTIFIED BY %s", ("reporting_" + role, password))
        privileges = "SELECT_PRIV" if role == "reader" else "SELECT_PRIV, LOAD_PRIV"
        _sql(f"GRANT {privileges} ON {_DATABASE}.* TO 'reporting_{role}'@'%'")
    # Prove read-only permission separately from mere SELECT success.
    with pymysql.connect(
        host="fe",
        port=9030,
        user=os.environ["REPORTING_IT_SECRET_READER_USER"],
        password=os.environ["REPORTING_IT_SECRET_READER_PASSWORD"],
    ) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(pymysql.MySQLError):
                cursor.execute(f"DROP TABLE {_DATABASE}.otel_logs")
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (_ARTIFACTS / "versions.json").write_text(
        json.dumps(
            {
                "ray": importlib.metadata.version("ray"),
                "otel_sdk": importlib.metadata.version("opentelemetry-sdk"),
                "collector_sha": os.environ["REPORTING_COLLECTOR_SHA"],
                "source_snapshot": Path("/workspace/tributo-src/.tributo-source-ready")
                .read_text()
                .strip(),
            }
        )
    )


def _config(*, advertise: str = "driver", bind: str = "0.0.0.0") -> ReportingConfig:
    port, health = _port(), _port()
    profiles = {
        role: DorisProfile(
            host="fe",
            database=_DATABASE,
            username_ref=f"env://REPORTING_IT_SECRET_{role.upper()}_USER",
            password_ref=f"env://REPORTING_IT_SECRET_{role.upper()}_PASSWORD",
            http_endpoint="http://fe:8030" if role == "writer" else None,
        )
        for role in ("reader", "writer")
    }
    return ReportingConfig(
        enabled=True,
        sender=SenderConfig(
            endpoint=f"http://{advertise}:{port}/v1/logs",
            trusted_network=True,
            namespaces=("tributo", "ray", "ray_doris", "ray_clickhouse"),
            interval_seconds=0.1,
            batch_size=8,
            send_timeout=3.0,
            shutdown_timeout=8.0,
        ),
        startup_timeout=20.0,
        collector=CollectorConfig(
            binary="/opt/reporting/otelcol-contrib",
            sha256=os.environ["REPORTING_COLLECTOR_SHA"],
            otlp_bind=f"{bind}:{port}",
            health_bind=f"127.0.0.1:{health}",
        ),
        read_profile=profiles["reader"],
        write_profile=profiles["writer"],
    )


def _evidence(mode: str, **values: Any) -> None:
    with (_ARTIFACTS / "evidence.jsonl").open("a") as stream:
        stream.write(json.dumps({"mode": mode, **values}) + "\n")


def _wait_rows(event: str, minimum: int = 1) -> list[Any]:
    deadline = time.monotonic() + 20
    while True:
        rows = _sql(
            f"SELECT body, CAST(log_attributes AS STRING) FROM {_DATABASE}.otel_logs WHERE body = %s",
            (event,),
        )
        if len(rows) >= minimum:
            return rows
        assert time.monotonic() < deadline, f"Doris event missing: {event}"
        time.sleep(0.1)


def test_no_ray_managed_probe_safe_records_and_drain() -> None:
    import ray

    assert not ray.is_initialized()
    config = _config()
    start = time.monotonic()
    with reporting_session(config) as owner:
        session = active_session()
        assert session and session._collector and session._collector.process
        process = session._collector.process
        startup = time.monotonic() - start
        with reporting_session(config) as nested:
            assert nested is owner
        sent = time.monotonic()
        for namespace in ("tributo", "ray", "ray_doris", "ray_clickhouse"):
            logger = logging.getLogger(namespace)
            logger.warning(
                "raw-secret SQL SELECT private",
                extra={"event": "tributo.it.no_ray", "run_id": "no-ray"},
            )
        assert session._sender and session._sender.flush(5.0)
        rows = _wait_rows("tributo.it.no_ray", 4)
        assert len(rows) == 4
        assert all(
            "raw-secret" not in str(row) and "SELECT private" not in str(row)
            for row in rows
        )
        status = Path(f"/proc/{process.pid}/status").read_text()
        cpu = Path(f"/proc/{process.pid}/stat").read_text().split()
        _evidence(
            "no-ray",
            startup_seconds=startup,
            collector_status=status,
            snapshot=owner.snapshot(),
            rows=rows,
            queryable_latency_seconds=time.monotonic() - sent,
            cpu_ticks=int(cpu[13]) + int(cpu[14]),
        )
        draining = time.monotonic()
    assert process.poll() is not None
    assert owner.state == "closed"
    assert not ray.is_initialized()
    _evidence(
        "no-ray-drain",
        elapsed=time.monotonic() - draining,
        collector_reaped=process.returncode is not None,
    )


def test_external_owner_survives_session_close() -> None:
    config = _config()
    diagnostics: list[str] = []
    collector = ManagedCollector(config, diagnostics.append)
    collector.start(time.monotonic() + 20)
    try:
        with reporting_session(config.model_copy(update={"mode": "external"})):
            logger = logging.getLogger("tributo")
            logger.warning("ignored text", extra={"event": "tributo.it.external"})
        assert collector.process and collector.process.poll() is None
        rows = _wait_rows("tributo.it.external")
        _evidence("external", rows=rows, external_alive=True)
    finally:
        assert collector.close()


def test_bad_readback_fails_before_business_and_cleans_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config().model_copy(update={"startup_timeout": 4.0})
    monkeypatch.setenv("REPORTING_IT_SECRET_READER_PASSWORD", "invalid-test-password")
    entered = False
    start = time.monotonic()
    with pytest.raises(ReportingError, match="reporting.probe.not_queryable"):
        with reporting_session(config):
            entered = True
    assert not entered and active_session() is None
    assert time.monotonic() - start < 15
    _evidence("bad-readback", business_entered=False, elapsed=time.monotonic() - start)


def test_runtime_collector_loss_preserves_business_result() -> None:
    config = _config()
    with reporting_session(config) as owner:
        session = active_session()
        assert session and session._collector and session._collector.process
        session._collector.process.terminate()
        session._collector.process.wait(timeout=10)
        logging.getLogger("tributo").warning(
            "ignored", extra={"event": "tributo.it.network_loss"}
        )
        assert session._sender and session._sender.flush(5.0)
        assert owner.state == "degraded"
        result = 42
    assert result == 42
    _evidence("runtime-loss", result=result, snapshot=owner.snapshot())


def test_local_parent_child_owned_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    from tributo.runtime_providers import run_local_entrypoint

    config = _config()
    # The parent is the sole owner. The child only needs the frozen producer policy.
    for name in ("RUN_ID", "ATTEMPT_ID", "SUBMISSION_ID"):
        monkeypatch.setenv("TRIBUTO_" + name, "parent-" + name.lower())
    config_path = _WORK / "local-config.json"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json", exclude={"enabled"}))
    )
    # This child explicitly opens reporting_session() to consume the inherited
    # producer policy instead of loading another owner configuration.
    run_local_entrypoint(
        "python -m tests.integrations.reporting_child",
        env_vars={
            "TRIBUTO_LOG_REPORTING_ENABLED": "1",
            "TRIBUTO_LOG_REPORTING_CONFIG": str(config_path),
            "APP": "keep",
            "TRIBUTO_ATTEMPT_ID": "local-attempt",
        },
        num_cpus=2,
        timeout=90,
    )
    rows = [
        row
        for event in ("child", "worker")
        for row in _wait_rows("tributo.it.local." + event)
    ]
    expected = {
        "run_id": "parent-run_id",
        "attempt_id": "local-attempt",
        "submission_id": "parent-submission_id",
    }
    assert all(expected.items() <= json.loads(row[1]).items() for row in rows)
    _evidence(
        "local-parent-child",
        result=json.loads((_ARTIFACTS / "local.json").read_text()),
        rows=rows,
    )


def _run_ray_matrix(config: ReportingConfig) -> None:
    import ray
    from ray import serve, train, tune
    from ray.train import BackendConfig, ScalingConfig
    from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from tests.integrations.reporting_workload import (
        Actor,
        emit,
        original_hook,
        task,
        train_loop,
        tune_trial,
    )
    from tributo._common.serve_utils import deploy_serve_app

    outcomes = {}
    for enabled in (False, True):
        started = time.monotonic()
        with reporting_session(
            config.model_copy(update={"enabled": enabled}), service_lifetime=True
        ) as owner:
            session = active_session()
            process = session._collector.process if session else None
            result = {
                "startup_seconds": time.monotonic() - started,
                "collector_processes": int(process is not None),
                "business_counts": [],
                "phases": {},
            }

            def measure(
                name, count, work, result=result, enabled=enabled, process=process
            ):
                began = time.monotonic()
                result["business_counts"].append(work())
                if enabled:
                    assert len(_wait_rows("tributo.it." + name, count)) == count
                    result[name + "_resources"] = _process_resources(process.pid)
                result["phases"][name + "_seconds"] = time.monotonic() - began

            if process:
                result["idle_resources"] = _process_resources(process.pid)
            ray.init(
                address="auto",
                runtime_env=owner.runtime_env(
                    {
                        "env_vars": {"REPORTING_IT_ORIGINAL": "preserved"},
                        "worker_process_setup_hook": original_hook,
                    }
                ),
            )
            try:
                deadline = time.monotonic() + 30
                while len(nodes := [n for n in ray.nodes() if n["Alive"]]) < 2:
                    assert time.monotonic() < deadline
                    time.sleep(0.5)
                assert len(nodes) == 2
                measure("local", 60, lambda: emit("tributo.it.local", 60)["count"])

                def workers(enabled=enabled):
                    results = ray.get(
                        [
                            ray.remote(task)
                            .options(
                                scheduling_strategy=NodeAffinitySchedulingStrategy(
                                    node["NodeID"], soft=False
                                )
                            )
                            .remote("tributo.it.task", 20)
                            for node in nodes
                        ]
                    )
                    if enabled:
                        assert all(r["role"] == "producer" for r in results)
                        assert all(r["original_hook"] == "1" for r in results)
                        assert all(not r["owner_credentials"] for r in results)
                        assert {r["node_id"] for r in results} == {
                            n["NodeID"] for n in nodes
                        }
                    return sum(r["count"] for r in results)

                measure("task", 40, workers)

                @serve.deployment(
                    ray_actor_options={
                        "num_cpus": 0.1,
                        "runtime_env": {"env_vars": {"SERVE_OPTION": "preserved"}},
                    }
                )
                class Reporter:
                    def __call__(self):
                        return emit("tributo.it.serve") | {
                            "option": os.environ.get("SERVE_OPTION")
                        }

                def requests(enabled=enabled):
                    deploy_serve_app(
                        Reporter,
                        app_name="reporting-it",
                        route_prefix="/reporting-it",
                        enable_http=False,
                    )
                    handle = serve.get_app_handle("reporting-it")
                    responses = [handle.remote().result(timeout_s=30) for _ in range(6)]
                    assert all(r["option"] == "preserved" for r in responses)
                    if enabled:
                        assert all(r["role"] == "producer" for r in responses)
                    return sum(r["count"] for r in responses)

                measure("serve", 6, requests)
                # Profile identical work before Train/Tune introduce extra load.
                if enabled:
                    actor = ray.remote(Actor).remote()
                    assert (
                        ray.get(actor.report.remote("tributo.it.actor"))["role"]
                        == "producer"
                    )
                    ray.kill(actor)
                    policy = session.producer_policy()
                    trained = DataParallelTrainer(
                        train_loop_per_worker=train_loop,
                        train_loop_config={"_core_reporting_policy": policy},
                        backend_config=BackendConfig(),
                        scaling_config=ScalingConfig(
                            num_workers=2, use_gpu=False, placement_strategy="SPREAD"
                        ),
                        run_config=train.RunConfig(storage_path=str(_WORK / "train")),
                    ).fit()
                    assert trained.error is None
                    rows = _wait_rows("tributo.it.train", 2)
                    assert {json.loads(row[1])["rank"] for row in rows} == {0, 1}
                    tuned = tune.Tuner(
                        tune.with_resources(tune_trial, {"cpu": 1}),
                        param_space={"policy": policy},
                        tune_config=tune.TuneConfig(num_samples=1),
                        run_config=tune.RunConfig(storage_path=str(_WORK / "tune")),
                    ).fit()
                    assert not tuned.errors and tuned[0].metrics["ok"] == 1
                    assert _wait_rows("tributo.it.actor") and _wait_rows(
                        "tributo.it.tune"
                    )
                    assert all(
                        "ray_job_id" in str(row)
                        for row in _wait_rows("tributo.it.task", 40)
                    )
                    _evidence(
                        "ray-multinode",
                        nodes=2,
                        train_rows=rows,
                        snapshot=owner.snapshot(),
                    )
                else:
                    serve.delete("reporting-it")
                closing = time.monotonic()
                if session:
                    session.close()
                result["reporting_close_seconds"] = time.monotonic() - closing
                result["collector_reaped"] = (
                    process is None or process.poll() is not None
                )
                assert result["collector_reaped"]
                assert "reporting-it" not in serve.status().applications
                outcomes["enabled" if enabled else "disabled"] = result
            finally:
                if session:
                    session.close()
                serve.shutdown()
                ray.shutdown()
    assert outcomes["disabled"]["collector_processes"] == 0
    assert outcomes["enabled"]["collector_processes"] == 1
    assert (
        outcomes["disabled"]["business_counts"]
        == outcomes["enabled"]["business_counts"]
        == [60, 40, 6]
    )
    (_ARTIFACTS / "resource-comparison.json").write_text(
        json.dumps(outcomes, indent=2) + "\n"
    )
    _evidence("resource-comparison", **outcomes)


def _remote_job(mode: str) -> dict[str, Any]:
    from ray.job_submission import JobStatus, JobSubmissionClient

    from tributo.ray_jobs import submit_ray_job

    config = _config(advertise="ray-head")
    path = _WORK / (mode + "-owner.json")
    path.write_text(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "credentials": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith("REPORTING_IT_SECRET_")
                },
            }
        )
    )
    path.chmod(0o600)
    # Submitting returns without waiting; remote driver owns its own Collector.
    submission = submit_ray_job(
        "python -m tests.integrations.reporting_workload " + mode + " " + str(path),
        operation_namespace="reporting",
        run_id="reporting-" + mode,
        dashboard_url="http://ray-head:8265",
    )
    client = JobSubmissionClient("http://ray-head:8265")
    deadline = time.monotonic() + 180
    while True:
        status = client.get_job_status(submission.submission_id)
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
            break
        assert time.monotonic() < deadline
        time.sleep(0.5)
    (_ARTIFACTS / (mode + "-ray-job.log")).write_text(
        client.get_job_logs(submission.submission_id)
    )
    assert status == JobStatus.SUCCEEDED
    if mode in {"matrix", "light", "regressions"}:
        return {"status": str(status), "submission_id": submission.submission_id}
    rows = {kind: _wait_rows("tributo.it.job." + kind) for kind in ("driver", "worker")}
    result = json.loads((_ARTIFACTS / "job.json").read_text())
    assert result["role"] == "owner" and result["worker"]["role"] == "producer"
    assert not result["worker"]["owner_credentials"]
    _evidence("tributo-job", status=str(status), result=result, rows=rows)
    return result


def test_multinode_task_actor_train_tune_and_serve() -> None:
    # Distributed batch execution belongs to a Ray Jobs execution domain.
    _remote_job("matrix")


def test_tributo_job_remote_owner_after_submitter_exit() -> None:
    _remote_job("job")


def _run_review_regressions(config: ReportingConfig) -> None:
    from dataclasses import replace

    import ray

    from tests.algorithms.conftest import (
        dispatcher_for,
        function_registration,
        request_for,
    )
    from tests.integrations.reporting_workload import PolicyScopeActor, original_hook
    from tributo.algorithms.api import (
        AlgorithmOperation,
        QualifiedReference,
    )
    from tributo.algorithms.input import FakeInputInvocation, FakeTabularPayload
    from tributo.algorithms.spi import InputExecutionContext
    from tributo.integrations.algorithm_runtimes.ray_task import RayTaskRuntime

    # A genuine caller connection and hook exist before reporting is enabled.
    ray.init(
        address="auto",
        runtime_env={
            "worker_process_setup_hook": original_hook,
            "env_vars": {"REPORTING_CALLER_ENV": "keep"},
        },
    )
    caller_job = ray.get_runtime_context().get_job_id()
    try:
        with reporting_session(config):
            session = active_session()
            assert session and session._collector and session._collector.process
            collector = session._collector.process
            registration = function_registration(data_parallel=True)
            registration = replace(
                registration,
                implementation=replace(
                    registration.implementation,
                    implementation_ref=QualifiedReference.parse(
                        "tests.integrations.reporting_workload:core_reporting_fragment"
                    ),
                ),
                runtime=replace(
                    registration.runtime,
                    result_reducer_ref=QualifiedReference.parse(
                        "tests.integrations.reporting_workload:core_reporting_reducer"
                    ),
                ),
            )
            dispatcher = dispatcher_for(registration, RayTaskRuntime())
            invocation = InputExecutionContext(
                {
                    "binary-fixture": FakeInputInvocation(
                        FakeTabularPayload(
                            {
                                "x0": (0, 1, 2, 3),
                                "x1": (4, 5, 6, 7),
                                "label": (0, 0, 1, 1),
                            }
                        )
                    )
                }
            )
            outcome = dispatcher.execute(
                request_for("external_function", AlgorithmOperation.FIT), invocation
            )
            assert (
                outcome.execution.status == "succeeded"
                and outcome.execution.metrics["row_count"] == 4
            )
            assert (
                ray.is_initialized()
                and ray.get_runtime_context().get_job_id() == caller_job
            )
            assert session._collector.process is collector and collector.poll() is None
            worker_rows = _wait_rows("tributo.fix3.core.worker", 2)
            reducer_rows = _wait_rows("tributo.fix3.core.reducer")
            assert len(worker_rows) == 2 and len(reducer_rows) == 1

            first = json.loads(session.producer_policy())
            first["context"] = {
                "operation_id": "op-kernel-a",
                "run_id": "run-kernel-a",
                "attempt_id": "attempt-a",
            }
            second = {
                **first,
                "context": {"operation_id": "op-kernel-b", "run_id": "run-kernel-b"},
            }
            actor = (
                ray.remote(PolicyScopeActor)
                .options(num_cpus=0.1)
                .remote(json.dumps(first), json.dumps(second))
            )
            try:
                assert ray.get(actor.run.remote()) == [[0], [0], [0]]
            finally:
                ray.kill(actor)
            rows = {
                kind: _wait_rows("tributo.fix3.kernel." + kind, count)
                for kind, count in (("load", 2), ("predict", 3), ("close", 2))
            }
            predictions = [json.loads(row[1]) for row in rows["predict"]]
            assert [entry["run_id"] for entry in predictions].count("run-kernel-b") == 2
            assert [entry["run_id"] for entry in predictions].count("run-kernel-a") == 1
            for records in rows.values():
                for _, raw in records:
                    entry = json.loads(raw)
                    if entry["run_id"] == "run-kernel-b":
                        assert (
                            entry["operation_id"] == "op-kernel-b"
                            and "attempt_id" not in entry
                        )
            _evidence(
                "review-regressions",
                caller_job=caller_job,
                worker_rows=worker_rows,
                reducer_rows=reducer_rows,
                kernels=rows,
            )
        assert collector.poll() is not None and ray.is_initialized()
    finally:
        ray.shutdown()


def test_reporting_review_regressions() -> None:
    _remote_job("regressions")


def _process_resources(pid: int) -> dict[str, float]:
    import re

    status = Path(f"/proc/{pid}/status").read_text()
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return {
        key: float(re.search(rf"{field}:\s+(\d+)", status).group(1))
        for key, field in (("rss_kib", "VmRSS"), ("peak_rss_kib", "VmHWM"))
    } | {"cpu_seconds": (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")}
