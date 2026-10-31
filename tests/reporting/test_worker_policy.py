"""Frozen-role propagation and separate-process real SDK construction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tributo._common.reporting import session as sessions
from tributo._common.reporting.config import POLICY_ENV, WORKER_HOOK
from tributo.observability import (
    DorisProfile,
    ReportingConfig,
    ReportingError,
)


def test_producer_without_policy_never_inherits_owner_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sessions, "_ACTIVE", None)
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_ENABLED", "1")
    monkeypatch.setenv(
        "TRIBUTO_LOG_REPORTING_CONFIG", "/must-not-load-worker-owner-config.json"
    )

    @sessions.reporting_operation
    def business() -> int:
        assert sessions.active_session() is None
        return 42

    with sessions.producer_scope(ray_worker=True) as producer:
        assert producer is None and business() == 42
    with pytest.raises(ReportingError, match="reporting.config.invalid"):
        business()


@pytest.mark.parametrize("enabled", [False, True])
def test_core_task_and_reducer_options_preserve_caller_hook(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    from tributo.algorithms.api import (
        AlgorithmExecutionResult,
        RuntimeTopology,
        WorkerExecutionResult,
    )
    from tributo.integrations.algorithm_runtimes import ray_task

    if enabled:
        _owner(monkeypatch)
    result = WorkerExecutionResult(AlgorithmExecutionResult(status="succeeded"), {})
    task, reducer = MagicMock(), MagicMock()
    task.options.return_value.remote.return_value = result
    reducer.remote.return_value = result
    reducer.options.return_value.remote.return_value = result
    monkeypatch.setattr(ray_task, "_execute_on_worker", task)
    monkeypatch.setattr(ray_task, "_reduce_on_worker", reducer)
    monkeypatch.setattr(ray_task.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray_task.ray, "get", lambda value: value)
    init = MagicMock(side_effect=AssertionError("must not reinitialize caller Ray"))
    monkeypatch.setattr(ray_task.ray, "init", init)
    topology = SimpleNamespace(
        topology=RuntimeTopology.DATA_PARALLEL,
        worker_count=2,
        framework_parallelism=1,
        num_cpus=0,
        num_gpus=0,
        memory_bytes=None,
        max_retries=0,
    )
    envelope = SimpleNamespace(
        plan=SimpleNamespace(runtime=topology),
        input_payloads=(object(), object()),
        artifacts=(),
        cancelled=False,
    )
    assert ray_task.RayTaskRuntime().execute(envelope) is result
    assert task.options.call_count == 2
    for call in task.options.call_args_list:
        if enabled:
            runtime_env = call.kwargs["runtime_env"]
            assert set(runtime_env) == {"env_vars"}
            policy = json.loads(runtime_env["env_vars"][POLICY_ENV])
            assert policy["role"] == "producer" and "read_profile" not in policy
        else:
            assert "runtime_env" not in call.kwargs
    assert reducer.options.call_count == int(enabled)
    init.assert_not_called()


def _owner(
    monkeypatch: pytest.MonkeyPatch, startup_environment: dict[str, str] | None = None
) -> sessions.ReportingSession:
    profile = DorisProfile(
        host="owner-only.invalid",
        database="logs",
        username_ref="env://OWNER_READ_USER",
        password_ref="env://OWNER_READ_PASSWORD",
    )
    owner = sessions.ReportingSession(
        ReportingConfig(enabled=True, mode="external", read_profile=profile),
        startup_environment=startup_environment,
    )
    monkeypatch.setattr(sessions, "_ACTIVE", owner)
    return owner


def test_policy_is_producer_only_and_runtime_env_is_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner(monkeypatch)
    caller = {
        "env_vars": {"USER_SETTING": "keep"},
        "worker_process_setup_hook": "trusted.user_hook",
        "num_cpus": 2,
    }
    original = deepcopy(caller)
    merged = sessions.merge_runtime_env(caller)
    payload = json.loads(merged["env_vars"][POLICY_ENV])
    assert payload["role"] == "producer" and payload["version"] == 1
    assert payload["context"]["operation_id"] == owner.operation_id
    decoded, context = sessions._decode_policy(merged["env_vars"][POLICY_ENV])
    assert (
        decoded == owner.sender_config and context["operation_id"] == owner.operation_id
    )
    assert (
        "read_profile" not in payload
        and "owner-only" not in merged["env_vars"][POLICY_ENV]
    )
    assert "OWNER_READ" not in merged["env_vars"][POLICY_ENV]
    assert caller == original
    assert merged["worker_process_setup_hook"] == WORKER_HOOK
    assert merged["env_vars"]["_TRIBUTO_REPORTING_USER_HOOK"] == "trusted.user_hook"
    assert sessions.merge_runtime_env(merged) == merged
    assert owner._collector is None


def test_remote_job_does_not_borrow_submitter_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _owner(monkeypatch)
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_ENABLED", "1")
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_CONFIG", "/remote/execution/config.json")
    result = sessions.merge_runtime_env(
        {"worker_process_setup_hook": "trusted.job_hook"}, for_job=True
    )
    assert POLICY_ENV not in result["env_vars"]
    assert (
        result["env_vars"]["TRIBUTO_LOG_REPORTING_CONFIG"]
        == "/remote/execution/config.json"
    )
    assert result["env_vars"]["_TRIBUTO_REPORTING_USER_HOOK"] == "trusted.job_hook"
    assert result["worker_process_setup_hook"] == WORKER_HOOK
    assert sessions.merge_runtime_env(result, for_job=True) == result
    from ray.runtime_env.runtime_env import _merge_runtime_env

    for name, value in result["env_vars"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("_TRIBUTO_LOG_JOB_HOOK", "external")
    monkeypatch.setenv(
        "RAY_JOB_CONFIG_JSON_ENV_VAR", json.dumps({"runtime_env": result})
    )
    driver = sessions.ray_options(driver=True)["runtime_env"]
    assert POLICY_ENV in driver["env_vars"]
    assert _merge_runtime_env(result, driver) is not None
    with pytest.raises(ReportingError, match="reserved"):
        sessions.merge_runtime_env({"env_vars": {POLICY_ENV: "forged"}}, for_job=True)
    with pytest.raises(ReportingError, match="reserved_hook"):
        sessions.merge_runtime_env(
            {"env_vars": {"_TRIBUTO_REPORTING_USER_HOOK": "forged.hook"}}
        )


def test_callable_hook_is_idempotently_composed_in_order(
    monkeypatch: pytest.MonkeyPatch,
    isolated_logging,
) -> None:
    import logging

    from tributo._common.logging_policy import LoggingPolicy, logging_runtime_env

    _owner(monkeypatch)
    from tributo._common.reporting import worker

    calls = []
    monkeypatch.setattr(
        worker, "bootstrap_producer", lambda **kwargs: calls.append("reporting")
    )

    def user_hook() -> None:
        assert logging.getLogger("tributo").level == logging.DEBUG
        calls.append("user")

    result = sessions.merge_runtime_env(
        logging_runtime_env(
            {"worker_process_setup_hook": user_hook},
            policy=LoggingPolicy(level="DEBUG"),
        )
    )
    repeated = sessions.merge_runtime_env(result)
    assert repeated["worker_process_setup_hook"] is result["worker_process_setup_hook"]
    repeated["worker_process_setup_hook"]()
    assert calls == ["reporting", "user"]


def test_string_hook_resolves_callable_and_preserves_user_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_logging,
) -> None:
    from tributo._common.logging_policy import LoggingPolicy, logging_runtime_env
    from tributo._common.reporting import worker

    _owner(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        worker, "bootstrap_producer", lambda **kwargs: calls.append("reporting")
    )
    runtime = sessions.merge_runtime_env(
        logging_runtime_env(
            {
                "worker_process_setup_hook": "tests.reporting.test_worker_policy._trusted_failing_hook"
            },
            policy=LoggingPolicy(level="DEBUG"),
        )
    )
    for name, value in runtime["env_vars"].items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="trusted-hook-failure"):
        worker.setup()
    assert calls == ["reporting"]


def _trusted_failing_hook() -> None:
    raise RuntimeError("trusted-hook-failure")


@pytest.mark.parametrize(
    "module", ["inference.batch_job", "explainability.batch_job", "vector_index.job"]
)
@pytest.mark.parametrize(
    "user_hook", [None, "tests.reporting.test_worker_policy._trusted_failing_hook"]
)
def test_public_job_submit_composes_hooks_once(
    monkeypatch, isolated_logging, tmp_path, module, user_hook
):
    import logging
    from importlib import import_module

    from tributo import job
    from tributo._common import worker_logging
    from tributo._common.runtime_env import build_runtime_env

    client = MagicMock()
    client.submit_job.return_value = "job-id"
    monkeypatch.setattr(job, "JobSubmissionClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(worker_logging, "_dispatched", None)
    monkeypatch.setenv("TRIBUTO_LOG_REPORTING_CONFIG", "/remote/reporting-config.json")
    (tmp_path / "tributo").mkdir()
    runtime = build_runtime_env(
        project_root=tmp_path,
        env_vars={"TRIBUTO_LOG_REPORTING_ENABLED": "1", "TRIBUTO_LOG_LEVEL": "DEBUG"},
    )
    assert (
        runtime["env_vars"]["TRIBUTO_LOG_REPORTING_CONFIG"]
        == "/remote/reporting-config.json"
    )
    if user_hook:
        runtime["worker_process_setup_hook"] = user_hook
    original = deepcopy(runtime)
    assert (
        job.TributoClient("http://unused.invalid:8265").submit(
            "python -m tributo." + module, runtime_env=runtime
        )
        == "job-id"
    )
    assert runtime == original
    submitted = client.submit_job.call_args.kwargs["runtime_env"]
    for key, value in submitted["env_vars"].items():
        monkeypatch.setenv(key, value)
    name, _, attribute = submitted["worker_process_setup_hook"].rpartition(".")
    hook = getattr(import_module(name), attribute)
    if user_hook:
        with pytest.raises(RuntimeError, match="trusted-hook-failure"):
            hook()
    else:
        hook()
    assert logging.getLogger("tributo").level == logging.DEBUG


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        (None, "original-run"),
        ({"APP": "keep"}, "original-run"),
        ({"TRIBUTO_RUN_ID": "local-run"}, "local-run"),
        ({"TRIBUTO_RUN_ID": ""}, None),
    ],
)
def test_correlation_environment_is_frozen_at_domain_start(
    monkeypatch, supplied, expected
):
    monkeypatch.setenv("TRIBUTO_RUN_ID", "original-run")
    before = dict(supplied or {})
    owner = _owner(monkeypatch, supplied)
    monkeypatch.setenv("TRIBUTO_RUN_ID", "changed-node-run")
    payload = json.loads(owner.producer_policy())
    assert payload["context"].get("run_id") == expected
    assert ("run_id" in payload["context"]) is (expected is not None)
    assert (supplied or {}) == before


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "ready"),
        *(
            ({key: "secret"}, "reporting.config.otel_exporter_conflict")
            for key in (
                "OTEL_EXPORTER_OTLP_HEADERS",
                "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
                "OTEL_EXPORTER_OTLP_CERTIFICATE",
                "OTEL_EXPORTER_OTLP_LOGS_CLIENT_KEY",
                "OTEL_EXPORTER_OTLP_LOGS_CLIENT_CERTIFICATE",
                "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
                "OTEL_PYTHON_EXPORTER_OTLP_HTTP_LOGS_CREDENTIAL_PROVIDER",
            )
        ),
        ({"OTEL_SDK_DISABLED": "true"}, "reporting.config.sdk_disabled_conflict"),
        (
            {
                "OTEL_SDK_DISABLED": "false",
                "OTEL_BLRP_MAX_QUEUE_SIZE": "invalid",
                "OTEL_SERVICE_NAME": "caller-owned-service",
                "OTEL_ATTRIBUTE_COUNT_LIMIT": "invalid",
                "OTEL_LOGRECORD_ATTRIBUTE_VALUE_LENGTH_LIMIT": "invalid",
                "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "caller-owned",
            },
            "ready",
        ),
    ],
)
def test_real_sdk_constructs_under_distinct_driver_worker_environments(
    changes: dict[str, str], expected: str
) -> None:
    code = """
import json, os
from tributo._common.reporting.sender import Sender
from tributo.observability import SenderConfig, ReportingError
from opentelemetry._logs import get_logger_provider
provider=get_logger_provider()
environment=dict(os.environ)
try:
    sender=Sender(SenderConfig(endpoint="http://127.0.0.1:1/v1/logs",shutdown_timeout=1.0),lambda code: None)
except ReportingError as error:
    print(json.dumps({"result":str(error)}))
else:
    assert sender._provider.resource.attributes["service.name"]=="tributo"
    assert sender.close()
    assert get_logger_provider() is provider
    print(json.dumps({"result":"ready"}))
assert dict(os.environ)==environment
"""
    env = {
        **os.environ,
        **changes,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert json.loads(result.stdout)["result"] == expected
    assert "secret" not in result.stdout + result.stderr
    for key, value in changes.items():
        assert os.environ.get(key) != value
