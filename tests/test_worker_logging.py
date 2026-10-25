"""Ray logging transport and setup composition without starting Ray."""

from __future__ import annotations

import json
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest
from ray import cloudpickle

from tributo._common.logging_policy import (
    JOB_HOOK_ENV,
    POLICY_ENV,
    SETUP_HOOK,
    USER_HOOK_ENV,
    LoggingPolicy,
    forward_logging_env,
    logging_runtime_env,
    override_logging_policy,
    submission_logging_env,
)


def test_forwarding_preserves_explicit_values_and_never_scans_other_env(monkeypatch):
    monkeypatch.setenv("TRIBUTO_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("UNRELATED_SECRET", "private")
    original = {"TRIBUTO_LOG_LEVEL": "WARNING", "KEEP": "kept"}
    result = forward_logging_env(original)
    assert result == original
    assert result is not original
    assert "UNRELATED_SECRET" not in result


def test_job_transport_preserves_user_hook_and_is_serializable(monkeypatch):
    monkeypatch.delenv("TRIBUTO_LOG_LEVEL", raising=False)
    original = {
        "env_vars": {"KEEP": "kept"},
        "pip": ["extension"],
        "worker_process_setup_hook": "user.setup",
    }
    result = logging_runtime_env(
        original, policy=LoggingPolicy(level="DEBUG"), jobs=True
    )
    assert result["worker_process_setup_hook"] == SETUP_HOOK
    assert result["env_vars"][USER_HOOK_ENV] == "user.setup"
    assert result["env_vars"][JOB_HOOK_ENV] == "1"
    assert result["env_vars"]["KEEP"] == "kept"
    assert json.loads(json.dumps(result)) == result
    assert (
        logging_runtime_env(result, policy=LoggingPolicy(level="DEBUG"), jobs=True)
        == result
    )
    assert USER_HOOK_ENV not in original["env_vars"]


def test_module_hook_runs_after_logging_once_and_errors_are_not_swallowed(monkeypatch):
    from tributo._common import logging as logging_module
    from tributo._common import worker_logging

    events = []
    module = types.ModuleType("test_logging_user_hook")

    def hook():
        events.append(logging.getLogger("tributo").level)
        logging.getLogger("tributo").setLevel(logging.ERROR)

    module.setup = hook
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(POLICY_ENV, LoggingPolicy(level="DEBUG").serialize())
    monkeypatch.setenv(USER_HOOK_ENV, f"{module.__name__}.setup")
    monkeypatch.setattr(worker_logging, "_dispatched", None)
    monkeypatch.setattr(logging_module, "_applied", None)
    logger = logging.getLogger("tributo")
    previous = logger.level
    try:
        worker_logging.setup()
        worker_logging.setup()
        assert events == [logging.DEBUG]
        assert logger.level == logging.ERROR

        def broken():
            raise RuntimeError("hook failed")

        module.broken = broken
        monkeypatch.setenv(USER_HOOK_ENV, f"{module.__name__}.broken")
        with pytest.raises(RuntimeError, match="hook failed"):
            worker_logging.setup()
    finally:
        logger.setLevel(previous)


def test_local_callable_composition_is_serializable_idempotent_and_not_jobs():
    def original():
        logging.getLogger("tributo.test.hook").setLevel(logging.ERROR)

    policy = LoggingPolicy()
    result = logging_runtime_env({"worker_process_setup_hook": original}, policy=policy)
    restored = cloudpickle.loads(cloudpickle.dumps(result["worker_process_setup_hook"]))
    assert restored.__name__ == "tributo_composed_setup"
    restored()
    assert logging.getLogger("tributo.test.hook").level == logging.ERROR
    assert (
        logging_runtime_env(result, policy=policy)["worker_process_setup_hook"]
        is result["worker_process_setup_hook"]
    )
    with pytest.raises(ValueError, match="string hook"):
        logging_runtime_env({"worker_process_setup_hook": original}, jobs=True)


def test_custom_jobs_are_not_rewritten_and_owned_jobs_install_hook(monkeypatch):
    for key in ("TRIBUTO_LOG_LEVEL", "TRIBUTO_LOG_FORMAT", POLICY_ENV):
        monkeypatch.delenv(key, raising=False)
    original = {"pip": ["extension"]}
    assert submission_logging_env("python train.py", original) == original
    result = submission_logging_env(
        "python -m tributo.inference.batch_job --config run.json", original
    )
    assert result["worker_process_setup_hook"] == SETUP_HOOK
    assert result["env_vars"][JOB_HOOK_ENV] == "1"
    assert original == {"pip": ["extension"]}


def test_conflicting_snapshot_is_rejected_without_echo():
    with pytest.raises(ValueError, match="Invalid application logging configuration"):
        logging_runtime_env(
            {"env_vars": {POLICY_ENV: "private"}}, policy=LoggingPolicy()
        )


def test_custom_job_hook_is_preserved_for_later_runtime_initialization(monkeypatch):
    from tributo._common.logging_policy import driver_logging_runtime_env

    original = {
        "worker_process_setup_hook": "application.setup",
        "env_vars": {"KEEP": "kept"},
    }
    result = submission_logging_env("python custom.py", original)
    assert result["worker_process_setup_hook"] == "application.setup"
    assert result["env_vars"][JOB_HOOK_ENV] == "external"
    assert JOB_HOOK_ENV not in original["env_vars"]
    monkeypatch.setenv(JOB_HOOK_ENV, "external")
    assert driver_logging_runtime_env(policy=LoggingPolicy(level="DEBUG")) == {}


def test_bad_logging_config_does_not_trigger_submission_reconciliation(monkeypatch):
    from tributo.ray_jobs import _submit_ray_job_with_client

    monkeypatch.delenv(POLICY_ENV, raising=False)
    client = MagicMock()
    with pytest.raises(ValueError, match="Invalid application logging configuration"):
        _submit_ray_job_with_client(
            client,
            entrypoint="python custom.py",
            run_id="run",
            attempt_id="attempt",
            submission_id="submission",
            runtime_env={"env_vars": {"TRIBUTO_LOG_LEVEL": "invalid"}},
        )
    client.submit_job.assert_not_called()
    client.get_job_status.assert_not_called()


def test_child_options_override_only_the_selected_leaves():
    original = LoggingPolicy(level="DEBUG", log_format="text", steps=True)
    selected = override_logging_policy(original, {"TRIBUTO_LOG_LEVEL": "WARNING"})
    assert selected == LoggingPolicy(level="WARNING", log_format="text", steps=True)
    assert original.level == "DEBUG"
    with pytest.raises(ValueError, match="Invalid application logging configuration"):
        override_logging_policy(original, {POLICY_ENV: LoggingPolicy().serialize()})
