"""Tests for cluster-side legacy and frozen-plan entry points."""

from __future__ import annotations

import argparse
import base64
import logging
from unittest.mock import MagicMock, patch

import pytest

from tests.inference.test_executor import _plan
from tributo._common.logging_policy import POLICY_ENV, SETUP_HOOK, LoggingPolicy
from tributo.inference.batch_job import _parse_args, _run_resolved_plan, main

pytestmark = pytest.mark.usefixtures("isolated_logging")


def _encoded_plan() -> str:
    return base64.urlsafe_b64encode(_plan().model_dump_json().encode("utf-8")).decode(
        "ascii"
    )


def test_cli_requires_exactly_one_input_mode() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])
    with pytest.raises(SystemExit):
        _parse_args(["--config", "inference.json", "--resolved-plan-env", "PLAN"])


def test_resolved_plan_identity_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setenv("PLAN", _encoded_plan())
    monkeypatch.setenv("TRIBUTO_RUN_ID", "other-run")
    monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", plan.attempt_id)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", plan.submission_id)

    with pytest.raises(ValueError, match="run_id conflicts"):
        _run_resolved_plan("PLAN")


def test_resolved_plan_submission_identity_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setenv("PLAN", _encoded_plan())
    monkeypatch.setenv("TRIBUTO_RUN_ID", plan.run_id)
    monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", plan.attempt_id)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", "other-submission")

    with pytest.raises(ValueError, match="submission_id conflicts"):
        _run_resolved_plan("PLAN")


def test_main_executes_frozen_plan_and_returns_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setenv("PLAN", _encoded_plan())
    monkeypatch.setenv("TRIBUTO_RUN_ID", plan.run_id)
    monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", plan.attempt_id)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", plan.submission_id)
    result = MagicMock(status="succeeded")

    with (
        patch("tributo.inference.batch_job.ray.init") as ray_init,
        patch("tributo.inference.api.run_resolved_inference", return_value=result),
    ):
        exit_code = main(argparse.Namespace(config=None, resolved_plan_env="PLAN"))

    assert exit_code == 0
    assert ray_init.call_count == 1
    assert ray_init.call_args.kwargs["address"] == "auto"
    runtime = ray_init.call_args.kwargs["runtime_env"]
    assert runtime["worker_process_setup_hook"] == SETUP_HOOK
    assert (
        LoggingPolicy.deserialize(runtime["env_vars"][POLICY_ENV]).log_format == "json"
    )


def test_main_returns_failure_for_structured_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setenv("PLAN", _encoded_plan())
    monkeypatch.setenv("TRIBUTO_RUN_ID", plan.run_id)
    monkeypatch.setenv("TRIBUTO_ATTEMPT_ID", plan.attempt_id)
    monkeypatch.setenv("TRIBUTO_SUBMISSION_ID", plan.submission_id)
    result = MagicMock(status="failed")
    result.model_dump.return_value = {"status": "failed"}

    with (
        patch("tributo.inference.batch_job.ray.init"),
        patch("tributo.inference.api.run_resolved_inference", return_value=result),
    ):
        exit_code = main(argparse.Namespace(config=None, resolved_plan_env="PLAN"))

    assert exit_code == 1


def test_main_logs_only_unhandled_exception_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("tributo.inference.batch_job.bootstrap_logging"),
        patch("tributo.inference.batch_job.ray.init"),
        patch(
            "tributo.inference.batch_job._run_resolved_plan",
            side_effect=RuntimeError("must-not-leak"),
        ),
        caplog.at_level(logging.ERROR, logger="tributo.inference.batch_job"),
    ):
        exit_code = main(argparse.Namespace(config=None, resolved_plan_env="PLAN"))

    assert exit_code == 1
    assert "RuntimeError" in caplog.text
    assert "must-not-leak" not in caplog.text


def test_owned_child_main_passes_frozen_driver_policy_to_workers(monkeypatch):
    from tributo._common.logging_policy import JOB_HOOK_ENV

    selected = LoggingPolicy(level="DEBUG", log_format="text")
    monkeypatch.setenv(POLICY_ENV, selected.serialize())
    monkeypatch.delenv(JOB_HOOK_ENV, raising=False)
    with (
        patch("tributo.inference.batch_job.ray.init") as init,
        patch(
            "tributo.inference.batch_job._run_resolved_plan",
            return_value=MagicMock(status="succeeded"),
        ),
    ):
        assert main(argparse.Namespace(config=None, resolved_plan_env="PLAN")) == 0
    runtime = init.call_args.kwargs["runtime_env"]
    assert LoggingPolicy.deserialize(runtime["env_vars"][POLICY_ENV]) == selected
    assert runtime["worker_process_setup_hook"] == SETUP_HOOK
