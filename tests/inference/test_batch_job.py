"""Tests for cluster-side legacy and frozen-plan entry points."""

from __future__ import annotations

import argparse
import base64
import logging
from unittest.mock import MagicMock, patch

import pytest

from tests.inference.test_executor import _plan
from tributo.inference.batch_job import _parse_args, _run_resolved_plan, main


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
    ray_init.assert_called_once_with(address="auto")


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
        patch("tributo.inference.batch_job.configure_logging"),
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
