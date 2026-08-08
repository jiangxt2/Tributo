"""Tests for Ray runtime environment construction."""

from __future__ import annotations

import logging

import pytest

from tributo._common.runtime_env import build_runtime_env


def test_runtime_env_debug_log_never_exposes_environment_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile_payload = "opaque-profile-payload"

    with caplog.at_level(logging.DEBUG, logger="tributo._common.runtime_env"):
        runtime_env = build_runtime_env(
            env_vars={"TRIBUTO_STORAGE_PROFILE_MODEL": profile_payload}
        )

    assert runtime_env["env_vars"]["TRIBUTO_STORAGE_PROFILE_MODEL"] == profile_payload
    assert profile_payload not in caplog.text
    assert "TRIBUTO_STORAGE_PROFILE_MODEL" in caplog.text
