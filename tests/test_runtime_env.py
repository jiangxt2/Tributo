"""Tests for Ray runtime environment construction."""

from __future__ import annotations

import logging

import pytest

from tributo._common.runtime_env import build_runtime_env


def test_runtime_env_does_not_inject_python_package_paths_by_default() -> None:
    runtime_env = build_runtime_env()

    assert "env_vars" not in runtime_env


def test_runtime_env_preserves_explicit_cluster_pythonpath() -> None:
    runtime_env = build_runtime_env(
        env_vars={"PYTHONPATH": "/opt/cluster/packages"},
    )

    assert runtime_env["env_vars"]["PYTHONPATH"] == "/opt/cluster/packages"


def test_runtime_env_appends_only_explicit_cluster_pythonpath_once() -> None:
    runtime_env = build_runtime_env(
        env_vars={"PYTHONPATH": "/opt/cluster/packages:/opt/shared"},
        pythonpath="/opt/shared:/opt/application",
    )

    assert runtime_env["env_vars"]["PYTHONPATH"] == (
        "/opt/cluster/packages:/opt/shared:/opt/application"
    )


def test_runtime_env_debug_log_never_exposes_environment_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile_payload = "opaque-profile-payload"

    with caplog.at_level(logging.DEBUG, logger="tributo._common.runtime_env"):
        runtime_env = build_runtime_env(
            env_vars={"TRIBUTO_STORAGE_PROFILE_MODEL": profile_payload}
        )

    assert runtime_env["env_vars"]["TRIBUTO_STORAGE_PROFILE_MODEL"] == profile_payload
    assert "PYTHONPATH" not in runtime_env["env_vars"]
    assert profile_payload not in caplog.text
    assert "TRIBUTO_STORAGE_PROFILE_MODEL" in caplog.text


def test_default_runtime_env_does_not_add_provider_dependencies(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test'\n", encoding="utf-8"
    )
    (tmp_path / "tributo").mkdir()
    runtime_env = build_runtime_env(project_root=tmp_path)
    assert runtime_env["py_modules"] == [str(tmp_path / "tributo")]
    assert "pip" not in runtime_env


def test_runtime_env_can_explicitly_inject_provider_dependencies(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test'\n", encoding="utf-8"
    )
    (tmp_path / "tributo").mkdir()
    runtime_env = build_runtime_env(
        project_root=tmp_path,
        extra_py_modules=[tmp_path / "provider"],
        runtime_pip_packages=["tributo-broker-redis==0.1.0"],
    )
    assert runtime_env["py_modules"][-1] == str(tmp_path / "provider")
    assert runtime_env["pip"] == ["tributo-broker-redis==0.1.0"]
