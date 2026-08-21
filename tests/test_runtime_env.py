"""Tests for Ray runtime environment construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tributo._common.runtime_env import DEFAULT_EXCLUDES, build_runtime_env
from tributo.algorithms.api.artifacts import AlgorithmArtifact
from tributo.training.execution_context import TrainingControlSpec


def test_runtime_env_does_not_inject_python_package_paths_by_default() -> None:
    runtime_env = build_runtime_env()

    assert "env_vars" not in runtime_env


def test_runtime_env_serializes_validated_worker_execution_context() -> None:
    runtime_env = build_runtime_env(
        execution_context={
            "cancellation": TrainingControlSpec(
                "provider.controls:create_checker",
                "job-1",
                {"cancel_key": "cancel:job-1"},
            ).as_dict()
        }
    )

    value = json.loads(runtime_env["env_vars"]["TRIBUTO_EXECUTION_CONTEXT"])
    assert value["schema"] == "tributo.execution-context"
    assert value["cancellation"]["job_id"] == "job-1"


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
    extension_module = "/deployment/private-extension-module"
    runtime_package = "private-extension-package==9.9.9"

    with caplog.at_level(logging.DEBUG, logger="tributo._common.runtime_env"):
        runtime_env = build_runtime_env(
            env_vars={"TRIBUTO_STORAGE_PROFILE_MODEL": profile_payload},
            extra_py_modules=[extension_module],
            runtime_pip_packages=[runtime_package],
        )

    assert runtime_env["env_vars"]["TRIBUTO_STORAGE_PROFILE_MODEL"] == profile_payload
    assert runtime_env["py_modules"][-1] == extension_module
    assert runtime_env["pip"] == [runtime_package]
    assert "PYTHONPATH" not in runtime_env["env_vars"]
    assert profile_payload not in caplog.text
    assert extension_module not in caplog.text
    assert runtime_package not in caplog.text
    assert "TRIBUTO_STORAGE_PROFILE_MODEL" in caplog.text


def test_default_runtime_env_does_not_add_extension_dependencies(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test'\n", encoding="utf-8"
    )
    (tmp_path / "tributo").mkdir()
    runtime_env = build_runtime_env(project_root=tmp_path)
    assert runtime_env == {
        "working_dir": str(tmp_path),
        "excludes": DEFAULT_EXCLUDES,
        "py_modules": [str(tmp_path / "tributo")],
    }


def test_runtime_env_appends_explicit_extension_modules_in_order(tmp_path) -> None:
    (tmp_path / "tributo").mkdir()
    first_extension = tmp_path / "extensions" / "driver.whl"
    second_extension = "s3://deployment-artifacts/shared-runtime.zip"

    runtime_env = build_runtime_env(
        project_root=tmp_path,
        extra_py_modules=[first_extension, second_extension],
    )

    assert runtime_env["py_modules"] == [
        str(tmp_path / "tributo"),
        str(first_extension),
        second_extension,
    ]


def test_runtime_env_adds_only_explicit_nonempty_pip_packages(tmp_path) -> None:
    (tmp_path / "tributo").mkdir()
    packages = ["driver-runtime==1.2.3", "/artifacts/support.whl"]

    runtime_env = build_runtime_env(
        project_root=tmp_path,
        runtime_pip_packages=packages,
    )

    assert runtime_env["pip"] == packages
    assert runtime_env["pip"] is not packages


@pytest.mark.parametrize("runtime_pip_packages", [None, []])
def test_runtime_env_omits_empty_pip_configuration(
    tmp_path: Path,
    runtime_pip_packages: list[str] | None,
) -> None:
    (tmp_path / "tributo").mkdir()

    runtime_env = build_runtime_env(
        project_root=tmp_path,
        runtime_pip_packages=runtime_pip_packages,
    )

    assert "pip" not in runtime_env


def test_runtime_env_rejects_competing_pip_owners(tmp_path) -> None:
    artifact = AlgorithmArtifact(source=str(tmp_path / "algorithm.whl"))

    with pytest.raises(
        ValueError,
        match="runtime_pip_packages cannot be combined with algorithm_artifact",
    ):
        build_runtime_env(
            project_root=tmp_path,
            runtime_pip_packages=["driver-runtime==1.2.3"],
            algorithm_artifact=artifact,
        )


def test_runtime_env_does_not_discover_host_python_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tributo").mkdir()
    host_site_packages = tmp_path / "host" / "site-packages"
    monkeypatch.syspath_prepend(str(host_site_packages))

    runtime_env = build_runtime_env(project_root=tmp_path)

    assert str(host_site_packages) not in runtime_env["py_modules"]
    assert "pip" not in runtime_env
