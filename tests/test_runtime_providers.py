"""Tests for explicit Ray runtime provider lifecycles."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from tributo import RuntimeTarget, open_job_submission_client, open_ray_client
from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.runtime_providers import (
    LocalRayJobsProvider,
    RayClusterLauncherProvider,
    RuntimeLease,
    run_local_entrypoint,
)


def test_attached_jobs_target_opens_jobs_client() -> None:
    target = RuntimeTarget.from_master("http://ray-head:8265")
    fake_client = object()
    with patch(
        "tributo.runtime_providers.JobSubmissionClient", return_value=fake_client
    ) as client_cls:
        with open_job_submission_client(target) as client:
            assert client is fake_client
        client_cls.assert_called_once_with("http://ray-head:8265")


def test_local_jobs_provider_exposes_dashboard_and_releases() -> None:
    fake_ray = type(
        "FakeRay",
        (),
        {
            "is_initialized": staticmethod(lambda: False),
            "init": staticmethod(
                lambda **kwargs: type(
                    "Context", (), {"dashboard_url": "127.0.0.1:8265"}
                )()
            ),
            "shutdown": staticmethod(lambda: None),
        },
    )()
    target = RuntimeTarget.from_master("local")
    with patch.dict(sys.modules, {"ray": fake_ray}):
        with LocalRayJobsProvider().provision(target) as lease:
            assert lease.address == "http://127.0.0.1:8265"


def test_managed_provider_provisions_waits_and_releases(tmp_path) -> None:
    provider_config = tmp_path / "provider.json"
    provider_config.write_text(
        json.dumps(
            {
                "cluster_config": str(tmp_path / "cluster.yaml"),
                "dashboard_url": "http://ray-head:8265/",
                "backend": "local",
                "ready_timeout_seconds": 1,
                "ready_poll_seconds": 0.01,
            }
        ),
        encoding="utf-8",
    )
    target = RuntimeTarget.from_master(
        f"managed://ray_cluster_launcher/{provider_config}"
    )
    commands: list[list[str]] = []

    def fake_run(args, **kwargs):
        del kwargs
        commands.append(args)

    with (
        patch("tributo.runtime_providers._run_provider_command", side_effect=fake_run),
        patch("tributo.runtime_providers.requests.get") as get,
    ):
        get.return_value.ok = True
        provider = RayClusterLauncherProvider()
        with provider.provision(target) as lease:
            assert lease.address == "http://ray-head:8265"
        assert commands == [
            ["ray", "up", "-y", str(tmp_path / "cluster.yaml")],
            ["ray", "down", "-y", str(tmp_path / "cluster.yaml")],
        ]


def test_managed_provider_rejects_kubernetes_backend(tmp_path) -> None:
    provider_config = tmp_path / "provider.json"
    provider_config.write_text(
        json.dumps(
            {
                "cluster_config": str(tmp_path / "cluster.yaml"),
                "dashboard_url": "http://ray-head:8265",
                "backend": "kubernetes",
            }
        ),
        encoding="utf-8",
    )
    target = RuntimeTarget.from_master(
        f"managed://ray_cluster_launcher/{provider_config}"
    )

    with pytest.raises(JobConfigurationError, match="KubeRay externally"):
        RayClusterLauncherProvider().provision(target)


def test_managed_provider_cleans_up_when_readiness_fails(tmp_path) -> None:
    provider_config = tmp_path / "provider.json"
    provider_config.write_text(
        json.dumps(
            {
                "cluster_config": str(tmp_path / "cluster.yaml"),
                "dashboard_url": "http://ray-head:8265",
                "backend": "local",
                "ready_timeout_seconds": 0.01,
                "ready_poll_seconds": 0.01,
            }
        ),
        encoding="utf-8",
    )
    target = RuntimeTarget.from_master(
        f"managed://ray_cluster_launcher/{provider_config}"
    )
    commands: list[list[str]] = []

    def fake_run(args, **kwargs):
        del kwargs
        commands.append(args)

    with (
        patch("tributo.runtime_providers._run_provider_command", side_effect=fake_run),
        patch("tributo.runtime_providers.requests.get") as get,
    ):
        get.side_effect = RuntimeError("not ready")
        with pytest.raises((JobExecutionError, RuntimeError)):
            RayClusterLauncherProvider().provision(target)
    assert commands[-1] == [
        "ray",
        "down",
        "-y",
        str(tmp_path / "cluster.yaml"),
    ]


def test_attached_target_does_not_resolve_a_provider() -> None:
    target = RuntimeTarget.from_master("http://ray-head:8265")
    assert not target.is_managed


def test_runtime_cleanup_error_does_not_hide_workload_error() -> None:
    def release() -> None:
        raise RuntimeError("cleanup failed")

    lease = RuntimeLease("http://ray-head:8265", release)
    with pytest.raises(ValueError, match="workload failed") as failure:
        with lease:
            raise ValueError("workload failed")

    assert failure.value.__notes__ == ["Ray runtime cleanup failed: RuntimeError"]


def test_unknown_provider_fails_before_cluster_creation() -> None:
    target = RuntimeTarget.from_master("managed://missing-provider/config.json")
    with pytest.raises(JobConfigurationError, match="not registered"):
        with open_job_submission_client(target):
            pass


def test_local_entrypoint_owns_native_ray_runtime() -> None:
    fake_ray = type(
        "FakeRay",
        (),
        {
            "is_initialized": staticmethod(lambda: False),
            "init": staticmethod(lambda **kwargs: None),
            "shutdown": staticmethod(lambda: None),
        },
    )()
    with (
        patch.dict(sys.modules, {"ray": fake_ray}),
        patch("tributo.runtime_providers.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        run_local_entrypoint(
            "python script.py",
            env_vars={"EXAMPLE": "value"},
            num_cpus=2,
            timeout=30,
        )

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == ["python", "script.py"]
    assert kwargs["env"]["RAY_ADDRESS"] == "auto"
    assert kwargs["env"]["EXAMPLE"] == "value"


def test_ray_client_owns_interactive_client_session() -> None:
    fake_ray = type(
        "FakeRay",
        (),
        {
            "is_initialized": staticmethod(lambda: False),
            "init": staticmethod(lambda **kwargs: None),
            "shutdown": staticmethod(lambda: None),
        },
    )()
    target = RuntimeTarget.from_master("ray://ray-head:10001")
    with patch.dict(sys.modules, {"ray": fake_ray}):
        with open_ray_client(target) as ray_module:
            assert ray_module is fake_ray
