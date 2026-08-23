"""Workload-neutral Ray Jobs submission identity tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ray.job_submission import JobStatus

from tributo import RuntimeTarget
from tributo.exceptions import JobConfigurationError
from tributo.ray_jobs import (
    RayJobSubmission,
    get_ray_job_logs,
    get_ray_job_status,
    stop_ray_job,
    submit_ray_job,
)


def test_submission_identity_is_workload_neutral_and_ray_job_id_is_real() -> None:
    client = MagicMock()
    client.submit_job.return_value = "ray-api-return-value"
    client.get_job_info.return_value = type(
        "JobInfo", (), {"job_id": "ray-core-job-1"}
    )()

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client),
        patch(
            "tributo.ray_jobs.build_runtime_env",
            autospec=True,
            return_value={"env_vars": {}},
        ),
    ):
        result = submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-1",
            attempt_id="attempt-2",
        )

    assert isinstance(result, RayJobSubmission)
    assert result.run_id == "run-1"
    assert result.attempt_id == "attempt-2"
    assert result.submission_id.startswith("tributo-broker-")
    assert result.ray_job_id == "ray-core-job-1"
    assert client.submit_job.call_args.kwargs["submission_id"] == result.submission_id


def test_submit_ray_job_uses_an_attached_runtime_target() -> None:
    client = MagicMock()
    client.get_job_info.return_value = type(
        "JobInfo", (), {"job_id": "ray-core-job-target"}
    )()
    target = RuntimeTarget.from_master("http://ray-head:8265")

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client) as get,
        patch(
            "tributo.ray_jobs.build_runtime_env",
            autospec=True,
            return_value={"env_vars": {}},
        ),
    ):
        result = submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-target",
            runtime_target=target,
        )

    get.assert_called_once_with("http://ray-head:8265")
    assert result.ray_job_id == "ray-core-job-target"


def test_submit_ray_job_rejects_owned_or_local_targets() -> None:
    with pytest.raises(JobConfigurationError, match="lifecycle-aware"):
        submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-local",
            runtime_target=RuntimeTarget.from_master("local"),
        )

    with pytest.raises(JobConfigurationError, match="lifecycle-aware"):
        submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-managed",
            runtime_target=RuntimeTarget.from_master(
                "managed://ray_cluster_launcher/provider.json"
            ),
        )


def test_submit_ray_job_forwards_trusted_runtime_env_extensions(
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.get_job_info.return_value = type(
        "JobInfo", (), {"job_id": "ray-core-job-2"}
    )()
    extensions: list[str | Path] = [
        tmp_path / "driver.whl",
        "s3://artifacts/support.zip",
    ]
    packages = ["driver-runtime==1.2.3", "/artifacts/support.whl"]
    built_runtime_env = {
        "py_modules": [str(tmp_path / "tributo"), *map(str, extensions)],
        "pip": list(packages),
    }

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client),
        patch(
            "tributo.ray_jobs.build_runtime_env",
            autospec=True,
            return_value=built_runtime_env,
        ) as build_runtime_env_mock,
    ):
        result = submit_ray_job(
            "python -m extension.execution_driver",
            operation_namespace="broker",
            run_id="run-2",
            attempt_id="attempt-3",
            env_vars={"DEPLOYMENT_MODE": "trusted"},
            project_root=tmp_path,
            extra_excludes=["local-cache/**"],
            extra_py_modules=extensions,
            runtime_pip_packages=packages,
            metadata={"operation_type": "training"},
            request_digest="digest-2",
            entrypoint_num_cpus=1.5,
            entrypoint_num_gpus=0.5,
            entrypoint_memory=1024,
        )

    build_runtime_env_mock.assert_called_once_with(
        project_root=tmp_path,
        env_vars={
            "DEPLOYMENT_MODE": "trusted",
            "TRIBUTO_RUN_ID": "run-2",
            "TRIBUTO_ATTEMPT_ID": "attempt-3",
            "TRIBUTO_SUBMISSION_ID": result.submission_id,
        },
        extra_excludes=["local-cache/**"],
        extra_py_modules=extensions,
        runtime_pip_packages=packages,
    )
    submit_call = client.submit_job.call_args.kwargs
    assert submit_call["runtime_env"] is built_runtime_env
    assert submit_call["metadata"] == {
        "operation_type": "training",
        "tributo.request_digest": "digest-2",
    }
    assert submit_call["submission_id"] == result.submission_id
    assert submit_call["entrypoint_num_cpus"] == 1.5
    assert submit_call["entrypoint_num_gpus"] == 0.5
    assert submit_call["entrypoint_memory"] == 1024
    assert result.ray_job_id == "ray-core-job-2"


def test_ambiguous_submission_reconciles_by_submission_id() -> None:
    client = MagicMock()
    client.submit_job.side_effect = TimeoutError("response lost")
    client.get_job_status.return_value = JobStatus.RUNNING
    client.get_job_info.side_effect = LookupError("driver not created")

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client),
        patch(
            "tributo.ray_jobs.build_runtime_env",
            autospec=True,
            return_value={"env_vars": {}},
        ),
    ):
        result = submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-1",
        )

    client.get_job_status.assert_called_once_with(result.submission_id)
    assert result.ray_job_id is None


def test_request_digest_is_optional_metadata_not_submission_identity() -> None:
    client = MagicMock()
    client.get_job_info.return_value = type("JobInfo", (), {"job_id": None})()

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client),
        patch(
            "tributo.ray_jobs.build_runtime_env",
            autospec=True,
            return_value={"env_vars": {}},
        ),
    ):
        first = submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-1",
            request_digest="digest-a",
        )
        second = submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-1",
            request_digest="digest-b",
        )

    assert first.submission_id == second.submission_id
    assert client.submit_job.call_args_list[0].kwargs["metadata"] == {
        "tributo.request_digest": "digest-a"
    }
    assert client.submit_job.call_args_list[1].kwargs["metadata"] == {
        "tributo.request_digest": "digest-b"
    }


def test_reserved_identity_environment_is_rejected_before_submission() -> None:
    with pytest.raises(ValueError, match="must not override Ray Job identity"):
        submit_ray_job(
            "python -m provider.driver",
            operation_namespace="broker",
            run_id="run-1",
            env_vars={"TRIBUTO_SUBMISSION_ID": "external"},
        )


def test_status_and_stop_use_submission_identity() -> None:
    client = MagicMock()
    client.get_job_status.return_value = JobStatus.RUNNING
    client.get_job_logs.return_value = "driver logs"
    client.stop_job.return_value = True

    with patch("tributo.ray_jobs._get_submission_client", return_value=client):
        assert get_ray_job_status("submission-1") == "RUNNING"
        assert get_ray_job_logs("submission-1") == "driver logs"
        assert stop_ray_job("submission-1") is True

    client.get_job_status.assert_called_once_with("submission-1")
    client.get_job_logs.assert_called_once_with("submission-1")
    client.stop_job.assert_called_once_with("submission-1")
