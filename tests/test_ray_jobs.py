"""Workload-neutral Ray Jobs submission identity tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ray.job_submission import JobStatus

from tributo.ray_jobs import (
    RayJobSubmission,
    get_ray_job_logs,
    get_ray_job_status,
    stop_ray_job,
    submit_ray_job,
)


def _runtime_env(*args: Any, **kwargs: Any) -> dict[str, Any]:
    del args
    return {"env_vars": kwargs.get("env_vars", {})}


def test_submission_identity_is_workload_neutral_and_ray_job_id_is_real() -> None:
    client = MagicMock()
    client.submit_job.return_value = "ray-api-return-value"
    client.get_job_info.return_value = type(
        "JobInfo", (), {"job_id": "ray-core-job-1"}
    )()

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client),
        patch("tributo.ray_jobs.build_runtime_env", side_effect=_runtime_env),
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


def test_ambiguous_submission_reconciles_by_submission_id() -> None:
    client = MagicMock()
    client.submit_job.side_effect = TimeoutError("response lost")
    client.get_job_status.return_value = JobStatus.RUNNING
    client.get_job_info.side_effect = LookupError("driver not created")

    with (
        patch("tributo.ray_jobs._get_submission_client", return_value=client),
        patch("tributo.ray_jobs.build_runtime_env", side_effect=_runtime_env),
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
        patch("tributo.ray_jobs.build_runtime_env", side_effect=_runtime_env),
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
