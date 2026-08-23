"""Tests for stable Ray job attempts and retry classification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ray.job_submission import JobStatus

from tributo import RuntimeTarget
from tributo.algorithms import AlgorithmArtifact, EnvironmentSpec, ImageProfile
from tributo.exceptions import JobConfigurationError
from tributo.training.job_submitter import (
    submit_training_job,
    submit_training_job_with_identity,
    submit_training_job_with_retry,
)


def _runtime_env(*args, **kwargs):
    return {"env_vars": kwargs.get("env_vars", {})}


class TestStableSubmission:
    def test_local_target_requires_lifecycle_aware_retry_api(self) -> None:
        with pytest.raises(JobConfigurationError, match="with_retry"):
            submit_training_job(
                "python train.py",
                runtime_target=RuntimeTarget.from_master("local"),
            )

    @pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING])
    def test_existing_active_attempt_is_reused(self, status: JobStatus) -> None:
        client = MagicMock()
        client.submit_job.side_effect = RuntimeError("submission already exists")
        client.get_job_status.return_value = status

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            job_id = submit_training_job(
                "python train.py",
                run_id="run-1",
                attempt_id="attempt-1",
            )

        assert job_id.startswith("tributo-train-")
        assert client.submit_job.call_count == 1

    def test_same_run_and_attempt_reuse_submission_id(self) -> None:
        client = MagicMock()
        client.submit_job.return_value = "job-1"

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            first = submit_training_job(
                "python train.py",
                run_id="run-1",
                attempt_id="attempt-1",
            )
            second = submit_training_job(
                "python train.py",
                run_id="run-1",
                attempt_id="attempt-1",
            )

        assert first == second
        assert first.startswith("tributo-train-")
        ids = [
            call.kwargs["submission_id"] for call in client.submit_job.call_args_list
        ]
        assert ids[0] == ids[1]
        worker_env = client.submit_job.call_args.kwargs["runtime_env"]["env_vars"]
        assert "TRIBUTO_RUN_ID" in worker_env
        assert worker_env["TRIBUTO_SUBMISSION_ID"] == ids[-1]

    def test_existing_failed_attempt_is_reconciled_without_timestamp_retry(
        self,
    ) -> None:
        client = MagicMock()
        client.submit_job.side_effect = RuntimeError("submission already exists")
        client.get_job_status.return_value = JobStatus.FAILED
        client.get_job_info.return_value = type(
            "JobInfo", (), {"job_id": "ray-job-1"}
        )()

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            submission = submit_training_job_with_identity(
                "python train.py",
                run_id="run-1",
                attempt_id="attempt-1",
            )

        assert submission.submission_id.startswith("tributo-train-")
        assert submission.ray_job_id == "ray-job-1"
        assert client.submit_job.call_count == 1


class TestRetryClassification:
    def test_local_target_is_released_after_retry_result(self) -> None:
        provider_client = MagicMock()
        provider_client.get_address.return_value = "http://local-ray:8265"
        provider_context = MagicMock()
        provider_context.__enter__.return_value = provider_client
        client = MagicMock()
        client.submit_job.return_value = "job-local"
        client.get_job_status.return_value = JobStatus.SUCCEEDED
        client.get_job_logs.return_value = "ok"

        with (
            patch(
                "tributo.training.job_submitter.open_job_submission_client",
                return_value=provider_context,
            ),
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_training_job_with_retry(
                "python train.py",
                runtime_target=RuntimeTarget.from_master("local"),
                timeout=1,
                poll_interval=0,
            )

        assert result.status == "SUCCEEDED"
        provider_context.__enter__.assert_called_once_with()
        provider_context.__exit__.assert_called_once()

    def test_failed_job_uses_next_attempt_and_succeeded_stops(self) -> None:
        client = MagicMock()
        client.submit_job.side_effect = ["job-1", "job-2"]
        client.get_job_status.side_effect = [JobStatus.FAILED, JobStatus.SUCCEEDED]
        client.get_job_logs.side_effect = ["transient failure", "ok"]

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_training_job_with_retry(
                "python train.py",
                run_id="run-1",
                max_attempts=2,
                timeout=1,
                poll_interval=0,
                retry_classifier=lambda status, logs: True,
            )

        assert result.status == "SUCCEEDED"
        assert result.attempts[0].status == "FAILED"
        assert result.attempts[0].attempt_id == "attempt-1"
        assert result.attempts[1].attempt_id == "attempt-2"
        assert result.attempts[1].status == "SUCCEEDED"
        assert result.attempts[0].submission_id != result.attempts[1].submission_id
        for call, attempt in zip(
            client.submit_job.call_args_list, result.attempts, strict=True
        ):
            assert (
                call.kwargs["runtime_env"]["env_vars"]["TRIBUTO_SUBMISSION_ID"]
                == attempt.submission_id
            )

    def test_stopped_job_is_never_retried(self) -> None:
        client = MagicMock()
        client.submit_job.return_value = "job-1"
        client.get_job_status.return_value = JobStatus.STOPPED
        client.get_job_logs.return_value = "cancelled"

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_training_job_with_retry(
                "python train.py",
                run_id="run-1",
                max_attempts=3,
                timeout=1,
                poll_interval=0,
            )

        assert result.status == "STOPPED"
        assert len(result.attempts) == 1
        assert client.submit_job.call_count == 1

    def test_failed_job_is_not_retried_without_classifier(self) -> None:
        client = MagicMock()
        client.submit_job.return_value = "job-1"
        client.get_job_status.return_value = JobStatus.FAILED
        client.get_job_logs.return_value = "configuration error"

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_training_job_with_retry(
                "python train.py",
                run_id="run-1",
                max_attempts=3,
                timeout=1,
                poll_interval=0,
            )

        assert result.status == "FAILED"
        assert len(result.attempts) == 1

    def test_metadata_is_sent_to_ray_not_worker_environment(self) -> None:
        client = MagicMock()
        client.submit_job.return_value = "job-1"

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            submit_training_job(
                "python train.py",
                run_id="run-1",
                metadata={"team": "training"},
            )

        call = client.submit_job.call_args.kwargs
        assert call["metadata"] == {"team": "training"}
        assert "team" not in call["runtime_env"]["env_vars"]

    def test_metadata_cannot_override_runtime_identity(self) -> None:
        with pytest.raises(ValueError, match="reserved keys"):
            submit_training_job(
                "python train.py",
                run_id="run-1",
                metadata={"TRIBUTO_RUN_ID": "other-run"},
            )

    def test_algorithm_artifact_and_environment_dependencies_reach_preflight(
        self,
    ) -> None:
        client = MagicMock()
        client.submit_job.return_value = "job-artifact"
        artifact = AlgorithmArtifact(source="algorithm.whl")
        profile = ImageProfile(
            profile_id="cpu.test",
            image_uri="tributo:test",
            image_digest="a" * 64,
        )

        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ) as build_runtime_env,
        ):
            job_id = submit_training_job(
                "python train.py",
                run_id="run-artifact",
                algorithm_artifact=artifact,
                image_profile=profile,
                environment=EnvironmentSpec(
                    environment_id="tests.sklearn",
                    dependencies=("scikit-learn>=1.6,<2",),
                ),
            )

        assert job_id.startswith("tributo-train-")
        assert build_runtime_env.call_args.kwargs["algorithm_artifact"] is artifact
        assert build_runtime_env.call_args.kwargs["image_profile"] is profile
        assert build_runtime_env.call_args.kwargs["declared_dependencies"] == (
            "scikit-learn<2,>=1.6",
        )

    def test_metadata_cannot_override_submission_identity(self) -> None:
        with pytest.raises(ValueError, match="TRIBUTO_SUBMISSION_ID"):
            submit_training_job(
                "python train.py",
                run_id="run-1",
                metadata={"TRIBUTO_SUBMISSION_ID": "other-submission"},
            )

    def test_submission_result_preserves_identity_and_optional_digest(self) -> None:
        client = MagicMock()
        client.submit_job.return_value = "submission-return"
        client.get_job_info.return_value = type(
            "JobInfo", (), {"job_id": "ray-job-1"}
        )()
        with (
            patch(
                "tributo.training.job_submitter._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.training.job_submitter.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_training_job_with_identity(
                "python -m worker",
                run_id="business-job-1",
                attempt_id="attempt-2",
                request_digest="request-digest",
            )
        assert result.run_id == "business-job-1"
        assert result.attempt_id == "attempt-2"
        assert result.submission_id.startswith("tributo-train-")
        assert result.ray_job_id == "ray-job-1"
        assert result.request_digest == "request-digest"
        worker_env = client.submit_job.call_args.kwargs["runtime_env"]["env_vars"]
        assert worker_env["TRIBUTO_SUBMISSION_ID"] == result.submission_id
        assert "TRIBUTO_EXECUTION_CONTEXT" not in worker_env
        assert client.submit_job.call_args.kwargs["metadata"] == {
            "tributo.request_digest": "request-digest"
        }
