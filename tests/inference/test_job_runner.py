"""Tests for inference/job_runner.py."""

from __future__ import annotations

import base64
import sys
from unittest.mock import MagicMock, patch

import pytest
from ray.job_submission import JobStatus

from tests.inference.test_executor import _plan
from tributo.inference.contracts import ResolvedInference
from tributo.inference.job_runner import (
    map_ray_job_status,
    submit_inference_job,
    submit_inference_request,
    submit_inference_request_with_retry,
    submit_resolved_inference,
    wait_for_job,
)


class TestSubmitInferenceJob:
    """Tests for submit_inference_job."""

    def test_passes_submission_id(self):
        """应生成并传递 submission_id 给 Ray API。"""
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-123"

        with patch(
            "tributo.inference.job_runner.JobSubmissionClient",
            return_value=mock_client,
        ):
            submit_inference_job(
                config_path="jobs/inference.yaml",
                dashboard_url="http://127.0.0.1:8265",
            )

        call_kwargs = mock_client.submit_job.call_args.kwargs
        assert "submission_id" in call_kwargs
        assert call_kwargs["submission_id"].startswith("tributo-infer-")

    def test_submission_id_is_deterministic(self):
        """相同参数应产生相同的 submission_id。"""
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-123"

        ids = []
        for _ in range(2):
            with patch(
                "tributo.inference.job_runner.JobSubmissionClient",
                return_value=mock_client,
            ):
                submit_inference_job(
                    config_path="jobs/inference.yaml",
                    dashboard_url="http://127.0.0.1:8265",
                )
            ids.append(mock_client.submit_job.call_args.kwargs["submission_id"])

        assert ids[0] == ids[1]

    def test_client_creation_is_retried_on_connection_error(self):
        """连接失败时应重试。"""
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-123"

        side_effects = [ConnectionError("refused"), mock_client]

        with patch(
            "tributo.inference.job_runner.JobSubmissionClient",
            side_effect=side_effects,
        ) as mock_constructor:
            job_id = submit_inference_job(
                config_path="jobs/inference.yaml",
                dashboard_url="http://127.0.0.1:8265",
            )

        assert job_id.startswith("tributo-infer-")
        assert mock_constructor.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))


def _runtime_env(*args, **kwargs):
    del args
    return {"env_vars": kwargs.get("env_vars", {})}


class TestResolvedRequestSubmission:
    def test_already_resolved_plan_is_submitted_without_a_resolver(self) -> None:
        plan = _plan()
        client = MagicMock()
        client.submit_job.return_value = "job-frozen"

        with (
            patch(
                "tributo.inference.job_runner._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.inference.job_runner.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            job_id = submit_resolved_inference(plan)

        assert job_id == plan.submission_id
        assert client.submit_job.call_args.kwargs["submission_id"] == plan.submission_id

    def test_frozen_plan_is_transported_without_re_resolution_in_job(self) -> None:
        plan = _plan()
        resolver = MagicMock()
        resolver.resolve.return_value = plan
        client = MagicMock()
        client.submit_job.return_value = "job-1"

        with (
            patch(
                "tributo.inference.job_runner._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.inference.job_runner.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            job_id = submit_inference_request(object(), resolver=resolver)

        assert job_id == plan.submission_id
        resolver.resolve.assert_called_once()
        call = client.submit_job.call_args.kwargs
        assert call["submission_id"] == plan.submission_id
        assert (
            call["runtime_env"]["env_vars"]["TRIBUTO_SUBMISSION_ID"]
            == plan.submission_id
        )
        assert "--resolved-plan-env TRIBUTO_INFERENCE_PLAN_B64" in call["entrypoint"]
        encoded = call["runtime_env"]["env_vars"]["TRIBUTO_INFERENCE_PLAN_B64"]
        decoded = base64.urlsafe_b64decode(encoded).decode("utf-8")
        transported = ResolvedInference.model_validate_json(decoded)
        assert transported == plan
        assert "secret_access_key" not in decoded

    def test_ambiguous_submission_is_reconciled_by_deterministic_id(self) -> None:
        plan = _plan()
        resolver = MagicMock()
        resolver.resolve.return_value = plan
        client = MagicMock()
        client.submit_job.side_effect = TimeoutError("ambiguous")
        client.get_job_status.return_value = JobStatus.RUNNING

        with (
            patch(
                "tributo.inference.job_runner._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.inference.job_runner.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            job_id = submit_inference_request(object(), resolver=resolver)

        assert job_id == plan.submission_id
        client.get_job_status.assert_called_once_with(plan.submission_id)

    def test_reserved_identity_environment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved inference keys"):
            submit_inference_request(
                object(), env_vars={"TRIBUTO_SUBMISSION_ID": "override"}
            )

    def test_oversized_frozen_plan_fails_before_ray_job_submission(self) -> None:
        plan = _plan()
        plan = plan.model_copy(
            update={
                "model": plan.model.model_copy(
                    update={"source_provenance": "p" * 70_000}
                )
            }
        )
        client = MagicMock()

        with patch(
            "tributo.inference.job_runner._get_submission_client",
            return_value=client,
        ):
            with pytest.raises(ValueError, match="65536-byte"):
                submit_resolved_inference(plan)

        client.submit_job.assert_not_called()


class TestJobStatusAndRetry:
    def test_timeout_boundary_rechecks_terminal_status(self) -> None:
        client = MagicMock()
        client.get_job_status.return_value = JobStatus.SUCCEEDED
        client.get_job_logs.return_value = "completed"

        result = wait_for_job(client, "job-1", timeout=0, poll_interval=0)

        assert result == {"status": JobStatus.SUCCEEDED, "logs": "completed"}
        client.get_job_status.assert_called_once_with("job-1")

    def test_terminal_status_survives_log_retrieval_failure(self, caplog) -> None:
        client = MagicMock()
        client.get_job_status.return_value = JobStatus.FAILED
        client.get_job_logs.side_effect = ConnectionError("head restarted")

        with caplog.at_level("WARNING", logger="tributo.inference.job_runner"):
            result = wait_for_job(client, "job-1", timeout=0, poll_interval=0)

        assert result == {"status": JobStatus.FAILED, "logs": ""}
        assert "ConnectionError" in caplog.text
        assert "head restarted" not in caplog.text

    def test_timeout_after_final_nonterminal_probe(self) -> None:
        client = MagicMock()
        client.get_job_status.return_value = JobStatus.RUNNING

        with pytest.raises(TimeoutError, match="did not complete"):
            wait_for_job(client, "job-1", timeout=0, poll_interval=0)

        client.get_job_logs.assert_not_called()

    @pytest.mark.parametrize(
        "ray_status, domain_status",
        [
            (JobStatus.PENDING, "pending"),
            (JobStatus.RUNNING, "running"),
            (JobStatus.SUCCEEDED, "succeeded"),
            (JobStatus.FAILED, "failed"),
            (JobStatus.STOPPED, "cancelled"),
        ],
    )
    def test_ray_status_mapping(self, ray_status, domain_status: str) -> None:
        assert map_ray_job_status(ray_status) == domain_status

    def test_stopped_is_cancelled_and_never_retried(self) -> None:
        resolver = MagicMock()
        resolver.resolve.return_value = _plan()
        client = MagicMock()
        client.submit_job.return_value = "job-1"
        client.get_job_status.return_value = JobStatus.STOPPED
        client.get_job_logs.return_value = "cancelled"
        classifier = MagicMock(return_value=True)

        with (
            patch(
                "tributo.inference.job_runner._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.inference.job_runner.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_inference_request_with_retry(
                object(),
                resolver=resolver,
                max_attempts=3,
                timeout=1,
                poll_interval=0,
                retry_classifier=classifier,
            )

        assert result.status == "cancelled"
        assert len(result.attempts) == 1
        assert client.submit_job.call_count == 1
        classifier.assert_not_called()

    def test_explicit_classifier_can_retry_failed_attempt(self) -> None:
        resolver = MagicMock()
        resolver.resolve.return_value = _plan()
        client = MagicMock()
        client.submit_job.side_effect = ["job-1", "job-2"]
        client.get_job_status.side_effect = [JobStatus.FAILED, JobStatus.SUCCEEDED]
        client.get_job_logs.side_effect = ["transient", "ok"]

        with (
            patch(
                "tributo.inference.job_runner._get_submission_client",
                return_value=client,
            ),
            patch(
                "tributo.inference.job_runner.build_runtime_env",
                side_effect=_runtime_env,
            ),
        ):
            result = submit_inference_request_with_retry(
                object(),
                resolver=resolver,
                max_attempts=2,
                timeout=1,
                poll_interval=0,
                retry_classifier=lambda status, logs: logs == "transient",
            )

        assert result.status == "succeeded"
        assert [attempt.attempt_id for attempt in result.attempts] == [
            "attempt-1",
            "attempt-2",
        ]
        assert result.attempts[0].submission_id != result.attempts[1].submission_id
