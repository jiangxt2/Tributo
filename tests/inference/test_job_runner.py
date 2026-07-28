"""Tests for inference/job_runner.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


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
            from tributo.inference.job_runner import submit_inference_job

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
                from tributo.inference.job_runner import submit_inference_job

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
            from tributo.inference.job_runner import submit_inference_job

            job_id = submit_inference_job(
                config_path="jobs/inference.yaml",
                dashboard_url="http://127.0.0.1:8265",
            )

        assert job_id == "job-123"
        assert mock_constructor.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
