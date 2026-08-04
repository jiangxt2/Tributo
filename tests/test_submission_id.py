"""Tests for _common/submission_id.py and idempotent job submission."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tributo._common.submission_id import generate_submission_id


class TestGenerateSubmissionId:
    """Tests for generate_submission_id."""

    def test_deterministic(self):
        """相同输入必须产生相同的 ID。"""
        id1 = generate_submission_id("embed", "bge-small-zh", "s3://a/in.parquet")
        id2 = generate_submission_id("embed", "bge-small-zh", "s3://a/in.parquet")
        assert id1 == id2

    def test_unique_per_input(self):
        """不同输入产生不同 ID。"""
        id1 = generate_submission_id("embed", "bge-small-zh", "s3://a/in.parquet")
        id2 = generate_submission_id("embed", "bge-small-zh", "s3://b/in.parquet")
        assert id1 != id2

    def test_prefix_included(self):
        """ID 应包含可读前缀。"""
        sid = generate_submission_id("train", "python train.py")
        assert sid.startswith("tributo-train-")

    def test_length_within_limit(self):
        """ID 长度应远低于 Ray 的 64 字符限制。"""
        sid = generate_submission_id(
            "embed",
            "bge-small-zh",
            "s3://bucket/path/input.parquet",
            "s3://bucket/path/output.lance",
        )
        assert len(sid) <= 40

    def test_different_prefixes_different_ids(self):
        """相同组件但不同 prefix 应产生不同 ID。"""
        id1 = generate_submission_id("embed", "a", "b")
        id2 = generate_submission_id("train", "a", "b")
        assert id1 != id2

    def test_inference_prefix(self):
        """inference 前缀应包含 'infer'。"""
        sid = generate_submission_id("infer", "jobs/inference.yaml")
        assert sid.startswith("tributo-infer-")


class TestJobRunnerSubmissionId:
    """Integration tests for submission_id parameter passing."""

    def test_embedding_job_passes_submission_id(self):
        """embeddings/job_runner 应将 submission_id 传给 Ray API。"""
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-123"

        with patch(
            "tributo.embeddings.job_runner.JobSubmissionClient",
            return_value=mock_client,
        ):
            from tributo.embeddings.job_runner import submit_embedding_job

            submit_embedding_job(
                s3_input_path="s3://bucket/in.parquet",
                s3_output_path="s3://bucket/out.lance",
                model_name="bge-small-zh",
                batch_size=64,
                concurrency=4,
            )

        call_kwargs = mock_client.submit_job.call_args.kwargs
        assert "submission_id" in call_kwargs
        assert call_kwargs["submission_id"].startswith("tributo-embed-")

    def test_canonical_source_with_inline_credentials_is_rejected(self):
        """Ray entrypoints must never carry inline source credentials."""
        from tributo.embeddings.job_runner import submit_embedding_job

        mock_client = MagicMock()
        with patch(
            "tributo.embeddings.job_runner.JobSubmissionClient",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="environment variables"):
                submit_embedding_job(
                    source={
                        "provider": "tributo.clickhouse",
                        "uri": "clickhouse://host/db",
                        "options": {
                            "sql": "SELECT * FROM events",
                            "password": "inline-secret",
                        },
                    },
                    s3_output_path="s3://bucket/out.lance",
                )

        mock_client.submit_job.assert_not_called()

    def test_canonical_sql_text_allows_credential_like_column_literals(self):
        """SQL business text should not be rejected by credential heuristics."""
        from tributo.embeddings.job_runner import submit_embedding_job

        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-sql"
        with patch(
            "tributo.embeddings.job_runner.JobSubmissionClient",
            return_value=mock_client,
        ):
            job_id = submit_embedding_job(
                source={
                    "provider": "tributo.clickhouse",
                    "uri": "clickhouse://host/db",
                    "options": {"sql": "SELECT * FROM users WHERE password = 'x'"},
                },
                s3_output_path="s3://bucket/out.lance",
            )

        assert job_id == "job-sql"
        entrypoint = mock_client.submit_job.call_args.kwargs["entrypoint"]
        assert "password" in entrypoint
        assert "SELECT" in entrypoint

    def test_canonical_source_with_credential_params_is_rejected(self):
        """Parameterized credentials must not enter the Ray entrypoint."""
        from tributo.embeddings.job_runner import submit_embedding_job

        mock_client = MagicMock()
        with patch(
            "tributo.embeddings.job_runner.JobSubmissionClient",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="environment variables"):
                submit_embedding_job(
                    source={
                        "provider": "tributo.clickhouse",
                        "uri": "clickhouse://host/db",
                        "options": {
                            "sql": "SELECT * FROM users WHERE id = {id}",
                            "params": {"password": "inline-secret"},
                        },
                    },
                    s3_output_path="s3://bucket/out.lance",
                )

        mock_client.submit_job.assert_not_called()

    def test_canonical_source_with_uri_userinfo_is_rejected(self):
        """URI userinfo must not enter the Ray entrypoint."""
        from tributo.embeddings.job_runner import submit_embedding_job

        mock_client = MagicMock()
        with patch(
            "tributo.embeddings.job_runner.JobSubmissionClient",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="environment variables"):
                submit_embedding_job(
                    source={
                        "provider": "tributo.clickhouse",
                        "uri": "clickhouse://user:pass@host/db",
                        "options": {"sql": "SELECT 1"},
                    },
                    s3_output_path="s3://bucket/out.lance",
                )

        mock_client.submit_job.assert_not_called()

    def test_training_job_passes_submission_id(self):
        """training/job_submitter 应将 submission_id 传给 Ray API。"""
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-456"

        with patch(
            "tributo.training.job_submitter.JobSubmissionClient",
            return_value=mock_client,
        ):
            from tributo.training.job_submitter import submit_training_job

            submit_training_job("python train.py", env_vars={"EPOCHS": "10"})

        call_kwargs = mock_client.submit_job.call_args.kwargs
        assert "submission_id" in call_kwargs
        assert call_kwargs["submission_id"].startswith("tributo-train-")

    def test_inference_job_passes_submission_id(self):
        """inference/job_runner 应将 submission_id 传给 Ray API。"""
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "job-789"

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


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
