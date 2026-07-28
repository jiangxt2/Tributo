"""Tests for _common/retry.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tributo._common.retry import retry_with_exponential_backoff


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff decorator."""

    def test_succeeds_on_second_attempt(self):
        """第1次失败，第2次成功。"""
        mock_fn = MagicMock(side_effect=[ConnectionError("timeout"), "success"])

        decorated = retry_with_exponential_backoff(max_retries=3)(mock_fn)
        result = decorated("arg1")

        assert result == "success"
        assert mock_fn.call_count == 2

    def test_exhausted_raises(self):
        """连续3次失败，最终抛出异常。"""
        mock_fn = MagicMock(side_effect=ConnectionError("timeout"))

        decorated = retry_with_exponential_backoff(max_retries=3)(mock_fn)
        with pytest.raises(ConnectionError, match="timeout"):
            decorated()

        assert mock_fn.call_count == 4  # initial + 3 retries

    def test_backoff_intervals(self):
        """验证指数退避间隔：1s, 2s, 4s。"""
        mock_fn = MagicMock(side_effect=ConnectionError("timeout"))
        sleeps = []

        with patch("time.sleep", side_effect=sleeps.append):
            decorated = retry_with_exponential_backoff(max_retries=3, base_delay=1.0)(
                mock_fn
            )
            with pytest.raises(ConnectionError):
                decorated()

        assert sleeps == [1.0, 2.0, 4.0]

    def test_no_retry_on_unexpected_exception(self):
        """非指定异常类型不应触发重试。"""
        mock_fn = MagicMock(side_effect=ValueError("bad input"))

        decorated = retry_with_exponential_backoff(
            max_retries=3, exceptions=(ConnectionError,)
        )(mock_fn)
        with pytest.raises(ValueError, match="bad input"):
            decorated()

        assert mock_fn.call_count == 1

    def test_zero_retries(self):
        """max_retries=0 时不重试。"""
        mock_fn = MagicMock(side_effect=ConnectionError("timeout"))

        decorated = retry_with_exponential_backoff(max_retries=0)(mock_fn)
        with pytest.raises(ConnectionError):
            decorated()

        assert mock_fn.call_count == 1

    def test_preserves_function_metadata(self):
        """装饰器应保留原函数的 __name__ 和 __doc__。"""

        def my_func():
            """My docstring."""
            return 42

        decorated = retry_with_exponential_backoff()(my_func)
        assert decorated.__name__ == "my_func"
        assert decorated.__doc__ == "My docstring."


class TestJobRunnerRetryIntegration:
    """Integration tests for retry in job submission paths."""

    def test_embedding_job_runner_retries_connection(self):
        """embeddings/job_runner 连接失败时触发重试。"""
        client_mock = MagicMock()
        client_mock.submit_job.return_value = "job-123"

        import tributo.embeddings.job_runner as ejr

        with patch.object(
            ejr,
            "JobSubmissionClient",
            side_effect=[
                ConnectionError("timeout"),
                ConnectionError("timeout"),
                client_mock,
            ],
        ) as mock_jsc:
            job_id = ejr.submit_embedding_job(
                s3_input_path="s3://bucket/in.parquet",
                s3_output_path="s3://bucket/out.lance",
            )

        assert job_id == "job-123"
        assert mock_jsc.call_count == 3

    def test_training_job_submitter_retries_connection(self):
        """training/job_submitter 连接失败时触发重试。"""
        client_mock = MagicMock()
        client_mock.submit_job.return_value = "job-456"

        import tributo.training.job_submitter as tjs

        with patch.object(
            tjs,
            "JobSubmissionClient",
            side_effect=[
                ConnectionError("timeout"),
                client_mock,
            ],
        ) as mock_jsc:
            job_id = tjs.submit_training_job("python train.py")

        assert job_id == "job-456"
        assert mock_jsc.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
