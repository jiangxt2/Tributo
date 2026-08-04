"""Tests for CLI commands using Click's CliRunner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tributo.cli import main


@pytest.fixture
def runner():
    """Provide a Click CliRunner."""
    return CliRunner()


# --- submit ---


class TestSubmitCommand:
    """Tests for the 'submit' CLI command."""

    def test_submit_success(self, runner):
        """submit should print the job ID on success."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit.return_value = "job-abc123"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "submit",
                    "--address",
                    "http://127.0.0.1:8265",
                    "--entrypoint",
                    "python script.py",
                ],
            )
            assert result.exit_code == 0
            assert "job-abc123" in result.output

    def test_submit_with_config_file(self, runner, tmp_path):
        """submit should load config from JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"num_cpus": 4.0, "num_gpus": 1.0}')
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit.return_value = "job-1"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "submit",
                    "--address",
                    "http://127.0.0.1:8265",
                    "--entrypoint",
                    "python train.py",
                    "--config",
                    str(config_file),
                ],
            )
            assert result.exit_code == 0

    def test_submit_failure(self, runner):
        """submit should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.submit.side_effect = RuntimeError("connection refused")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "submit",
                    "--address",
                    "http://127.0.0.1:8265",
                    "--entrypoint",
                    "python script.py",
                ],
            )
            assert result.exit_code == 1
            assert "connection refused" in result.output

    def test_submit_missing_entrypoint(self, runner):
        """submit should fail when --entrypoint is missing."""
        result = runner.invoke(
            main,
            [
                "submit",
                "--address",
                "http://127.0.0.1:8265",
            ],
        )
        assert result.exit_code != 0


# --- status ---


class TestStatusCommand:
    """Tests for the 'status' CLI command."""

    def test_status_success(self, runner):
        """status should print the job status."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_status.return_value = "RUNNING"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "status",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 0
            assert "RUNNING" in result.output

    def test_status_failure(self, runner):
        """status should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_status.side_effect = RuntimeError("not found")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "status",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 1
            assert "not found" in result.output


# --- logs ---


class TestLogsCommand:
    """Tests for the 'logs' CLI command."""

    def test_logs_success(self, runner):
        """logs should print the job logs."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_logs.return_value = "hello\nworld\n"
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "logs",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 0
            assert "hello" in result.output

    def test_logs_failure(self, runner):
        """logs should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.get_logs.side_effect = RuntimeError("timeout")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "logs",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 1
            assert "timeout" in result.output


# --- stop ---


class TestStopCommand:
    """Tests for the 'stop' CLI command."""

    def test_stop_success(self, runner):
        """stop should print success message."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.stop_job.return_value = True
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "stop",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 0
            assert "stopped successfully" in result.output

    def test_stop_failure(self, runner):
        """stop should print error and exit 1 on failure."""
        with patch("tributo.cli.TributoClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.stop_job.side_effect = RuntimeError("forbidden")
            mock_cls.return_value = mock_client
            result = runner.invoke(
                main,
                [
                    "stop",
                    "--address",
                    "http://127.0.0.1:8265",
                    "job-1",
                ],
            )
            assert result.exit_code == 1
            assert "forbidden" in result.output


# --- embed ---


class TestEmbedCommand:
    """Tests for the 'embed' CLI command group."""

    def test_embed_list(self, runner):
        """embed list should print model names."""
        with patch("tributo.embeddings.registry.list_models") as mock_list:
            mock_list.return_value = ["bge-small-zh", "bge-base-zh"]
            result = runner.invoke(main, ["embed", "list"])
            assert result.exit_code == 0
            assert "bge-small-zh" in result.output
            assert "bge-base-zh" in result.output

    def test_embed_batch_success(self, runner):
        """embed batch should print the job ID on success."""
        with patch("tributo.embeddings.job_runner.submit_embedding_job") as mock_submit:
            mock_submit.return_value = "embed-job-1"
            result = runner.invoke(
                main,
                [
                    "embed",
                    "batch",
                    "--input",
                    "s3://bucket/in.parquet",
                    "--output",
                    "s3://bucket/out.lance",
                ],
            )
            assert result.exit_code == 0
            assert "embed-job-1" in result.output

    def test_embed_batch_failure(self, runner):
        """embed batch should print error and exit 1 on failure."""
        with patch("tributo.embeddings.job_runner.submit_embedding_job") as mock_submit:
            mock_submit.side_effect = RuntimeError("S3 error")
            result = runner.invoke(
                main,
                [
                    "embed",
                    "batch",
                    "--input",
                    "s3://bucket/in.parquet",
                    "--output",
                    "s3://bucket/out.lance",
                ],
            )
            assert result.exit_code == 1
            assert "S3 error" in result.output

    def test_embed_batch_accepts_source(self, runner):
        """embed batch should accept the canonical source option."""
        with patch("tributo.embeddings.job_runner.submit_embedding_job") as mock_submit:
            mock_submit.return_value = "embed-job-2"
            result = runner.invoke(
                main,
                [
                    "embed",
                    "batch",
                    "--source",
                    '{"provider":"tributo.parquet","uri":"s3://bucket/in.parquet"}',
                    "--output",
                    "s3://bucket/out.lance",
                ],
            )
            assert result.exit_code == 0
            assert "embed-job-2" in result.output
            assert mock_submit.call_args.kwargs["source"] == {
                "provider": "tributo.parquet",
                "uri": "s3://bucket/in.parquet",
            }

    def test_embed_batch_rejects_mixed_input_options(self, runner):
        """--source and legacy --input are mutually exclusive."""
        result = runner.invoke(
            main,
            [
                "embed",
                "batch",
                "--source",
                '{"provider":"tributo.parquet","uri":"s3://bucket/in.parquet"}',
                "--input",
                "s3://bucket/legacy.parquet",
                "--output",
                "s3://bucket/out.lance",
            ],
        )
        assert result.exit_code != 0
        assert "exactly one" in result.output or "provide exactly one" in result.output


# --- version ---


def test_version(runner):
    """--version should print the version."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.0" in result.output


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
