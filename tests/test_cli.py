"""Tests for CLI commands using Click's CliRunner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tributo.cli import main
from tributo.exceptions import PostPublishCallbackError


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


class TestExportCommand:
    def test_parses_explicit_hook_binding(self, runner, tmp_path):
        bundle_result = MagicMock()
        bundle_result.canonical_uri = str(tmp_path / "bundle")
        bundle_result.bundle_id = "bundle-1"
        bundle_result.manifest_uri = str(tmp_path / "bundle" / "manifest.json")
        bundle_result.status = "succeeded"
        bundle_result.alias_uri = None
        bundle_result.hook_receipts = ()

        with (
            patch("tributo.exporting.service.BundleExportService") as service_cls,
            patch(
                "tributo.integrations.sources.ray_xgboost.RayXGBoostSourceProvider"
            ) as provider_cls,
        ):
            service_cls.return_value.export_bundle.return_value = bundle_result
            provider_cls.return_value.open_source.return_value.__enter__.return_value = MagicMock()
            result = runner.invoke(
                main,
                [
                    "export",
                    "--source",
                    str(tmp_path / "checkpoint"),
                    "--targets",
                    "onnx",
                    "--output",
                    str(tmp_path / "bundle"),
                    "--hook",
                    '{"hook_id":"mlflow-log-artifacts-v1",'
                    '"required":true,"options":{"experiment_name":"demo"}}',
                ],
            )

        assert result.exit_code == 0
        config = service_cls.return_value.export_bundle.call_args.kwargs["config"]
        assert config.hooks[0].hook_id == "mlflow-log-artifacts-v1"
        assert config.hooks[0].required is True
        assert config.hooks[0].options == {"experiment_name": "demo"}

    def test_rejects_invalid_hook_json(self, runner, tmp_path):
        result = runner.invoke(
            main,
            [
                "export",
                "--source",
                str(tmp_path / "checkpoint"),
                "--targets",
                "onnx",
                "--output",
                str(tmp_path / "bundle"),
                "--hook",
                "not-json",
            ],
        )
        assert result.exit_code == 1
        assert "Hook must be a valid HookBinding JSON object" in result.output

    def test_required_hook_failure_reports_committed_bundle(self, runner, tmp_path):
        bundle_result = MagicMock()
        bundle_result.canonical_uri = str(tmp_path / "committed-bundle")
        bundle_result.manifest_uri = str(
            tmp_path / "committed-bundle" / "manifest.json"
        )
        receipt = MagicMock()
        receipt.hook_id = "mlflow-log-artifacts-v1"
        receipt.status.value = "retryable_failed"

        with (
            patch("tributo.exporting.service.BundleExportService") as service_cls,
            patch(
                "tributo.integrations.sources.ray_xgboost.RayXGBoostSourceProvider"
            ) as provider_cls,
        ):
            service_cls.return_value.export_bundle.side_effect = (
                PostPublishCallbackError(
                    "required Hook failed",
                    bundle_result=bundle_result,
                    receipts=(receipt,),
                )
            )
            provider_cls.return_value.open_source.return_value.__enter__.return_value = MagicMock()
            result = runner.invoke(
                main,
                [
                    "export",
                    "--source",
                    str(tmp_path / "checkpoint"),
                    "--targets",
                    "onnx",
                    "--output",
                    str(tmp_path / "bundle"),
                ],
            )

        assert result.exit_code == 1
        assert "Bundle committed" in result.output
        assert bundle_result.canonical_uri in result.output
        assert "retryable_failed" in result.output


# --- version ---


def test_version(runner):
    """--version should print the version."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "1.0" in result.output


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
