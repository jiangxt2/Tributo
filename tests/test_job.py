"""Tests for TributoClient and RayJob with mocked JobSubmissionClient."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tributo.algorithms import AlgorithmArtifact, ImageProfile
from tributo.config import JobConfig
from tributo.exceptions import JobExecutionError, JobSubmissionError
from tributo.job import RayJob, TributoClient


@pytest.fixture
def mock_client():
    """Provide a mocked JobSubmissionClient."""
    with patch("tributo.job.JobSubmissionClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


# ── TributoClient ──


class TestTributoClient:
    """Tests for TributoClient."""

    @pytest.fixture
    def client(self, mock_ray_address, mock_client):
        return TributoClient(mock_ray_address)

    def test_submit_returns_job_id(self, client, mock_client):
        mock_client.submit_job.return_value = "job-abc123"
        result = client.submit(entrypoint="python script.py")
        assert result == "job-abc123"

    def test_submit_passes_args_to_client(self, client, mock_client):
        mock_client.submit_job.return_value = "job-1"
        client.submit(
            entrypoint="python train.py",
            runtime_env={"pip": ["numpy"]},
            metadata={"team": "ml"},
            submission_id="sub-1",
            entrypoint_num_cpus=4.0,
            entrypoint_num_gpus=1.0,
            entrypoint_memory=1024,
        )
        mock_client.submit_job.assert_called_once_with(
            entrypoint="python train.py",
            runtime_env={"pip": ["numpy"]},
            metadata={"team": "ml"},
            submission_id="sub-1",
            entrypoint_num_cpus=4.0,
            entrypoint_num_gpus=1.0,
            entrypoint_memory=1024,
        )

    def test_submit_builds_artifact_runtime_environment(self, client, mock_client):
        mock_client.submit_job.return_value = "job-artifact"
        artifact = AlgorithmArtifact(source="algorithm.whl")
        profile = ImageProfile(
            profile_id="cpu.test",
            image_uri="tributo:test",
            image_digest="a" * 64,
        )
        with patch(
            "tributo.job.build_runtime_env",
            return_value={"working_dir": "/tmp/bundle", "env_vars": {}},
        ) as build_runtime_env:
            result = client.submit(
                entrypoint="python train.py",
                algorithm_artifact=artifact,
                image_profile=profile,
                declared_dependencies=("numpy>=2",),
            )

        assert result == "job-artifact"
        assert build_runtime_env.call_args.kwargs["algorithm_artifact"] is artifact
        assert build_runtime_env.call_args.kwargs["image_profile"] is profile
        assert build_runtime_env.call_args.kwargs["declared_dependencies"] == (
            "numpy>=2",
        )

    def test_submit_rejects_distribution_fields_without_artifact(
        self, client, mock_client
    ):
        profile = ImageProfile(
            profile_id="cpu.test",
            image_uri="tributo:test",
            image_digest="a" * 64,
        )

        with pytest.raises(JobSubmissionError, match="require algorithm_artifact"):
            client.submit(
                entrypoint="python train.py",
                image_profile=profile,
            )

        mock_client.submit_job.assert_not_called()

    def test_submit_rejects_artifact_without_profile(self, client, mock_client):
        with pytest.raises(JobSubmissionError, match="requires image_profile"):
            client.submit(
                entrypoint="python train.py",
                algorithm_artifact=AlgorithmArtifact(source="algorithm.whl"),
            )

        mock_client.submit_job.assert_not_called()

    def test_submit_raises_on_failure(self, client, mock_client):
        mock_client.submit_job.side_effect = RuntimeError("connection refused")
        with pytest.raises(JobSubmissionError, match="connection refused"):
            client.submit(entrypoint="python script.py")

    def test_get_status_returns_value(self, client, mock_client):
        mock_status = MagicMock()
        mock_status.value = "RUNNING"
        mock_client.get_job_status.return_value = mock_status
        assert client.get_status("job-1") == "RUNNING"

    def test_get_status_raises_on_failure(self, client, mock_client):
        mock_client.get_job_status.side_effect = RuntimeError("not found")
        with pytest.raises(JobExecutionError, match="not found"):
            client.get_status("job-1")

    def test_get_logs_returns_string(self, client, mock_client):
        mock_client.get_job_logs.return_value = "hello\nworld\n"
        assert client.get_logs("job-1") == "hello\nworld\n"

    def test_get_logs_raises_on_failure(self, client, mock_client):
        mock_client.get_job_logs.side_effect = RuntimeError("timeout")
        with pytest.raises(JobExecutionError, match="timeout"):
            client.get_logs("job-1")

    def test_stop_job_returns_true(self, client, mock_client):
        mock_client.stop_job.return_value = True
        assert client.stop_job("job-1") is True

    def test_stop_job_returns_false(self, client, mock_client):
        mock_client.stop_job.return_value = False
        assert client.stop_job("job-1") is False

    def test_stop_job_raises_on_failure(self, client, mock_client):
        mock_client.stop_job.side_effect = RuntimeError("forbidden")
        with pytest.raises(JobExecutionError, match="forbidden"):
            client.stop_job("job-1")


# ── TributoClient lazy init ──


def test_client_is_lazily_initialized(mock_ray_address, mock_client):
    """The underlying JobSubmissionClient is created on first access."""
    with patch("tributo.job.JobSubmissionClient") as mock_cls:
        client = TributoClient(mock_ray_address)
        mock_cls.assert_not_called()
        _ = client._get_client()
        mock_cls.assert_called_once_with(mock_ray_address)


# ── RayJob backward compatibility ──


class TestRayJob:
    """Tests for the RayJob backward-compatible wrapper."""

    @pytest.fixture
    def job(self, mock_ray_address, mock_client):
        config = JobConfig(entrypoint="python script.py")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return RayJob(address=mock_ray_address, config=config)

    def test_submit_delegates_to_client(self, job, mock_client):
        mock_client.submit_job.return_value = "job-1"
        result = job.submit()
        assert result == "job-1"
        mock_client.submit_job.assert_called_once()

    def test_submit_passes_config_fields(self, job, mock_client):
        mock_client.submit_job.return_value = "job-1"
        job.submit()
        call_kwargs = mock_client.submit_job.call_args[1]
        assert call_kwargs["entrypoint"] == "python script.py"
        assert call_kwargs["runtime_env"] == {}
        assert call_kwargs["metadata"] == {}

    def test_submit_passes_project_root_to_client(self, job):
        job.config.project_root = Path("/workspace/project")
        job._client.submit = MagicMock(return_value="job-1")

        assert job.submit() == "job-1"
        assert job._client.submit.call_args.kwargs["project_root"] == Path(
            "/workspace/project"
        )

    def test_get_status_delegates(self, job, mock_client):
        mock_status = MagicMock()
        mock_status.value = "SUCCEEDED"
        mock_client.get_job_status.return_value = mock_status
        assert job.get_status("job-1") == "SUCCEEDED"

    def test_get_logs_delegates(self, job, mock_client):
        mock_client.get_job_logs.return_value = "output"
        assert job.get_logs("job-1") == "output"

    def test_stop_job_delegates(self, job, mock_client):
        mock_client.stop_job.return_value = True
        assert job.stop_job("job-1") is True

    def test_emits_deprecation_warning(self, mock_ray_address, mock_client):
        config = JobConfig(entrypoint="python script.py")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RayJob(address=mock_ray_address, config=config)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TributoClient" in str(w[0].message)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
