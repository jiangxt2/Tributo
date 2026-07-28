"""Tests for job configuration."""

from __future__ import annotations

import pytest

from tributo.config import JobConfig


def test_job_config_basic():
    """Test basic job configuration."""
    config = JobConfig(entrypoint="python script.py")
    assert config.entrypoint == "python script.py"
    assert config.runtime_env == {}
    assert config.num_cpus is None
    assert config.num_gpus is None


def test_job_config_with_resources():
    """Test job configuration with resource specifications."""
    config = JobConfig(
        entrypoint="python script.py",
        num_cpus=4.0,
        num_gpus=1.0,
        memory=1024 * 1024 * 1024,
    )
    assert config.num_cpus == 4.0
    assert config.num_gpus == 1.0
    assert config.memory == 1024 * 1024 * 1024


def test_job_config_empty_entrypoint():
    """Test that empty entrypoint raises validation error."""
    with pytest.raises(ValueError, match="Entrypoint cannot be empty"):
        JobConfig(entrypoint="")


def test_job_config_negative_resources():
    """Test that negative resources raise validation error."""
    with pytest.raises(ValueError):
        JobConfig(entrypoint="python script.py", num_cpus=-1.0)


@pytest.mark.parametrize(
    "entrypoint,expected",
    [
        ("python script.py", "python script.py"),
        ("  python script.py  ", "python script.py"),
    ],
)
def test_job_config_entrypoint_strip(entrypoint: str, expected: str):
    """Test that entrypoint is stripped of whitespace."""
    config = JobConfig(entrypoint=entrypoint)
    assert config.entrypoint == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
