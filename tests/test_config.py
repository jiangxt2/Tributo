"""Tests for job configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from tributo.algorithms import AlgorithmArtifact, ImageProfile
from tributo.config import JobConfig


def test_job_config_basic():
    """Test basic job configuration."""
    config = JobConfig(entrypoint="python script.py")
    assert config.entrypoint == "python script.py"
    assert config.runtime_env == {}
    assert config.num_cpus is None
    assert config.num_gpus is None
    assert config.project_root is None


def test_job_config_accepts_project_root(tmp_path: Path) -> None:
    config = JobConfig(entrypoint="python script.py", project_root=tmp_path)
    assert config.project_root == tmp_path


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


def test_job_config_accepts_artifact_and_image_profile() -> None:
    config = JobConfig(
        entrypoint="python train.py",
        algorithm_artifact=AlgorithmArtifact(source="algorithm.whl"),
        image_profile=ImageProfile(
            profile_id="cpu.test",
            image_uri="tributo:test",
            image_digest="a" * 64,
        ),
        declared_dependencies=("scikit-learn>=1.6,<2",),
    )

    assert config.algorithm_artifact is not None
    assert config.image_profile is not None
    assert config.declared_dependencies == ("scikit-learn<2,>=1.6",)


def test_job_config_rejects_artifact_without_profile() -> None:
    with pytest.raises(ValueError, match="requires image_profile"):
        JobConfig(
            entrypoint="python train.py",
            algorithm_artifact=AlgorithmArtifact(source="algorithm.whl"),
        )


def test_job_config_rejects_runtime_env_override_of_artifact_fields() -> None:
    with pytest.raises(ValueError, match="artifact-owned fields"):
        JobConfig(
            entrypoint="python train.py",
            runtime_env={"pip": {"packages": ["numpy"]}},
            algorithm_artifact=AlgorithmArtifact(source="algorithm.whl"),
            image_profile=ImageProfile(
                profile_id="cpu.test",
                image_uri="tributo:test",
                image_digest="a" * 64,
            ),
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
