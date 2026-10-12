"""Root-package import and lazy Ray Jobs boundary tests."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path


def test_root_import_does_not_eagerly_import_ray_job_submission() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, tributo; "
                "assert 'ray.job_submission' not in sys.modules; "
                "assert 'TributoClient' in dir(tributo)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_lazy_root_client_preserves_public_identity() -> None:
    import tributo
    from tributo.job import RayJob, TributoClient

    assert tributo.TributoClient is TributoClient
    assert tributo.RayJob is RayJob


def test_installed_root_version_matches_project_metadata() -> None:
    tributo = importlib.import_module("tributo")

    root = Path(__file__).resolve().parents[1]
    expected = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]

    assert tributo.__version__ == importlib.metadata.version("tributo") == expected


def test_source_checkout_version_fallback_matches_project_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    script = """
import importlib.metadata
original_version = importlib.metadata.version
def without_core_metadata(name):
    if name == "tributo":
        raise importlib.metadata.PackageNotFoundError(name)
    return original_version(name)
importlib.metadata.version = without_core_metadata
import tributo
print(tributo.__version__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected
