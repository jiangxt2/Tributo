"""Guardrails that keep distributed algorithm tests out of default CI."""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_PYPROJECT = _ROOT / "pyproject.toml"
_DISTRIBUTED_RUNTIME_TEST = (
    _ROOT / "tests" / "algorithms" / "test_distributed_ray_runtime.py"
)


def test_distributed_runtime_tests_are_explicitly_marked() -> None:
    source = _DISTRIBUTED_RUNTIME_TEST.read_text(encoding="utf-8")

    assert "pytest.mark.distributed" in source


def test_default_pytest_selection_excludes_distributed_tests() -> None:
    config = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]

    assert "not distributed" in addopts


def test_pr_unit_test_matrix_excludes_distributed_tests() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    unit_job = workflow.split("\n  unit-tests:", maxsplit=1)[1].split(
        "\n  # Domain gate",
        maxsplit=1,
    )[0]

    assert (
        '-m "not integration and not slow and not distributed '
        'and not minio_compat and not ray_runtime_env"'
    ) in unit_job
