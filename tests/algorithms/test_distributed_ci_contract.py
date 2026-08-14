"""Guardrails that keep distributed algorithm tests out of default CI."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_PYPROJECT = _ROOT / "pyproject.toml"
_MANIFEST = _ROOT / "ci" / "test-suites.json"
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


def test_pr_unit_test_matrix_delegates_to_the_guarded_manifest_runner() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    unit = next(suite for suite in payload["suites"] if suite["id"] == "unit")

    marker_expression = unit["args"][unit["args"].index("-m") + 1]
    assert "not distributed" in marker_expression
    assert "not manual_it" in marker_expression
    assert "not quarantine" in marker_expression
    assert "scripts/ci_test_plan.py run --suite unit --prepare" in workflow
    assert "run_distributed_algorithm_it.sh" not in workflow
