"""Keep Tributo Core independent of the aggregate algorithm distribution."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_ROOT = Path(__file__).resolve().parents[2]
_ALGORITHM_DISTRIBUTION = canonicalize_name("tributo-algorithms")


def test_core_metadata_does_not_depend_on_aggregate_algorithm_distribution() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    assert all(
        canonicalize_name(Requirement(requirement).name) != _ALGORITHM_DISTRIBUTION
        for requirement in requirements
    )

    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert all(
        canonicalize_name(package["name"]) != _ALGORITHM_DISTRIBUTION
        for package in lock["package"]
    )


def test_core_ci_suites_use_core_owned_contract_fixtures() -> None:
    manifest = json.loads((_ROOT / "ci/test-suites.json").read_text(encoding="utf-8"))
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((_ROOT / ".github/workflows").glob("*.y*ml"))
    )
    suite_runtime = json.dumps(
        [
            {
                key: suite[key]
                for key in ("entrypoint", "args", "extras", "test_paths")
                if key in suite
            }
            for suite in manifest["suites"]
        ]
    )

    assert "TRIBUTO_ALGORITHMS_ROOT" not in suite_runtime
    assert "tributo-algorithms" not in suite_runtime
    assert "TRIBUTO_ALGORITHMS_ROOT" not in workflows
    assert "tributo-algorithms" not in workflows
    assert not any(
        suite["id"] in {"distributed-algorithm-cluster", "kuberay-rayjob"}
        for suite in manifest["suites"]
    )
    assert (_ROOT / "tests/algorithms/test_conformance_cli.py").is_file()
    assert (
        _ROOT / "tests/fixtures/distributed_algorithm_plugin/pyproject.toml"
    ).is_file()
    for path in (
        _ROOT / "tests/integrations/runtime-profiles.json",
        _ROOT / "tools/tributo-runtime-full.json",
        _ROOT / "docker/tributo-runtime/Dockerfile",
    ):
        assert "tributo-algorithms" not in path.read_text(encoding="utf-8")
