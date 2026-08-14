"""Contracts for the repository PR precheck script."""

from __future__ import annotations

import importlib.util
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
PRECHECK_PATH = ROOT / "scripts" / "pr-precheck.py"
COMPONENT_VERSIONS = ROOT / "tests" / "integrations" / "component-versions.env"
COMPONENT_VERSIONS_RELATIVE = "tests/integrations/component-versions.env"


def _load_precheck() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tributo_pr_precheck", PRECHECK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_component_contract(root: Path, content: str) -> None:
    destination = root / COMPONENT_VERSIONS_RELATIVE
    destination.parent.mkdir(parents=True)
    destination.write_text(content)


def test_changed_test_filter_matches_the_default_ci_suite() -> None:
    precheck = _load_precheck()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    addopts = project["tool"]["pytest"]["ini_options"]["addopts"]

    assert addopts == f"-m '{precheck.CHANGED_TEST_MARKER_FILTER}'"


def test_public_component_version_contract_is_allowed(tmp_path: Path) -> None:
    precheck = _load_precheck()
    _write_component_contract(tmp_path, COMPONENT_VERSIONS.read_text())

    assert precheck.check_hygiene(str(tmp_path), [COMPONENT_VERSIONS_RELATIVE]) == []


def test_component_contract_with_unexpected_key_is_rejected(tmp_path: Path) -> None:
    precheck = _load_precheck()
    content = COMPONENT_VERSIONS.read_text() + "\nAWS_SECRET_ACCESS_KEY=not-a-secret\n"
    _write_component_contract(tmp_path, content)

    issues = precheck.check_hygiene(str(tmp_path), [COMPONENT_VERSIONS_RELATIVE])

    assert any("Suspicious file" in issue for issue in issues)


def test_unrelated_env_file_remains_rejected(tmp_path: Path) -> None:
    precheck = _load_precheck()
    relative_path = "tests/integrations/local.env"
    destination = tmp_path / relative_path
    destination.parent.mkdir(parents=True)
    destination.write_text("PUBLIC_VERSION=1.0\n")

    issues = precheck.check_hygiene(str(tmp_path), [relative_path])

    assert any("Suspicious file" in issue for issue in issues)


@pytest.mark.parametrize(
    "path",
    [
        "docs/index.md",
        "src/tributo/training/dnn_trainer.py",
        "tests/docs/test_public_api.py",
        "requirements-doc.lock",
        "tools/generate_algorithm_support_matrix.py",
    ],
)
def test_docs_ci_affected_matches_workflow_paths(path: str) -> None:
    precheck = _load_precheck()

    assert precheck.docs_ci_affected([path]) is True


def test_docs_ci_ignores_unrelated_paths() -> None:
    precheck = _load_precheck()

    assert precheck.docs_ci_affected(["tests/test_cli.py"]) is False


def test_docs_environment_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    precheck = _load_precheck()
    commands: list[list[str]] = []

    def fail_create(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Python 3.12 is not available in the offline cache",
        )

    monkeypatch.setattr(precheck.subprocess, "run", fail_create)

    issues = precheck.prepare_docs_environment(str(tmp_path), allow_network=False)

    assert issues[0].startswith("FAIL: docs environment creation failed")
    assert "--allow-network" in issues[0]
    assert commands == [
        [
            "uv",
            "venv",
            ".docs-venv",
            "--python",
            "3.12",
            "--offline",
        ]
    ]


def test_docs_ci_runs_ci_commands_and_reports_spelling_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    precheck = _load_precheck()
    bin_dir = tmp_path / ".docs-venv" / "bin"
    bin_dir.mkdir(parents=True)
    docs_python = bin_dir / "python"
    sphinx_build = bin_dir / "sphinx-build"
    docs_python.touch()
    sphinx_build.touch()
    commands: list[list[str]] = []

    def run_command(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:2] == ["make", "spelling"]:
            return subprocess.CompletedProcess(
                command,
                2,
                stdout="Spell check: MapReduce\nFound 1 misspelled words",
                stderr="make: spelling failed",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(precheck.subprocess, "run", run_command)

    issues = precheck.check_docs_ci(
        str(tmp_path),
        ["docs/index.md"],
        allow_network=False,
    )

    assert issues[0].startswith("FAIL: documentation spelling failed")
    assert "MapReduce" in issues[0]
    assert "make: spelling failed" in issues[0]
    assert commands == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(docs_python),
            "--requirement",
            "requirements-doc.lock",
            "--offline",
        ],
        [str(docs_python), "tools/check_docs.py", "--static-only"],
        ["make", "strict", f"SPHINXBUILD={sphinx_build}"],
        ["make", "spelling", f"SPHINXBUILD={sphinx_build}"],
    ]
