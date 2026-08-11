"""Contracts for the repository PR precheck script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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
