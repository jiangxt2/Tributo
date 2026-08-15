"""Static boundaries for internal bounded-ingestion consumers."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "tributo"

_LEGACY_COMPATIBILITY_FILES = {
    SRC_ROOT / "data" / "base.py",
}


def _internal_python_files() -> list[Path]:
    """Return production modules outside the exact legacy compatibility files."""
    return [
        path
        for path in SRC_ROOT.rglob("*.py")
        if path not in _LEGACY_COMPATIBILITY_FILES
    ]


def test_internal_consumers_do_not_import_legacy_read_apis() -> None:
    """Keep old Connector and Ray compatibility APIs outside business consumers."""
    forbidden_modules = {"tributo.data._compat_read", "tributo.data.registry"}
    forbidden_names = {
        "DataConnector",
        "get_connector",
        "list_connectors",
        "register_connector",
        "open_ray_compat",
    }
    violations: list[str] = []
    for path in _internal_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                if imported & forbidden_modules:
                    violations.append(
                        f"{path}: import {sorted(imported & forbidden_modules)}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}
                if module in forbidden_modules:
                    violations.append(f"{path}: from {module} import ...")
                elif imported_names & forbidden_names:
                    violations.append(
                        f"{path}: import {sorted(imported_names & forbidden_names)}"
                    )

    assert not violations, "legacy ingestion imports found:\n" + "\n".join(violations)
