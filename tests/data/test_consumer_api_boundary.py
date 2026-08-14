"""Enforce the bounded-ingestion consumer and extension API boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import tributo.data as data_api
from tributo.data.engine_binding import BindingDescriptor, BindingKey, EngineBinding
from tributo.data.scan_plan import FileScan, ScanKind
from tributo.util.annotations import get_stability

_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "tributo"
_CONSUMER_PATHS = (
    _SOURCE_ROOT / "algorithms",
    _SOURCE_ROOT / "training",
    _SOURCE_ROOT / "inference",
    _SOURCE_ROOT / "data" / "graph.py",
)
_FORBIDDEN_MODULE_PREFIXES = (
    "tributo.data.binding_plugins",
    "tributo.data.bindings",
    "tributo.data.engine_binding",
    "tributo.data.provider",
    "tributo.data.provider_plugins",
    "tributo.data.provider_registry",
    "tributo.data.refs",
    "tributo.data.scan_plan",
)
_FORBIDDEN_ROOT_IMPORTS = frozenset(
    {
        "BindingCompilation",
        "BindingCompileRequest",
        "BindingDescriptor",
        "BindingKey",
        "BindingPlanConstraints",
        "DataSourceProvider",
        "DatasetHandle",
        "EngineBinding",
        "ResolvedSource",
        "ProviderDescriptor",
        "resolve_provider",
    }
)


def _consumer_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for path in _CONSUMER_PATHS:
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        else:
            files.append(path)
    return tuple(sorted(files))


def _is_forbidden_module(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _FORBIDDEN_MODULE_PREFIXES
    )


def test_algorithm_consumers_only_import_the_public_ingestion_facade() -> None:
    violations: list[str] = []
    for path in _consumer_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module(alias.name):
                        violations.append(
                            f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno} "
                            f"imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if _is_forbidden_module(node.module):
                    violations.append(
                        f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno} "
                        f"imports {node.module}"
                    )
                elif node.module == "tributo.data":
                    forbidden = sorted(
                        alias.name
                        for alias in node.names
                        if alias.name in _FORBIDDEN_ROOT_IMPORTS
                    )
                    if forbidden:
                        violations.append(
                            f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno} "
                            f"imports internal name(s) {forbidden}"
                        )

    assert violations == []


def test_binding_contracts_are_not_exported_from_consumer_root() -> None:
    for name in (
        "BindingCompilation",
        "BindingCompileRequest",
        "BindingDescriptor",
        "BindingKey",
        "BindingPlanConstraints",
        "EngineBinding",
        "ProviderDescriptor",
    ):
        assert name not in data_api.__all__
        assert not hasattr(data_api, name)


def test_binding_and_scan_contracts_are_developer_api() -> None:
    for contract in (BindingDescriptor, BindingKey, EngineBinding, FileScan, ScanKind):
        assert get_stability(contract) == "developer"
