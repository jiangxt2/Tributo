"""Unit tests for documentation-specific tooling."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click

from tools.check_docs import (
    autodoc_targets,
    click_directive,
    command_paths,
    resolve_target,
    root_toctree_targets,
    validate_mock_inventory,
    validate_navigation,
    validate_python_examples,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "docs" / "_ext"))
stability = importlib.import_module("stability")


def test_autodoc_targets_preserve_source_order() -> None:
    text = """
```{autoclass} tributo.JobConfig
```
```{autofunction} tributo.exporting.export
```
"""
    assert autodoc_targets(text) == (
        "tributo.JobConfig",
        "tributo.exporting.export",
    )


def test_click_directive_requires_explicit_full_tree_options() -> None:
    text = """
```{click} tributo.cli:main
:prog: tributo
:nested: full
```
"""
    assert click_directive(text) == (
        "tributo.cli:main",
        {"prog": "tributo", "nested": "full"},
    )


def test_root_toctree_targets_preserve_titled_entry_order() -> None:
    text = """
```{toctree}
:hidden:
:maxdepth: 4

Overview <overview/index>
Getting Started <quickstart>
Installation <installation>
```
"""
    assert root_toctree_targets(text) == (
        "overview/index",
        "quickstart",
        "installation",
    )


def test_repository_navigation_matches_ray_style_contract() -> None:
    assert validate_navigation(REPOSITORY_ROOT / "docs") == []


def test_resolve_target_imports_longest_module_prefix() -> None:
    assert resolve_target("pathlib.Path") is Path


def test_mock_inventory_is_sorted_and_third_party_only() -> None:
    assert validate_mock_inventory() == []


def test_stability_extension_handles_every_contract_level() -> None:
    @PublicAPI(stability="stable")
    class StableAPI:
        pass

    @PublicAPI(stability="beta")
    def beta_api() -> None:
        pass

    @PublicAPI(stability="alpha")
    def alpha_api() -> None:
        pass

    @DeveloperAPI
    def developer_api() -> None:
        pass

    def unannotated() -> None:
        pass

    assert stability.get_object_stability(StableAPI) == "stable"
    assert stability.get_object_stability(beta_api) == "beta"
    assert stability.get_object_stability(alpha_api) == "alpha"
    assert stability.get_object_stability(developer_api) == "developer"
    assert stability.get_object_stability(unannotated) == "developer"


def test_stability_extension_only_marks_top_level_objects() -> None:
    @PublicAPI(stability="stable")
    class StableAPI:
        pass

    lines = ["Existing documentation."]
    stability.add_stability_to_docstring(
        None, "class", "example.StableAPI", StableAPI, None, lines
    )
    assert ".. admonition:: Stable API" in lines

    method_lines = ["Method documentation."]
    stability.add_stability_to_docstring(
        None, "method", "example.StableAPI.method", object(), None, method_lines
    )
    assert method_lines == ["Method documentation."]


def test_command_paths_walk_nested_click_groups() -> None:
    @click.group()
    def cli() -> None:
        """Test command."""

    @cli.group()
    def serve() -> None:
        """Serve commands."""

    @serve.command()
    def start() -> None:
        """Start serving."""

    assert command_paths(cli) == ("tributo serve", "tributo serve start")


def test_python_example_validation_supports_explicit_pseudocode(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "valid.md").write_text(
        "```python\nvalue = 1\n```\n",
        encoding="utf-8",
    )
    (docs_root / "pseudocode.md").write_text(
        "<!-- docs-check: skip-python -->\n"
        "```python\nProvider.open(Source) -> Handle\n```\n",
        encoding="utf-8",
    )
    assert validate_python_examples(docs_root) == []
