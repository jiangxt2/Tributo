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
    validate_support_matrix,
    validate_system_landscape_svg,
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


def test_repository_system_landscape_svg_matches_contract() -> None:
    path = REPOSITORY_ROOT / "docs" / "images" / "tributo-system-landscape.svg"
    assert validate_system_landscape_svg(path) == []


def test_system_landscape_svg_validation_rejects_unsafe_and_inaccessible_svg(
    tmp_path: Path,
) -> None:
    path = tmp_path / "landscape.svg"
    path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"
  role="img" aria-labelledby="title">
  <title id="title">\u67b6\u6784</title>
  <desc id="description">Description</desc>
  <style>
    @import "https://example.com/theme.css";
    .unsafe { fill: url(https://example.com/fill.svg); }
    @media (max-width: 9px) { .narrow { display: block; } }
  </style>
  <g class="desktop"><rect class="boundary"/></g>
  <g class="narrow"/>
  <foreignObject/>
  <script>javascript:alert(1)</script>
  <image href="https://example.com/external.png"/>
</svg>
""",
        encoding="utf-8",
    )

    errors = validate_system_landscape_svg(path)

    assert any("CJK characters" in error for error in errors)
    assert any("CSS @import" in error for error in errors)
    assert any("SVG CSS resource" in error for error in errors)
    assert any("desc id must appear in aria-labelledby" in error for error in errors)
    assert any("forbidden SVG element foreignObject" in error for error in errors)
    assert any("forbidden SVG element script" in error for error in errors)
    assert any("internal fragments" in error for error in errors)
    assert any(
        "narrow layout must show the framework boundary" in error for error in errors
    )
    assert any("forced-colors theme" in error for error in errors)


def test_system_landscape_svg_validation_rejects_malformed_xml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "landscape.svg"
    path.write_text("<svg>", encoding="utf-8")

    assert any(
        "malformed SVG XML" in error for error in validate_system_landscape_svg(path)
    )


def test_system_landscape_svg_validation_accepts_nested_media_query_variants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "landscape.svg"
    path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"
  role="img" aria-labelledby="title description">
  <title id="title">Landscape</title>
  <desc id="description">Description</desc>
  <defs>
    <style>
      @MeDiA(max-width : 9px) { .narrow { display: block; } }
      @MEDIA ( forced-colors : ACTIVE ) { text { fill: CanvasText; } }
    </style>
  </defs>
  <g class="desktop"><rect class="boundary"/></g>
  <g class="narrow"><rect class="boundary"/></g>
</svg>
""",
        encoding="utf-8",
    )

    assert validate_system_landscape_svg(path) == []


def test_system_landscape_svg_validation_ignores_media_queries_in_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "landscape.svg"
    path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"
  role="img" aria-labelledby="title description">
  <title id="title">Landscape</title>
  <desc id="description">Description</desc>
  <style>
    /* @media (max-width: 9px) and @media (forced-colors: active) */
  </style>
  <g class="desktop"><rect class="boundary"/></g>
  <g class="narrow"><rect class="boundary"/></g>
</svg>
""",
        encoding="utf-8",
    )

    errors = validate_system_landscape_svg(path)

    assert any("narrow-screen media query" in error for error in errors)
    assert any("forced-colors theme" in error for error in errors)


def test_static_support_matrix_validation_rejects_duplicate_markers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support-matrix.md"
    path.write_text(
        "<!-- BEGIN GENERATED: TRIBUTO ALGORITHM SUPPORT -->\n"
        "<!-- BEGIN GENERATED: TRIBUTO ALGORITHM SUPPORT -->\n"
        "<!-- END GENERATED: TRIBUTO ALGORITHM SUPPORT -->\n",
        encoding="utf-8",
    )

    errors = validate_support_matrix(path, compare_snapshot=False)

    assert len(errors) == 1
    assert "found begin=2, end=1" in errors[0]
