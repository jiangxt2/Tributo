"""Unit tests for documentation-specific tooling."""

from __future__ import annotations

import ast
import importlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import click
import pytest

import tools.generate_public_api_reference as public_api_generator
from tools.check_docs import (
    autodoc_targets,
    click_directive,
    command_paths,
    resolve_target,
    root_toctree_targets,
    validate_api_reference,
    validate_doc_code,
    validate_mock_inventory,
    validate_navigation,
    validate_publication_metadata,
    validate_python_examples,
    validate_ray_writing_style,
    validate_repository_examples,
    validate_support_matrix,
    validate_system_landscape_svg,
)
from tools.generate_public_api_reference import (
    build_inventory,
    check_pages,
    component_for,
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


def test_repository_doc_code_compiles() -> None:
    assert validate_doc_code(REPOSITORY_ROOT / "docs") == []


def test_repository_cluster_examples_compile() -> None:
    assert validate_repository_examples(REPOSITORY_ROOT) == []


def test_repository_headings_match_ray_style_contract() -> None:
    assert validate_ray_writing_style(REPOSITORY_ROOT / "docs") == []


def test_generated_public_api_reference_covers_source_inventory() -> None:
    inventory = build_inventory()

    assert inventory
    assert len({symbol.target for symbol in inventory}) == len(inventory)
    assert {component_for(symbol) for symbol in inventory} == {
        "algorithms-training",
        "core",
        "data",
        "vector-index",
        "extensions",
        "inference-serving",
        "model-lifecycle",
    }
    pipeline_symbols = tuple(
        symbol for symbol in inventory if symbol.module.startswith("tributo.pipeline.")
    )
    assert pipeline_symbols
    assert {component_for(symbol) for symbol in pipeline_symbols} == {"extensions"}
    assert all(
        page.endswith("\n") and not page.endswith("\n\n")
        for page in public_api_generator.expected_pages(inventory).values()
    )
    assert check_pages(inventory) == []


def test_public_api_inventory_classifies_exceptions_by_base_class() -> None:
    tree = ast.parse(
        """class BrokerError:
    pass

class ActualError(Exception):
    pass
"""
    )
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    assert [public_api_generator._is_exception_class(node) for node in classes] == [
        False,
        True,
    ]


def test_api_reference_validation_reuses_source_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = public_api_generator.build_inventory
    calls = 0

    def counting_build_inventory(
        source_root: Path = public_api_generator.SOURCE_ROOT,
    ) -> tuple[public_api_generator.PublicSymbol, ...]:
        nonlocal calls
        calls += 1
        return original(source_root)

    monkeypatch.setattr(
        public_api_generator,
        "build_inventory",
        counting_build_inventory,
    )

    assert validate_api_reference(REPOSITORY_ROOT / "docs", import_objects=False) == []
    assert calls == 1


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
    assert ":::{admonition} Stable API" in lines
    assert ":class: tributo-stability tributo-stability-stable" in lines

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


def test_repository_example_validation_rejects_invalid_python(
    tmp_path: Path,
) -> None:
    examples_root = tmp_path / "examples"
    examples_root.mkdir()
    (examples_root / "invalid.py").write_text(
        "if True print('invalid')\n",
        encoding="utf-8",
    )

    errors = validate_repository_examples(tmp_path)

    assert len(errors) == 1
    assert "invalid repository example" in errors[0]


def test_publication_metadata_matches_project_documentation_url(
    tmp_path: Path,
) -> None:
    documentation_url = "https://docs.example.test/en/latest/"
    docs_root = tmp_path / "docs"
    static_root = docs_root / "_static"
    static_root.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "example"\n'
        "[project.urls]\n"
        f'Documentation = "{documentation_url}"\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        f"[Docs]({documentation_url})\n"
        f"[Quickstart]({documentation_url}getting-started/quickstart/)\n"
        f"[Support]({documentation_url}reference/support-matrix/)\n",
        encoding="utf-8",
    )
    versions_path = static_root / "versions.json"
    versions_path.write_text(
        json.dumps([{"name": "latest", "version": "latest", "url": documentation_url}]),
        encoding="utf-8",
    )

    assert validate_publication_metadata(docs_root) == []

    versions_path.write_text(
        json.dumps(
            [
                {
                    "name": "latest",
                    "version": "latest",
                    "url": "https://stale.example.test/",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert any(
        "does not match project Documentation URL" in error
        for error in validate_publication_metadata(docs_root)
    )


def _load_docs_conf(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> dict[str, Any]:
    for name in (
        "GITHUB_EVENT_NAME",
        "GITHUB_REF_NAME",
        "READTHEDOCS_GIT_IDENTIFIER",
        "READTHEDOCS_VERSION_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return runpy.run_path(str(REPOSITORY_ROOT / "docs" / "conf.py"))


def test_docs_conf_uses_editable_branch_and_canonical_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_docs_conf(
        monkeypatch,
        {
            "READTHEDOCS_GIT_IDENTIFIER": "docs/example",
            "READTHEDOCS_VERSION_TYPE": "branch",
        },
    )

    documentation_url = config["documentation_url"]
    assert config["html_baseurl"] == documentation_url
    assert config["html_context"]["github_version"] == "docs/example"
    assert config["html_theme_options"]["use_edit_page_button"] is True
    assert config["html_theme_options"]["switcher"]["json_url"] == (
        f"{documentation_url}_static/versions.json"
    )


@pytest.mark.parametrize(
    "environment",
    [
        {
            "READTHEDOCS_GIT_IDENTIFIER": "123",
            "READTHEDOCS_VERSION_TYPE": "external",
        },
        {
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_REF_NAME": "feature/pr-preview",
        },
    ],
)
def test_docs_conf_disables_edit_button_for_pull_request_previews(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    config = _load_docs_conf(monkeypatch, environment)

    assert config["html_theme_options"]["use_edit_page_button"] is False


def test_ray_style_validation_ignores_code_and_rejects_numbered_title_case(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    path = docs_root / "guide.md"
    path.write_text(
        "# API Reference\n\n## 1. Setup\n\n```python\n# 2. Code Comment\n```\n",
        encoding="utf-8",
    )

    errors = validate_ray_writing_style(docs_root)

    assert any("sentence case" in error for error in errors)
    assert any("numeric prefix" in error for error in errors)
    assert all("Code Comment" not in error for error in errors)


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
