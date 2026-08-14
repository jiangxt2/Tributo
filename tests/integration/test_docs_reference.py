"""Real-import integration tests for generated documentation references."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from tools.check_docs import (
    validate_api_reference,
    validate_cli_reference,
    validate_python_examples,
    validate_support_matrix,
)
from tributo.cli import main
from tributo.training.catalog import get_algorithm_catalog
from tributo.training.support_snapshot import (
    build_algorithm_support_snapshot,
    snapshot_json_objects,
)

pytestmark = pytest.mark.integration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
DOCS_BUILD_ROOT = Path(
    os.environ.get("TRIBUTO_DOCS_BUILD_ROOT", DOCS_ROOT / "_build-real")
)


def test_documented_api_targets_import_with_real_dependencies() -> None:
    errors = validate_api_reference(
        DOCS_ROOT,
        import_objects=True,
    )
    assert errors == []


def test_complete_cli_tree_imports_with_real_dependencies() -> None:
    errors = validate_cli_reference(
        DOCS_ROOT / "cli.md",
        import_cli=True,
    )
    assert errors == []


def test_documentation_python_examples_compile() -> None:
    assert validate_python_examples(DOCS_ROOT) == []


def test_support_matrix_and_cli_share_registry_snapshot() -> None:
    snapshot = build_algorithm_support_snapshot(get_algorithm_catalog().list_records())
    result = CliRunner().invoke(main, ["algo", "list", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == snapshot_json_objects(snapshot)
    assert (
        validate_support_matrix(
            DOCS_ROOT / "reference" / "support-matrix.md",
            compare_snapshot=True,
        )
        == []
    )


def test_generated_sidebar_contains_global_navigation() -> None:
    html_path = (
        DOCS_BUILD_ROOT / "html" / "getting-started" / "installation" / "index.html"
    )
    html = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'<nav id="main-sidebar".*?</nav>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    sidebar = match.group(0)

    # The dirhtml builder emits directory-style URLs, so from the
    # getting-started/installation/index.html page every root-level link is
    # two hops up.
    expected_hrefs = (
        "../../overview/",
        "../",
        "../../examples/",
        "../../data/",
        "../../algorithms/",
        "../../model-lifecycle/",
        "../../inference/",
        "../../vector-index/",
        "../../reference/",
        "../../ray-jobs/",
        "../../operations/",
        "../../developer/",
        "../../security/",
    )
    positions = [sidebar.index(f'href="{href}"') for href in expected_hrefs]
    assert positions == sorted(positions)


def test_generated_api_renders_stability_admonitions() -> None:
    html_path = DOCS_BUILD_ROOT / "html" / "reference" / "api" / "core" / "index.html"
    html = html_path.read_text(encoding="utf-8")

    assert '<div class="tributo-stability tributo-stability-stable admonition">' in html
    assert '<div class="tributo-stability tributo-stability-beta admonition">' in html
    assert '<div class="tributo-stability tributo-stability-alpha admonition">' in html
    assert ".. admonition::" not in html
