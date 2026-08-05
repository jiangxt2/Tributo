"""Real-import integration tests for generated documentation references."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.check_docs import (
    validate_api_reference,
    validate_cli_reference,
    validate_python_examples,
)

pytestmark = pytest.mark.integration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"


def test_documented_api_targets_import_with_real_dependencies() -> None:
    errors = validate_api_reference(
        DOCS_ROOT / "api.md",
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


def test_generated_sidebar_contains_global_navigation() -> None:
    html_path = DOCS_ROOT / "_build-real" / "html" / "installation" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'<nav id="main-sidebar".*?</nav>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    sidebar = match.group(0)

    # The dirhtml builder emits directory-style URLs, so from the
    # installation/index.html page every root-level link is one hop up.
    expected_hrefs = (
        "../overview/",
        "../quickstart/",
        "../user-guide/",
        "../examples/",
        "../integrations/",
        "../data/",
        "../training/",
        "../model-lifecycle/",
        "../inference/",
        "../embeddings/",
        "../reference/",
        "../ray-jobs/",
        "../operations/",
        "../developer/",
        "../architecture/",
        "../security/",
    )
    positions = [sidebar.index(f'href="{href}"') for href in expected_hrefs]
    assert positions == sorted(positions)
