"""Sphinx configuration for the Tributo documentation."""

from __future__ import annotations

import importlib
import os
import sys
import tomllib
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

DOC_MOCK_IMPORTS = importlib.import_module("mock_imports").DOC_MOCK_IMPORTS

project_metadata = tomllib.loads(
    (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]

project = "Tributo"
author = "Tributo Contributors"
copyright = "2026, Tributo Contributors"
release = str(project_metadata["version"])
version = release
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_click",
    "sphinxcontrib.spelling",
    "stability",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", ".DS_Store", "Thumbs.db"]
templates_path = ["_templates"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_preserve_defaults = True
autoclass_content = "class"

# The RTD and lightweight CI environments import Tributo source code with
# third-party runtime packages mocked. The real-import CI job disables all
# mocks and repeats the strict build in the locked project environment.
use_real_imports = os.environ.get("SPHINX_REAL_IMPORTS") == "1"
autodoc_mock_imports = [] if use_real_imports else list(DOC_MOCK_IMPORTS)
sphinx_click_mock_imports = autodoc_mock_imports

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "ray": ("https://docs.ray.io/en/latest", None),
}
intersphinx_timeout = 10

nitpicky = True
nitpick_ignore_regex = [
    (
        "py:class",
        r"(?:numpy|pandas|pyarrow|ray|torch|xgboost)(?:\..*)?",
    ),
    (
        "py:obj",
        r"(?:numpy|pandas|pyarrow|ray|torch|xgboost)(?:\..*)?",
    ),
]

html_theme = "pydata_sphinx_theme"
html_title = f"Tributo {version}"
html_logo = "images/tributo-logo.svg"
html_favicon = "images/tributo-favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jiangxt2/tributo",
            "icon": "fa-brands fa-github",
        }
    ],
}
html_sidebars = {
    "**": ["main-sidebar"],
}
html_context = {
    "github_user": "jiangxt2",
    "github_repo": "tributo",
    "github_version": "master",
    "doc_path": "docs",
}

linkcheck_anchors = False
linkcheck_retries = 2
linkcheck_timeout = 10
linkcheck_ignore = [
    r"http://127\.0\.0\.1(?::\d+)?(?:/.*)?",
    r"http://localhost(?::\d+)?(?:/.*)?",
]

spelling_lang = "en_US"
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_show_suggestions = False
