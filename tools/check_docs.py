"""Validate the Sphinx API, CLI, mock, and example contracts."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_ROOT = REPOSITORY_ROOT / "docs"
DOCS_EXTENSIONS = DEFAULT_DOCS_ROOT / "_ext"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(DOCS_EXTENSIONS))

DOC_MOCK_IMPORTS = importlib.import_module("mock_imports").DOC_MOCK_IMPORTS

_AUTODOC_PATTERN = re.compile(
    r"^```\{(?:autoclass|autoexception|autofunction|autodata)\}"
    r"\s+(?P<target>\S+)\s*$",
    re.MULTILINE,
)
_CLICK_PATTERN = re.compile(
    r"^```\{click\}\s+(?P<target>\S+)\s*\n"
    r"(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
_PYTHON_BLOCK_PATTERN = re.compile(
    r"(?P<skip><!-- docs-check: skip-python -->\s*)?"
    r"```python\s*\n(?P<code>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
_TOCTREE_PATTERN = re.compile(
    r"^```\{toctree\}\s*\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CSS_URL_PATTERN = re.compile(
    r"url\(\s*['\"]?(?P<target>[^)'\"\s]+)['\"]?\s*\)",
    re.IGNORECASE,
)
_CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
_NARROW_MEDIA_PATTERN = re.compile(
    r"@media\s*\(\s*max-width\s*:",
    re.IGNORECASE,
)
_FORCED_COLORS_MEDIA_PATTERN = re.compile(
    r"@media\s*\(\s*forced-colors\s*:\s*active\s*\)",
    re.IGNORECASE,
)

_RAY_STYLE_ROOT_TARGETS = (
    "overview/index",
    "getting-started/index",
    "examples/index",
    "data/index",
    "algorithms/index",
    "model-lifecycle/index",
    "inference/index",
    "vector-index/index",
    "reference/index",
    "ray-jobs/index",
    "operations/index",
    "developer/index",
    "security/index",
)
_RAY_SIDEBAR_OPTIONS = (
    "startdepth=0",
    "show_nav_level=0",
    "maxdepth=4",
    "collapse=False",
    "includehidden=True",
)

_REQUIRED_ROOT_API_TARGETS = frozenset(
    {
        "tributo.DataSourceError",
        "tributo.JobConfig",
        "tributo.JobConfigurationError",
        "tributo.JobExecutionError",
        "tributo.JobSubmissionError",
        "tributo.JobTimeoutError",
        "tributo.ModelExportError",
        "tributo.RayJob",
        "tributo.TributoClient",
        "tributo.TributoError",
    }
)
_REQUIRED_TOP_LEVEL_COMMANDS = frozenset(
    {
        "algo",
        "explain",
        "export",
        "export-gc",
        "inspect",
        "logs",
        "registry",
        "serve",
        "status",
        "stop",
        "submit",
        "tune",
        "vector",
    }
)
_REQUIRED_NESTED_COMMANDS = frozenset(
    {
        "tributo algo config-schema",
        "tributo registry transition",
        "tributo serve grpc start",
        "tributo serve streaming start",
        "tributo tune run",
        "tributo vector build",
        "tributo vector compact",
        "tributo vector optimize",
        "tributo vector result",
        "tributo vector search",
    }
)
_REQUIRED_CLI_GUIDE_LINKS = frozenset(
    {
        "(algorithms/index.md)",
        "(vector-index/index.md)",
        "(inference/index.md)",
        "(model-lifecycle/index.md)",
        "(ray-jobs/index.md)",
    }
)

_RAY_HEADING_PROPER_NAMES = frozenset(
    {
        "API",
        "Bundle",
        "CLI",
        "Jobs",
        "MLflow",
        "ONNX",
        "Ray",
        "Tributo",
        "X-Learner",
    }
)


def autodoc_targets(text: str) -> tuple[str, ...]:
    """Extract explicit MyST autodoc targets."""
    return tuple(match.group("target") for match in _AUTODOC_PATTERN.finditer(text))


def click_directive(text: str) -> tuple[str, dict[str, str]] | None:
    """Return the single sphinx-click target and its options."""
    matches = list(_CLICK_PATTERN.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    options: dict[str, str] = {}
    for line in match.group("body").splitlines():
        option = re.fullmatch(r":([^:]+):\s*(.*)", line.strip())
        if option:
            options[option.group(1)] = option.group(2)
    return match.group("target"), options


def root_toctree_targets(text: str) -> tuple[str, ...]:
    """Extract ordered targets from the root MyST toctree."""
    match = _TOCTREE_PATTERN.search(text)
    if match is None:
        return ()

    targets: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        titled_target = re.fullmatch(r".+?\s*<(?P<target>[^>]+)>", line)
        targets.append(
            titled_target.group("target") if titled_target is not None else line
        )
    return tuple(targets)


def validate_navigation(docs_root: Path) -> list[str]:
    """Validate the Ray-style global documentation navigation contract."""
    errors: list[str] = []
    index_path = docs_root / "index.md"
    targets = root_toctree_targets(index_path.read_text(encoding="utf-8"))
    if targets != _RAY_STYLE_ROOT_TARGETS:
        errors.append(
            f"{index_path}: unexpected root navigation order: "
            f"{targets!r}; expected {_RAY_STYLE_ROOT_TARGETS!r}"
        )

    for target in _RAY_STYLE_ROOT_TARGETS:
        source = docs_root / target
        candidates = (
            source.with_suffix(".md"),
            source.with_suffix(".rst"),
        )
        if not any(candidate.is_file() for candidate in candidates):
            errors.append(f"{index_path}: navigation target does not exist: {target}")

    sidebar_path = docs_root / "_templates" / "main-sidebar.html"
    if not sidebar_path.is_file():
        errors.append(f"{sidebar_path}: global sidebar template is missing")
        return errors

    sidebar = sidebar_path.read_text(encoding="utf-8")
    if "generate_toctree_html(" not in sidebar:
        errors.append(f"{sidebar_path}: global toctree renderer is missing")
    for option in _RAY_SIDEBAR_OPTIONS:
        if option not in sidebar:
            errors.append(f"{sidebar_path}: missing Ray-style option {option}")
    return errors


def resolve_target(target: str) -> Any:
    """Import a dotted object without hiding dependency import failures."""
    parts = target.split(".")
    for boundary in range(len(parts), 0, -1):
        module_name = ".".join(parts[:boundary])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        for attribute in parts[boundary:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(f"Cannot resolve documented target {target!r}")


def validate_mock_inventory() -> list[str]:
    """Validate the shared third-party-only mock inventory."""
    errors: list[str] = []
    if tuple(sorted(DOC_MOCK_IMPORTS)) != DOC_MOCK_IMPORTS:
        errors.append("DOC_MOCK_IMPORTS must remain sorted")
    if len(set(DOC_MOCK_IMPORTS)) != len(DOC_MOCK_IMPORTS):
        errors.append("DOC_MOCK_IMPORTS contains duplicates")
    first_party = [
        name
        for name in DOC_MOCK_IMPORTS
        if name == "tributo" or name.startswith("tributo.")
    ]
    if first_party:
        errors.append(f"First-party modules must not be mocked: {first_party}")
    return errors


def validate_api_reference(docs_root: Path, *, import_objects: bool) -> list[str]:
    """Validate generated API coverage and optional real imports."""
    from tools.generate_public_api_reference import build_inventory, check_pages

    api_root = docs_root / "reference" / "api"
    paths = tuple(sorted(api_root.glob("*.md")))
    errors: list[str] = []
    if not paths:
        errors.append(f"{api_root}: no generated API pages found")
        return errors

    targets = tuple(
        target
        for path in paths
        for target in autodoc_targets(path.read_text(encoding="utf-8"))
    )
    target_counts = Counter(targets)
    duplicates = sorted(target for target, count in target_counts.items() if count > 1)
    if duplicates:
        errors.append(f"{api_root}: duplicate autodoc targets: {duplicates}")

    inventory = build_inventory()
    expected = {symbol.documentation_target: symbol for symbol in inventory}
    missing_public = sorted(set(expected) - set(targets))
    if missing_public:
        errors.append(f"{api_root}: missing PublicAPI targets: {missing_public}")
    unexpected = sorted(set(targets) - set(expected))
    if unexpected:
        errors.append(f"{api_root}: undocumented source for targets: {unexpected}")
    missing_static = sorted(_REQUIRED_ROOT_API_TARGETS - set(targets))
    if missing_static:
        errors.append(f"{api_root}: missing root API targets: {missing_static}")

    if docs_root.resolve() == DEFAULT_DOCS_ROOT.resolve():
        errors.extend(check_pages(inventory))

    if not import_objects:
        return errors

    from tributo.util.annotations import get_stability

    for target in targets:
        symbol = expected.get(target)
        if symbol is None:
            continue
        try:
            obj = resolve_target(target)
        except (AttributeError, ImportError) as exc:
            errors.append(f"{target}: import failed: {type(exc).__name__}: {exc}")
            continue
        stability = get_stability(obj)
        if stability != symbol.stability:
            errors.append(
                f"{target}: runtime stability {stability!r} does not match "
                f"source inventory {symbol.stability!r}"
            )
    return errors


def command_paths(command: Any, prefix: str = "tributo") -> tuple[str, ...]:
    """Return every descendant Click command path."""
    paths: list[str] = []
    commands = getattr(command, "commands", None)
    if not isinstance(commands, dict):
        return ()
    for name, child in sorted(commands.items()):
        path = f"{prefix} {name}"
        paths.append(path)
        paths.extend(command_paths(child, path))
    return tuple(paths)


def validate_cli_reference(path: Path, *, import_cli: bool) -> list[str]:
    """Validate the sphinx-click directive and optionally the real tree."""
    text = path.read_text(encoding="utf-8")
    directive = click_directive(text)
    if directive is None:
        return [f"{path}: expected exactly one click directive"]

    target, options = directive
    errors: list[str] = []
    if target != "tributo.cli:main":
        errors.append(f"{path}: unexpected Click target {target!r}")
    if options.get("prog") != "tributo":
        errors.append(f"{path}: click directive must set :prog: tributo")
    if options.get("nested") != "full":
        errors.append(f"{path}: click directive must set :nested: full")
    missing_guides = sorted(
        target for target in _REQUIRED_CLI_GUIDE_LINKS if target not in text
    )
    if missing_guides:
        errors.append(f"{path}: missing component guide links: {missing_guides}")

    if not import_cli:
        return errors

    module_name, attribute = target.split(":", 1)
    module = importlib.import_module(module_name)
    main = getattr(module, attribute)
    commands = getattr(main, "commands", {})
    missing_top = sorted(_REQUIRED_TOP_LEVEL_COMMANDS - set(commands))
    if missing_top:
        errors.append(f"CLI is missing top-level commands: {missing_top}")

    discovered = set(command_paths(main))
    missing_nested = sorted(_REQUIRED_NESTED_COMMANDS - discovered)
    if missing_nested:
        errors.append(f"CLI is missing nested commands: {missing_nested}")

    for path_name in sorted(discovered):
        current: Any = main
        for segment in path_name.split()[1:]:
            current = current.commands[segment]
        if not (getattr(current, "help", None) or getattr(current, "short_help", None)):
            errors.append(f"{path_name}: command has no help text")
    return errors


def validate_python_examples(docs_root: Path) -> list[str]:
    """Compile executable Python fences without running cluster workloads."""
    errors: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in _PYTHON_BLOCK_PATTERN.finditer(text):
            if match.group("skip"):
                continue
            line = text.count("\n", 0, match.start()) + 1
            try:
                compile(match.group("code"), f"{path}:{line}", "exec")
            except SyntaxError as exc:
                errors.append(f"{path}:{line}: invalid Python example: {exc.msg}")
    return errors


def validate_doc_code(docs_root: Path) -> list[str]:
    """Compile repository-backed documentation examples."""
    errors: list[str] = []
    root = docs_root / "examples" / "doc_code"
    if not root.is_dir():
        return [f"{root}: executable documentation example directory is missing"]
    paths = tuple(sorted(root.glob("*.py")))
    if not paths:
        return [f"{root}: no executable documentation examples found"]
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path}:{exc.lineno}: invalid doc_code: {exc.msg}")
    return errors


def validate_repository_examples(repository_root: Path) -> list[str]:
    """Compile root cluster examples without starting external infrastructure."""
    errors: list[str] = []
    root = repository_root / "examples"
    if not root.is_dir():
        return [f"{root}: repository example directory is missing"]
    paths = tuple(sorted(root.glob("*.py")))
    if not paths:
        return [f"{root}: no repository Python examples found"]
    for path in paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path}:{exc.lineno}: invalid repository example: {exc.msg}")
    return errors


def validate_publication_metadata(docs_root: Path) -> list[str]:
    """Keep published documentation links aligned with project metadata."""
    repository_root = docs_root.parent
    pyproject_path = repository_root / "pyproject.toml"
    readme_path = repository_root / "README.md"
    versions_path = docs_root / "_static" / "versions.json"
    errors: list[str] = []

    try:
        project_metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))[
            "project"
        ]
        documentation_url = project_metadata["urls"]["Documentation"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
        return [f"{pyproject_path}: cannot read project Documentation URL: {exc}"]

    if not isinstance(documentation_url, str):
        return [f"{pyproject_path}: project Documentation URL must be a string"]
    if not documentation_url.startswith("https://"):
        errors.append(f"{pyproject_path}: project Documentation URL must use HTTPS")
    if not documentation_url.endswith("/"):
        errors.append(f"{pyproject_path}: project Documentation URL must end with '/'")

    try:
        readme = readme_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{readme_path}: cannot read README: {exc}")
    else:
        required_readme_urls = (
            documentation_url,
            f"{documentation_url}getting-started/quickstart/",
            f"{documentation_url}reference/support-matrix/",
        )
        for required_url in required_readme_urls:
            if required_url not in readme:
                errors.append(
                    f"{readme_path}: missing canonical documentation URL "
                    f"{required_url!r}"
                )

    try:
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"{versions_path}: cannot read version switcher data: {exc}")
    else:
        if not isinstance(versions, list):
            errors.append(f"{versions_path}: version switcher data must be a list")
        else:
            latest = [
                entry
                for entry in versions
                if isinstance(entry, dict) and entry.get("version") == "latest"
            ]
            if len(latest) != 1:
                errors.append(
                    f"{versions_path}: expected exactly one latest version entry"
                )
            elif latest[0].get("url") != documentation_url:
                errors.append(
                    f"{versions_path}: latest URL {latest[0].get('url')!r} does "
                    f"not match project Documentation URL {documentation_url!r}"
                )
    return errors


def validate_ray_writing_style(docs_root: Path) -> list[str]:
    """Enforce objective Ray heading rules without linting code fences."""
    errors: list[str] = []
    numbered = re.compile(r"^(?:ADR\s+)?\d+(?:[.):]|\s)", re.IGNORECASE)
    word_pattern = re.compile(r"[A-Za-z][A-Za-z0-9+-]*")
    for path in sorted(docs_root.rglob("*.md")):
        in_fence = False
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = re.fullmatch(r"(?P<marks>#{1,6})\s+(?P<title>.+?)\s*", line)
            if match is None:
                continue
            title = match.group("title")
            if numbered.match(title):
                errors.append(
                    f"{path}:{line_number}: heading must not use a numeric prefix"
                )
            if "—" in title or "–" in title:
                errors.append(
                    f"{path}:{line_number}: heading must not use an em or en dash"
                )
            if len(match.group("marks")) != 1:
                continue
            words = word_pattern.findall(title)
            unexpected_title_case = [
                word
                for word in words[1:]
                if word[0].isupper()
                and not word.isupper()
                and word not in _RAY_HEADING_PROPER_NAMES
            ]
            if unexpected_title_case:
                errors.append(
                    f"{path}:{line_number}: H1 must use sentence case; "
                    f"unexpected title words {unexpected_title_case}"
                )
    return errors


def _xml_local_name(name: str) -> str:
    """Return an XML name without its namespace."""
    return name.rsplit("}", 1)[-1]


def validate_system_landscape_svg(path: Path) -> list[str]:
    """Validate the portable and accessible System Landscape SVG contract."""
    if not path.is_file():
        return [f"{path}: System Landscape SVG is missing"]

    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if _CJK_PATTERN.search(text):
        errors.append(f"{path}: SVG must not contain CJK characters")
    if "<!DOCTYPE" in text.upper():
        errors.append(f"{path}: SVG must not contain a DOCTYPE declaration")
        return errors
    if "<?xml-stylesheet" in text.lower():
        errors.append(f"{path}: SVG must not load an external stylesheet")
    if re.search(r"@import\b", text, re.IGNORECASE):
        errors.append(f"{path}: SVG must not use CSS @import")
    if re.search(r"javascript\s*:", text, re.IGNORECASE):
        errors.append(f"{path}: SVG must not contain javascript URLs")
    for match in _CSS_URL_PATTERN.finditer(text):
        target = match.group("target")
        if not target.startswith("#"):
            errors.append(f"{path}: SVG CSS resource must be an internal fragment")

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        errors.append(f"{path}: malformed SVG XML: {exc}")
        return errors

    if _xml_local_name(root.tag) != "svg":
        errors.append(f"{path}: root element must be svg")
        return errors
    if not root.get("viewBox"):
        errors.append(f"{path}: root svg must define viewBox")
    if root.get("role") != "img":
        errors.append(f"{path}: root svg must set role=img")

    labelled_ids = set(root.get("aria-labelledby", "").split())
    for element_name in ("title", "desc"):
        elements = [
            child for child in root if _xml_local_name(child.tag) == element_name
        ]
        if len(elements) != 1:
            errors.append(f"{path}: root svg must contain exactly one {element_name}")
            continue
        element = elements[0]
        element_id = element.get("id")
        if not element_id or element_id not in labelled_ids:
            errors.append(f"{path}: {element_name} id must appear in aria-labelledby")
        if not "".join(element.itertext()).strip():
            errors.append(f"{path}: {element_name} must not be empty")

    layout_groups: dict[str, ET.Element] = {}
    for element in root.iter():
        local_name = _xml_local_name(element.tag)
        if local_name in {"foreignObject", "script"}:
            errors.append(f"{path}: forbidden SVG element {local_name}")
        for attribute_name, value in element.attrib.items():
            if _xml_local_name(attribute_name) not in {"href", "src"}:
                continue
            if value and not value.startswith("#"):
                errors.append(
                    f"{path}: SVG resource references must use internal fragments"
                )
        if local_name == "g":
            for class_name in element.get("class", "").split():
                if class_name in {"desktop", "narrow"}:
                    layout_groups[class_name] = element

    for layout_name in ("desktop", "narrow"):
        layout = layout_groups.get(layout_name)
        if layout is None:
            errors.append(f"{path}: missing {layout_name} layout group")
            continue
        has_boundary = any(
            _xml_local_name(element.tag) == "rect"
            and "boundary" in element.get("class", "").split()
            for element in layout.iter()
        )
        if not has_boundary:
            errors.append(
                f"{path}: {layout_name} layout must show the framework boundary"
            )

    style_text = "\n".join(
        "".join(element.itertext())
        for element in root.iter()
        if _xml_local_name(element.tag) == "style"
    )
    style_text = _CSS_COMMENT_PATTERN.sub("", style_text)
    if not _NARROW_MEDIA_PATTERN.search(style_text):
        errors.append(f"{path}: SVG must define a narrow-screen media query")
    if not _FORCED_COLORS_MEDIA_PATTERN.search(style_text):
        errors.append(f"{path}: SVG must define a forced-colors theme")
    return errors


def validate_support_matrix(path: Path, *, compare_snapshot: bool) -> list[str]:
    """Validate generated markers and optionally compare real Registry facts."""
    from tools.generate_algorithm_support_matrix import (
        check_support_matrix,
        validate_marker_structure,
    )

    if not compare_snapshot:
        return validate_marker_structure(path)
    return check_support_matrix(path)


def run_checks(docs_root: Path, *, static_only: bool) -> list[str]:
    """Run all documentation contract checks."""
    errors = validate_mock_inventory()
    errors.extend(validate_navigation(docs_root))
    errors.extend(
        validate_api_reference(
            docs_root,
            import_objects=not static_only,
        )
    )
    errors.extend(
        validate_cli_reference(
            docs_root / "cli.md",
            import_cli=not static_only,
        )
    )
    errors.extend(validate_python_examples(docs_root))
    errors.extend(validate_doc_code(docs_root))
    errors.extend(validate_repository_examples(docs_root.parent))
    errors.extend(validate_publication_metadata(docs_root))
    errors.extend(validate_ray_writing_style(docs_root))
    errors.extend(
        validate_system_landscape_svg(
            docs_root / "images" / "tributo-system-landscape.svg"
        )
    )
    errors.extend(
        validate_support_matrix(
            docs_root / "reference" / "support-matrix.md",
            compare_snapshot=not static_only,
        )
    )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the command-line documentation checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args(argv)

    errors = run_checks(args.docs_root, static_only=args.static_only)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
