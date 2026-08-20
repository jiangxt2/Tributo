"""Generate component API pages from top-level ``@PublicAPI`` annotations."""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "tributo"
OUTPUT_ROOT = REPOSITORY_ROOT / "docs" / "reference" / "api"


@dataclass(frozen=True, order=True)
class PublicSymbol:
    """One source-level public object and its documentation target."""

    module: str
    name: str
    kind: str
    stability: str
    source_path: str
    line: int

    @property
    def target(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def documentation_target(self) -> str:
        return _ROOT_ALIASES.get(self.target, self.target)


@dataclass(frozen=True)
class Component:
    """A generated API page and its source-package routing rule."""

    slug: str
    title: str


COMPONENTS: Final[tuple[Component, ...]] = (
    Component("core", "Core API"),
    Component("data", "Data API"),
    Component("algorithms-training", "Algorithms and training API"),
    Component("model-lifecycle", "Model lifecycle API"),
    Component("inference-serving", "Inference and serving API"),
    Component("vector-index", "Vector-index API"),
    Component("extensions", "Pipeline and extension API"),
)

_ROOT_ALIASES: Final[dict[str, str]] = {
    "tributo.config.JobConfig": "tributo.JobConfig",
    "tributo.exceptions.DataSourceError": "tributo.DataSourceError",
    "tributo.exceptions.JobConfigurationError": "tributo.JobConfigurationError",
    "tributo.exceptions.JobExecutionError": "tributo.JobExecutionError",
    "tributo.exceptions.JobSubmissionError": "tributo.JobSubmissionError",
    "tributo.exceptions.JobTimeoutError": "tributo.JobTimeoutError",
    "tributo.exceptions.ModelExportError": "tributo.ModelExportError",
    "tributo.exceptions.TributoError": "tributo.TributoError",
    "tributo.job.RayJob": "tributo.RayJob",
    "tributo.job.TributoClient": "tributo.TributoClient",
}


class PublicAPIInventoryError(ValueError):
    """The source annotations cannot produce an unambiguous inventory."""


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _decorator_stability(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name) and decorator.id == "PublicAPI":
        return "beta"
    if not isinstance(decorator, ast.Call):
        return None
    function = decorator.func
    if not isinstance(function, ast.Name) or function.id != "PublicAPI":
        return None

    value: ast.expr | None = None
    for keyword in decorator.keywords:
        if keyword.arg == "stability":
            value = keyword.value
            break
    if value is None and decorator.args:
        value = decorator.args[0]
    if value is None:
        return "beta"
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        if value.value.id == "Stability":
            return value.attr.lower()
    raise PublicAPIInventoryError(
        "PublicAPI stability must be a string literal or Stability member"
    )


def _is_exception_class(node: ast.ClassDef) -> bool:
    """Classify exception types from their bases, not their domain name."""
    base_names = [
        base.id
        if isinstance(base, ast.Name)
        else base.attr
        if isinstance(base, ast.Attribute)
        else ""
        for base in node.bases
    ]
    return any(name.endswith(("Error", "Exception")) for name in base_names)


def build_inventory(source_root: Path = SOURCE_ROOT) -> tuple[PublicSymbol, ...]:
    """Return every top-level source object annotated with ``@PublicAPI``."""
    symbols: list[PublicSymbol] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(path, source_root)
        for node in tree.body:
            if not isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            stability_values = [
                stability
                for decorator in node.decorator_list
                if (stability := _decorator_stability(decorator)) is not None
            ]
            if not stability_values:
                continue
            if len(set(stability_values)) != 1:
                raise PublicAPIInventoryError(
                    f"{path}:{node.lineno}: conflicting PublicAPI decorators"
                )
            stability = stability_values[0]
            if stability not in {"stable", "beta", "alpha"}:
                raise PublicAPIInventoryError(
                    f"{path}:{node.lineno}: unsupported stability {stability!r}"
                )
            if isinstance(node, ast.ClassDef):
                kind = "exception" if _is_exception_class(node) else "class"
            else:
                kind = "function"
            symbols.append(
                PublicSymbol(
                    module=module,
                    name=node.name,
                    kind=kind,
                    stability=stability,
                    source_path=str(path.relative_to(REPOSITORY_ROOT)),
                    line=node.lineno,
                )
            )

    target_counts = Counter(symbol.target for symbol in symbols)
    duplicates = sorted(target for target, count in target_counts.items() if count > 1)
    if duplicates:
        raise PublicAPIInventoryError(f"duplicate public targets: {duplicates}")
    return tuple(sorted(symbols))


def component_for(symbol: PublicSymbol) -> str:
    """Route a public symbol to one user-facing component page."""
    parts = symbol.module.split(".")
    package = parts[1] if len(parts) > 1 else "core"
    if package in {"config", "exceptions", "job", "ray_jobs", "_common"}:
        return "core"
    if package in {"data", "streaming"}:
        return "data"
    if package in {"algorithms", "training"}:
        return "algorithms-training"
    if package in {"exporting", "registry"}:
        return "model-lifecycle"
    if package in {"inference", "serving", "explainability"}:
        return "inference-serving"
    if package == "vector_index":
        return "vector-index"
    if package == "pipeline":
        return "extensions"
    if package == "integrations" and len(parts) > 2:
        integration = parts[2]
        if integration in {"algorithm_inputs", "algorithm_runtimes"}:
            return "algorithms-training"
        if integration == "sources":
            return "data"
        if integration in {"exporters", "hooks", "storage", "validators"}:
            return "model-lifecycle"
        if integration in {"flavors", "model_importers", "sinks"}:
            return "inference-serving"
    return "extensions"


def render_component(component: Component, symbols: tuple[PublicSymbol, ...]) -> str:
    """Render one generated MyST API page."""
    lines = [
        f"# {component.title}",
        "",
        "```{important}",
        "This page is generated from top-level `@PublicAPI` annotations. Do not edit it",
        "by hand. Run `python tools/generate_public_api_reference.py` after changing",
        "a public annotation or moving a public object.",
        "```",
        "",
        "Stable, Beta, and Alpha objects appear because Ray-style API policy requires",
        "documentation for every public stability tier.",
    ]
    by_module: dict[str, list[PublicSymbol]] = {}
    for symbol in symbols:
        by_module.setdefault(symbol.module, []).append(symbol)
    for module, module_symbols in sorted(by_module.items()):
        lines.extend(("", f"## `{module}`", ""))
        for symbol in module_symbols:
            directive = {
                "class": "autoclass",
                "exception": "autoexception",
                "function": "autofunction",
            }[symbol.kind]
            lines.append(f"```{{{directive}}} {symbol.documentation_target}")
            if symbol.kind == "class":
                lines.append(":no-members:")
            lines.extend(("```", ""))
    return "\n".join(lines).rstrip("\n") + "\n"


def expected_pages(
    inventory: tuple[PublicSymbol, ...] | None = None,
) -> dict[Path, str]:
    """Return every generated path and its expected content."""
    resolved = build_inventory() if inventory is None else inventory
    pages: dict[Path, str] = {}
    for component in COMPONENTS:
        symbols = tuple(
            symbol for symbol in resolved if component_for(symbol) == component.slug
        )
        pages[OUTPUT_ROOT / f"{component.slug}.md"] = render_component(
            component, symbols
        )
    return pages


def check_pages(
    inventory: tuple[PublicSymbol, ...] | None = None,
) -> list[str]:
    """Return errors for missing, stale, or unexpected generated pages."""
    expected = expected_pages(inventory)
    errors: list[str] = []
    for path, content in expected.items():
        if not path.is_file():
            errors.append(f"{path}: generated API page is missing")
        elif path.read_text(encoding="utf-8") != content:
            errors.append(f"{path}: generated API page is stale")
    expected_paths = set(expected)
    if OUTPUT_ROOT.is_dir():
        for path in OUTPUT_ROOT.glob("*.md"):
            if path not in expected_paths:
                errors.append(f"{path}: unexpected generated API page")
    return errors


def write_pages() -> int:
    """Write generated pages and return the number that changed."""
    expected = expected_pages()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    changed = 0
    for path, content in expected.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    """Generate API pages or check them without writing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check:
            errors = check_pages()
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print("Public API reference is current.")
            return 0
        changed = write_pages()
    except (OSError, SyntaxError, PublicAPIInventoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Updated {changed} public API reference page(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
