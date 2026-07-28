"""Check that all symbols listed in ``__all__`` have ``@PublicAPI`` decorators.

Scans every Python source file under ``src/tributo/`` and reports any exported
symbol that is missing a ``@PublicAPI(stability=...)`` annotation.  For
``__init__.py`` re-exports, follows imports to verify the original definition.

Exit codes:
    0 — all exported symbols are annotated (or no ``__all__`` found)
    1 — one or more violations found

Usage::

    python scripts/check_public_api_annotations.py
    python scripts/check_public_api_annotations.py --verbose
"""

from __future__ import annotations

import ast
import argparse
import sys
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────

_PUBLIC_API_DECORATOR = "PublicAPI"
_DEVELOPER_API_DECORATOR = "DeveloperAPI"

# Symbols explicitly exempted from the @PublicAPI / @DeveloperAPI requirement.
# — PublicAPI / Stability / DeveloperAPI: self-referential
_EXEMPT_NAMES: set[str] = {"PublicAPI", "Stability", "get_stability", "is_public_api", "DeveloperAPI"}


# ── AST helpers ──────────────────────────────────────────────────────────────


class _ModuleCollector(ast.NodeVisitor):
    """Collect ``__all__`` exports, ``@PublicAPI`` names, and imports in one pass."""

    def __init__(self) -> None:
        self.exported: list[str] = []
        self.annotated: set[str] = set()
        # {imported_name: ("module.path", "original_name")}
        self.imports: dict[str, tuple[str, str]] = {}

    # -- __all__ ---------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.exported.append(elt.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id == "__all__":
            if isinstance(node.value, ast.List):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        self.exported.append(elt.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Handle __all__.extend([...]), __all__.append("x"), etc.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "__all__"
            and node.func.attr in ("extend", "append")
        ):
            if node.args:
                arg = node.args[0]
                names: list[str] = []
                if isinstance(arg, ast.List):
                    for elt in arg.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.append(elt.value)
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.append(arg.value)
                self.exported.extend(names)
        self.generic_visit(node)

    # -- @PublicAPI ------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._has_api_decorator(node.decorator_list):
            self.annotated.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._has_api_decorator(node.decorator_list):
            self.annotated.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._has_api_decorator(node.decorator_list):
            self.annotated.add(node.name)
        self.generic_visit(node)

    # -- imports ---------------------------------------------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        if (node.level or 0) > 0:
            # Relative imports (from .x import Y) — skip; the checker
            # cannot resolve them without knowing the package context.
            return
        module_path = node.module
        for alias in node.names:
            imported_as = alias.asname or alias.name
            self.imports[imported_as] = (module_path, alias.name)

    def visit_Import(self, node: ast.Import) -> None:
        # Handles bare ``import tributo.job`` — saves the module name
        # so that symbols in __all__ whose name matches the module can
        # still be resolved.
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = (alias.name, alias.name)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _has_api_decorator(decorator_list: list[ast.expr]) -> bool:
        for dec in decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                if dec.func.id in (_PUBLIC_API_DECORATOR, _DEVELOPER_API_DECORATOR):
                    return True
            elif isinstance(dec, ast.Name):
                if dec.id in (_PUBLIC_API_DECORATOR, _DEVELOPER_API_DECORATOR):
                    return True
        return False


# ── import resolution ────────────────────────────────────────────────────────


def _resolve_module(src_root: Path, module_path: str) -> Path | None:
    """Resolve a dotted module path to a .py file.

    *src_root* is the ``src/tributo/`` package directory.  Module paths like
    ``tributo.job`` are resolved against ``src_root.parent`` (``src/``).
    """
    parts = module_path.split(".")
    base = src_root.parent  # resolve tributo.* from src/
    # Try as package (__init__.py)
    pkg_path = base.joinpath(*parts, "__init__.py")
    if pkg_path.is_file():
        return pkg_path
    # Try as module (.py)
    mod_path = base.joinpath(*parts[:-1], parts[-1] + ".py")
    if mod_path.is_file():
        return mod_path
    return None


def _check_symbol_annotated(
    src_root: Path,
    collector: _ModuleCollector,
    filepath: Path,
    symbol: str,
    visited: set[str] | None = None,
) -> bool:
    """Check whether *symbol* (exported in *filepath*) has ``@PublicAPI``.

    Looks first in the current file, then follows imports to the defining module.
    *visited* prevents infinite import chains.
    """
    if visited is None:
        visited = set()

    cache_key = f"{filepath}:{symbol}"
    if cache_key in visited:
        return False
    visited.add(cache_key)

    # 1. Annotated directly in this file
    if symbol in collector.annotated:
        return True

    # 2. Follow import to the defining module
    if symbol in collector.imports:
        module_path, origin = collector.imports[symbol]
        defining_file = _resolve_module(src_root, module_path)
        if defining_file is not None and defining_file.is_file():
            try:
                source = defining_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(defining_file))
            except (SyntaxError, OSError):
                return False
            sub = _ModuleCollector()
            sub.visit(tree)
            if _check_symbol_annotated(src_root, sub, defining_file, origin, visited):
                return True

    return False


# ── scanning ─────────────────────────────────────────────────────────────────


def _check_file(src_root: Path, filepath: Path) -> list[tuple[str, str]]:
    """Check one source file. Returns list of (symbol, reason) violations."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, OSError) as exc:
        return [(f"<read-error>", f"cannot read or parse: {exc}")]

    collector = _ModuleCollector()
    collector.visit(tree)

    if not collector.exported:
        return []

    violations: list[tuple[str, str]] = []
    for symbol in collector.exported:
        if symbol in _EXEMPT_NAMES:
            continue
        if not _check_symbol_annotated(src_root, collector, filepath, symbol):
            violations.append(
                (symbol, "exported in __all__ but no @PublicAPI found (checked imports)")
            )

    return violations


def _scan_src(src_root: Path, verbose: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Scan all .py files under src_root. Returns {relpath: [(symbol, reason)]}."""
    results: dict[str, list[tuple[str, str]]] = {}

    for pyfile in sorted(src_root.rglob("*.py")):
        violations = _check_file(src_root, pyfile)
        if violations:
            rel = pyfile.relative_to(src_root.parent)  # relative to repo root
            results[str(rel)] = violations
            if verbose:
                for symbol, reason in violations:
                    print(f"  {rel}: {symbol} — {reason}")

    return results


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that __all__ exports have @PublicAPI annotations.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every violation as it is found",
    )
    parser.add_argument(
        "--src-root",
        default=None,
        help="Path to src/tributo directory (default: auto-detect)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    src_root = Path(args.src_root) if args.src_root else Path(__file__).resolve().parent.parent / "src" / "tributo"
    if not src_root.is_dir():
        print(f"Error: source root not found: {src_root}", file=sys.stderr)
        sys.exit(2)

    results = _scan_src(src_root, verbose=args.verbose)

    if not results:
        print("✅ All __all__ exports have @PublicAPI annotations.")
        sys.exit(0)

    total_violations = sum(len(v) for v in results.values())
    print(f"\n❌ {total_violations} violation(s) across {len(results)} module(s):\n")

    for filepath, violations in sorted(results.items()):
        print(f"  {filepath}:")
        for symbol, reason in violations:
            print(f"    • {symbol}: {reason}")

    sys.exit(1)


if __name__ == "__main__":
    main()
