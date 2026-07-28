#!/usr/bin/env python3
"""Verify that Python code examples in README.md parse correctly and reference
real import paths.

Usage:
    python scripts/verify_readme_examples.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def extract_python_blocks(readme_path: str) -> list[tuple[int, str]]:
    """Extract Python code blocks from a markdown file.

    Returns list of (starting_line_number, code_string) tuples.
    """
    with open(readme_path) as f:
        lines = f.readlines()

    blocks: list[tuple[int, str]] = []
    in_block = False
    block_lines: list[str] = []
    block_start = 0

    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```python"):
            in_block = True
            block_lines = []
            block_start = i
        elif line.strip() == "```" and in_block:
            in_block = False
            code = "".join(block_lines).strip()
            if code:
                blocks.append((block_start, code))
        elif in_block:
            block_lines.append(line)

    return blocks


def check_syntax(code: str, start_line: int) -> list[str]:
    """Check that Python code parses correctly. Returns list of error messages."""
    errors: list[str] = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"  Line {start_line}: SyntaxError — {e.msg} (line {e.lineno})")
    return errors


def extract_imports(code: str) -> list[tuple[int, str, str]]:
    """Extract import statements from code. Returns list of (line, module, name)."""
    imports: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name, alias.asname or alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    full = f"{node.module}.{alias.name}"
                    imports.append((node.lineno, full, alias.asname or alias.name))
    return imports


def verify_imports(
    imports: list[tuple[int, str, str]], start_line: int
) -> list[str]:
    """Verify that imported modules exist in the package or are known external deps.

    Returns list of warning/error messages.
    """
    messages: list[str] = []

    for _lineno, module, _name in imports:  # noqa: B007
        top_level = module.split(".")[0]

        if top_level == "tributo":
            # For 'from X import Y', the module is X (parent of the full path)
            # Example: 'tributo.training.pu_trainer.run_pu_training_from_json'
            #   → check 'src/tributo/training/pu_trainer.py'
            # For bare 'import tributo', check 'src/tributo/__init__.py'
            parent_parts = module.split(".")[:-1]  # Remove the imported symbol
            if not parent_parts:
                parent_parts = module.split(".")  # Bare import case
            module_path = Path("src") / Path(*parent_parts)
            py_file = module_path.with_suffix(".py")
            init_file = module_path / "__init__.py"

            if not py_file.exists() and not init_file.exists():
                messages.append(
                    f"  Line ~{start_line}: import '{module}' — "
                    f"module '{'.'.join(parent_parts)}' not found in src/"
                )
        # External deps are skipped (may not be installed)

    return messages


def main() -> int:
    readme = Path("README.md")
    if not readme.exists():
        print("README.md not found")
        return 1

    blocks = extract_python_blocks(str(readme))
    if not blocks:
        print("No Python code blocks found in README.md")
        return 1

    errors: list[str] = []
    for start_line, code in blocks:
        # Syntax check
        errors.extend(check_syntax(code, start_line))

        # Import verification
        try:
            imports = extract_imports(code)
            errors.extend(verify_imports(imports, start_line))
        except SyntaxError:
            pass  # Already reported
        except Exception as e:
            errors.append(f"  Line ~{start_line}: Could not analyze imports — {e}")

    if errors:
        print(f"❌ {len(errors)} issue(s) found in README examples:")
        for err in errors:
            print(err)
        return 1

    print(f"✅ {len(blocks)} Python code block(s) verified — all imports valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
