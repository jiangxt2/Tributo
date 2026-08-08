#!/usr/bin/env python3
"""Tributo PR pre-push static checks.

Layered architecture (Ray/Daft-style):
  Layer 0:  Format & Lint — ruff check + ruff format + mypy
  Layer 0.5: Dependency Resolution — cross-platform uv pip compile
  Layer 1:  API Stability — @PublicAPI annotation coverage
  Layer 2:  Python Safety — inline imports, error swallowing, None-safety
  Layer 2.5: Warning Suppressions — reject new type: ignore / noqa comments
  Layer 3:  Commit Message — Signed-off-by + format check + claims vs diff
  Layer 4:  General Hygiene — unintended files, merge conflicts, large files
  Layer 5:  Run Changed Tests — pytest on changed test files
  Layer 5.5: CI-parity Collection — dev-only environment collects the CI
             unit suite (catches optional-dependency imports in tests)
  Layer 5.6: Docs Spelling — runs the CI docs spelling build when the docs
             environment is available

Usage:
    uv run --locked --no-sync python scripts/pr-precheck.py
    uv run --locked --no-sync python scripts/pr-precheck.py --skip-tests
    uv run --locked --no-sync python scripts/pr-precheck.py --allow-network

Exit code 0 = all checks passed, 1 = issues found.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

DEFAULT_WORKTREE = None  # Auto-detect from git

# Run repository checks from the locked project environment. ``uvx`` is
# intentionally not used: it creates a separate tool environment and may
# access PyPI even when the project is already provisioned.
UV_RUN = ("uv", "run", "--locked", "--no-sync", "--offline")

# Keep this in sync with the unit-test job in
# ``.github/workflows/pr-test-suite.yml``.  S3 contract tests use the
# in-process Moto service and are safe here; MinIO and Ray runtime-env tests
# require external infrastructure and belong to their dedicated CI jobs.
CHANGED_TEST_MARKER_FILTER = (
    "not integration and not slow and not minio_compat and not ray_runtime_env"
)

MERGE_CONFLICT_MARKER_PATTERN = re.compile(
    r"^(<<<<<<<|=======|>>>>>>>)(?:[ \t].*)?\r?$", re.MULTILINE
)

# =============================================================================
# Utilities
# =============================================================================


def detect_worktree() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def run_project_command(
    root: str,
    args: list[str],
    *,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command from the locked, already-synced project environment."""
    return subprocess.run(
        [*UV_RUN, *args],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=timeout,
    )


def get_changed_files(root: str) -> list[str]:
    """Return tracked and standard-untracked files changed in the worktree.

    ``git diff`` does not report untracked files.  Include them explicitly so
    a newly created source, test, or tooling file cannot bypass the checks
    before its first commit.
    """
    changed: set[str] = set()
    base_found = False
    for base in [
        "upstream/main",
        "upstream/master",
        "origin/main",
        "origin/master",
        "main",
        "master",
    ]:
        check = subprocess.run(
            ["git", "rev-parse", "--verify", base],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if check.returncode != 0:
            continue
        merge_base = subprocess.run(
            ["git", "merge-base", base, "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if merge_base.returncode != 0:
            continue
        fork_point = merge_base.stdout.strip()
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", fork_point],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if result.returncode == 0:
            changed.update(f for f in result.stdout.splitlines() if f)
            base_found = True
            break

    if not base_found:
        print(
            "WARN: Could not find a base ref; tracked changes cannot be "
            "determined, so only standard-untracked files will be checked.",
            file=sys.stderr,
        )

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if untracked.returncode == 0:
        changed.update(f for f in untracked.stdout.splitlines() if f)

    return sorted(changed)


def get_diff_added_lines(root: str, rel_path: str) -> list[tuple[int, str]]:
    """Return (line_number, content) for lines added in a file's diff.

    For a standard-untracked file there is no Git diff yet, so treat the
    complete file as added.  This keeps suppression checks effective before
    the first commit of a new source or test file.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel_path],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if tracked.returncode != 0:
        full_path = os.path.join(root, rel_path)
        try:
            return list(enumerate(read_file(full_path).splitlines(), start=1))
        except OSError:
            return []

    try:
        for base in [
            "upstream/main",
            "upstream/master",
            "origin/main",
            "origin/master",
            "main",
            "master",
        ]:
            check = subprocess.run(
                ["git", "rev-parse", "--verify", base],
                capture_output=True,
                text=True,
                cwd=root,
            )
            if check.returncode != 0:
                continue
            merge_base = subprocess.run(
                ["git", "merge-base", base, "HEAD"],
                capture_output=True,
                text=True,
                cwd=root,
            )
            if merge_base.returncode != 0:
                continue
            fork_point = merge_base.stdout.strip()
            break
        else:
            return []

        result = subprocess.run(
            ["git", "diff", fork_point, "--", rel_path],
            capture_output=True,
            text=True,
            cwd=root,
        )
        added = []
        current_lineno = 0
        for line in result.stdout.split("\n"):
            if line.startswith("@@"):
                m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
                if m:
                    current_lineno = int(m.group(1)) - 1
                continue
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("+"):
                current_lineno += 1
                added.append((current_lineno, line[1:]))
            elif not line.startswith("-"):
                current_lineno += 1
        return added
    except Exception:
        return []


# =============================================================================
# Layer 0: Format & Lint
# =============================================================================


def check_format_lint(root: str, changed_files: list[str]) -> list[str]:
    issues = []

    # Mirrors pyproject.toml [tool.ruff] extend-exclude: protobuf code
    # is checked via CI's directory scan (which respects exclude), not
    # via explicit file arguments (which bypass it).
    EXCLUDED = ("src/tributo/serving/proto/inference_pb2",)
    py_files = [
        f for f in changed_files if f.endswith(".py") and not f.startswith(EXCLUDED)
    ]

    if not py_files:
        return issues

    # ruff format --check
    try:
        result = run_project_command(
            root,
            ["ruff", "format", "--check"] + [os.path.join(root, f) for f in py_files],
        )
        if result.returncode != 0:
            reformat = [
                line for line in result.stdout.split("\n") if "Would reformat" in line
            ]
            count = len(reformat) if reformat else 0
            issues.append(
                f"FAIL: ruff format --check — {count} file(s) need reformatting. "
                f"Run: uv run --locked --no-sync ruff format <file>"
            )
    except FileNotFoundError:
        issues.append("WARN: uv not found — skipping ruff format check")

    # ruff check
    try:
        result = run_project_command(
            root,
            ["ruff", "check"] + [os.path.join(root, f) for f in py_files],
        )
        if result.returncode != 0:
            error_count = (
                result.stdout.strip().count("\n") + 1 if result.stdout.strip() else 0
            )
            issues.append(
                f"FAIL: ruff check — {error_count} error(s) found. "
                f"Run: uv run --locked --no-sync ruff check --fix <file>"
            )
    except FileNotFoundError:
        issues.append("WARN: uv not found — skipping ruff check")

    # mypy
    try:
        result = run_project_command(
            root,
            ["mypy", "src/tributo"],
            timeout=120,
        )
        if result.returncode != 0:
            error_lines = [
                line for line in result.stdout.split("\n") if ": error:" in line
            ]
            issues.append(
                f"FAIL: mypy — {len(error_lines)} type error(s). "
                f"Run: uv run --locked --no-sync mypy src/tributo"
            )
    except FileNotFoundError:
        issues.append("WARN: uv/mypy not found — skipping mypy check")
    except subprocess.TimeoutExpired:
        issues.append("WARN: mypy timed out (>120s)")
    except Exception as e:
        issues.append(f"WARN: mypy check failed: {e}")

    return issues


# =============================================================================
# Layer 0.5: Dependency Resolution
# =============================================================================


def _parse_min_python_version(root: str) -> str | None:
    """Extract the minimum Python version from pyproject.toml.

    Parses ``requires-python = ">=X.Y,..."`` and returns ``"X.Y"``.
    """
    pyproject = os.path.join(root, "pyproject.toml")
    if not os.path.exists(pyproject):
        return None
    try:
        content = read_file(pyproject)
        m = re.search(r'requires-python\s*=\s*">=\s*(\d+\.\d+)', content)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _parse_python_versions_from_ci(root: str) -> list[str]:
    """Extract Python versions from the CI matrix in pr-test-suite.yml."""
    ci_path = os.path.join(root, ".github", "workflows", "pr-test-suite.yml")
    if not os.path.exists(ci_path):
        return []
    try:
        content = read_file(ci_path)
        m = re.search(r"python-version:\s*\[([^\]]+)\]", content)
        if m:
            return re.findall(r'"(\d+\.\d+)"', m.group(1))
    except Exception:
        pass
    return []


def check_dependency_resolution(root: str, allow_network: bool = False) -> list[str]:
    """Verify the dependency tree resolves for all declared Python versions.

    The default check is deliberately offline and validates the committed
    lockfile.  Cross-platform resolution is available explicitly with
    ``--allow-network``; it is not run implicitly from a developer machine.
    """
    issues = []

    if not allow_network:
        result = subprocess.run(
            ["uv", "lock", "--check"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            issues.append(
                "FAIL: uv.lock is out of date or cannot be validated. "
                "Run: uv lock\n"
                f"  {details[:300]}"
            )
        return issues

    min_ver = _parse_min_python_version(root)
    ci_vers = _parse_python_versions_from_ci(root)

    if not ci_vers:
        ci_vers = ["3.10", "3.11", "3.12", "3.13"]

    if min_ver and min_ver not in ci_vers:
        ci_vers.append(min_ver)
        ci_vers = sorted(set(ci_vers))

    try:
        with tempfile.TemporaryDirectory(prefix="tributo-precheck-") as output_dir:
            for py_ver in ci_vers:
                output_path = os.path.join(output_dir, f"requirements-{py_ver}.txt")
                result = subprocess.run(
                    [
                        "uv",
                        "pip",
                        "compile",
                        "--python-version",
                        py_ver,
                        "--python-platform",
                        "linux",
                        "pyproject.toml",
                        "-o",
                        output_path,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=root,
                    timeout=60,
                )
                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    issues.append(
                        f"FAIL: dependency resolution for Python {py_ver} on "
                        f"Linux failed — a required package may have dropped "
                        f"support for this Python version.\n"
                        f"  {stderr[:300]}"
                    )
    except FileNotFoundError:
        issues.append("WARN: uv not found — skipping dependency check")
    except subprocess.TimeoutExpired:
        issues.append("WARN: dependency resolution timed out")
    except Exception as e:
        issues.append(f"WARN: dependency check failed: {e}")

    return issues


# =============================================================================
# Layer 1: API Stability
# =============================================================================


def check_api_stability(root: str) -> list[str]:
    """Run the existing @PublicAPI annotation coverage checker."""
    issues = []

    check_script = os.path.join(root, "tools", "check_public_api_annotations.py")
    if not os.path.exists(check_script):
        issues.append("WARN: tools/check_public_api_annotations.py not found")
        return issues

    try:
        result = subprocess.run(
            [*UV_RUN, "python", check_script],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if result.returncode != 0:
            # Parse checker output: lines starting with "    • " are violations
            violation_lines = [
                line
                for line in result.stdout.split("\n")
                if line.lstrip().startswith("• ")
            ]
            count = len(violation_lines)
            issues.append(
                f"FAIL: {count} public symbol(s) without @PublicAPI annotation. "
                f"Run: uv run --locked --no-sync python "
                f"tools/check_public_api_annotations.py"
            )
    except Exception as e:
        issues.append(f"WARN: API annotation check failed: {e}")

    return issues


# =============================================================================
# Layer 2: Python Safety
# =============================================================================


def check_python_safety(root: str, changed_files: list[str]) -> list[str]:
    issues = []

    for rel_path in changed_files:
        if not rel_path.endswith(".py"):
            continue

        full_path = os.path.join(root, rel_path)
        if not os.path.exists(full_path):
            continue

        content = read_file(full_path)
        lines = content.split("\n")

        # 2a: Inline imports in test files
        if "/tests/" in rel_path or rel_path.startswith("tests/"):
            issues.extend(_check_inline_imports(rel_path, content))

        # 2b: Error swallowing
        issues.extend(_check_error_swallowing(rel_path, lines))

        # 2c: dict.get(key, default) None risk
        issues.extend(_check_dict_get_none(rel_path, lines))

        # 2d: Hardcoded sensitive patterns (IPs, tokens, internal hostnames)
        issues.extend(_check_hardcoded_sensitive(rel_path, lines))

        # 2e: KNOVA/ChinaMobile/internal references in new code.  The
        # checker itself necessarily contains the terms it searches for, so
        # applying this rule to its own source would report false failures.
        if rel_path != "scripts/pr-precheck.py":
            issues.extend(_check_internal_references(rel_path, root))

    return issues


def _check_inline_imports(rel_path: str, content: str) -> list[str]:
    issues = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return issues

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for parent in ast.walk(tree):
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_descendant(node, parent):
                        lineno = node.lineno
                        prefix = (
                            f"  from {node.module} import "
                            if isinstance(node, ast.ImportFrom)
                            else "  import "
                        )
                        names = ", ".join(alias.name for alias in node.names)
                        issues.append(
                            f"WARN: {rel_path}:{lineno}: inline import inside "
                            f"'{parent.name}()' — consider moving to module level\n"
                            f"{prefix}{names}"
                        )
                        break
    return issues


def _is_descendant(node: ast.AST, ancestor: ast.AST) -> bool:
    for child in ast.walk(ancestor):
        if child is node:
            return True
    return False


def _check_error_swallowing(rel_path: str, lines: list[str]) -> list[str]:
    issues = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not re.match(r"except\s+Exception\b", stripped):
            continue

        next_lines = []
        for j in range(i + 1, min(i + 4, len(lines))):
            next_lines.append(lines[j].strip())

        next_block = "\n".join(next_lines)

        has_log = bool(
            re.search(
                r"(logger\.|logging\.|warnings\.|\.error\(|\.warning\(|exc_info)",
                next_block,
            )
        )
        has_raise = bool(re.search(r"\braise\b", next_block))

        if has_log or has_raise:
            continue

        if re.search(r"^\s*(pass|return|continue)\b", next_block, re.MULTILINE):
            issues.append(
                f"WARN: {rel_path}:{i + 1}: broad 'except Exception' silently "
                f"swallowed — add logging or catch specific exception types\n"
                f"  {stripped[:120]}"
            )

    return issues


def _check_dict_get_none(rel_path: str, lines: list[str]) -> list[str]:
    issues = []

    for i, line in enumerate(lines):
        # dict.get(key, {}) — returns None when key exists but value is None
        if re.search(
            r"\.get\([^)]*,\s*\{\s*\}\s*\)\s*\.\s*(copy|items|keys|values|update|pop)\b",
            line,
        ):
            issues.append(
                f"WARN: {rel_path}:{i + 1}: dict.get(key, {{}}) returns None "
                f"when value is None — use (dict.get(key) or {{}})\n"
                f"  {line.strip()[:120]}"
            )

    return issues


def _check_hardcoded_sensitive(rel_path: str, lines: list[str]) -> list[str]:
    """Detect hardcoded IPs, tokens, internal hostnames in new code."""
    issues = []

    patterns = [
        (r"https?://172\.(1[6-9]|2\d|3[01])\.\d+\.\d+", "internal Docker IP"),
        (
            r"(password|passwd|token|secret|api_key)\s*=\s*[\"'][^\"']{8,}[\"']",
            "potential credential",
        ),
    ]

    for i, line in enumerate(lines):
        for pattern, desc in patterns:
            if re.search(pattern, line):
                issues.append(
                    f"WARN: {rel_path}:{i + 1}: {desc} detected\n  {line.strip()[:120]}"
                )

    return issues


def _check_internal_references(rel_path: str, root: str) -> list[str]:
    """Flag KNOVA / ChinaMobile references in added lines."""
    issues = []

    added = get_diff_added_lines(root, rel_path)

    # Only check added lines (not whole-file scan for pre-existing references)
    for lineno, line_text in added:
        if re.search(r"\bKNOVA\b", line_text, re.IGNORECASE):
            issues.append(
                f"FAIL: {rel_path}:{lineno}: 'KNOVA' reference in new code — "
                f"use 'TRIBUTO' instead"
            )
        if re.search(r"ChinaMobile|中国移动", line_text):
            issues.append(f"WARN: {rel_path}:{lineno}: internal reference in new code")

    return issues


# =============================================================================
# Layer 2.5: Warning Suppressions
# =============================================================================

# Zero-tolerance scope (governance boundary, ~/GitHub/plans/tributo-suppression-cleanup.md):
# no line-level suppression comments in new code. `# pragma: no cover` is
# excluded (kept as coverage policy, not an alert suppression).
#
# Known heuristic limitation: lines whose string literals contain suppression
# words inside string literals may false-positive; review such hits
# manually. This regex is the precise version of the CI plan-two grep.
SUPPRESSION_PATTERN = re.compile(r"# (type: ignore|mypy: ignore|pyright: ignore|noqa)")


def check_suppressions(root: str, changed_files: list[str]) -> list[str]:
    """Reject new warning-suppression comments in added lines.

    Mirrors the CI diff-scope check (plan item P1-a): scans only added lines of
    changed files under src/ and tests/ (docs are out of scope), so pre-existing
    suppressions do not block unrelated PRs. Once legacy suppressions are
    cleaned up, a --full mode can switch this to a whole-repo grep (CI plan
    item one).
    """
    issues = []
    for rel_path in changed_files:
        if not rel_path.endswith(".py"):
            continue
        if not (rel_path.startswith("src/") or rel_path.startswith("tests/")):
            continue
        for lineno, content in get_diff_added_lines(root, rel_path):
            if SUPPRESSION_PATTERN.search(content):
                issues.append(
                    f"FAIL: {rel_path}:{lineno}: new warning suppression "
                    f"comment {content.strip()!r} — remove it; use "
                    f"config-level exclusions (per-file-ignores) instead"
                )
    return issues


# =============================================================================
# Layer 3: Commit Message Validation
# =============================================================================


def check_commit_messages(root: str) -> list[str]:
    issues = []

    base_ref = None
    for ref in [
        "upstream/main",
        "upstream/master",
        "origin/main",
        "origin/master",
        "main",
        "master",
    ]:
        check = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if check.returncode == 0:
            base_ref = ref
            break

    if base_ref is None:
        return issues

    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--format=%H",
                "--reverse",
                "--first-parent",
                f"{base_ref}..HEAD",
            ],
            capture_output=True,
            text=True,
            cwd=root,
            check=True,
        )
        commits = [h for h in result.stdout.strip().split("\n") if h]
    except subprocess.CalledProcessError:
        return issues

    if not commits:
        return issues

    for commit_hash in commits:
        try:
            parent_check = subprocess.run(
                ["git", "cat-file", "-p", commit_hash],
                capture_output=True,
                text=True,
                cwd=root,
            )
            header = parent_check.stdout.split("\n\n", 1)[0]
            parent_count = sum(
                1 for line in header.split("\n") if line.startswith("parent ")
            )
            if parent_count > 1:
                continue

            result = subprocess.run(
                ["git", "log", "--format=%B", "-1", commit_hash],
                capture_output=True,
                text=True,
                cwd=root,
                check=True,
            )
            msg = result.stdout
        except subprocess.CalledProcessError:
            continue

        short = commit_hash[:10]
        first_line = msg.strip().split("\n")[0]

        # 3a: Signed-off-by
        if "Signed-off-by: jiangxt2" not in msg:
            issues.append(
                f"FAIL: Commit {short} missing "
                f"'Signed-off-by: jiangxt2 <jiangxt2@vip.qq.com>'"
            )

        # 3b: format check — accept [component] or Conventional Commits
        if not re.match(
            r"^(feat|fix|chore|perf|refactor|docs|test|ci|build|style|revert)"
            r"(\([a-z0-9_-]+\))?[!]?: ",
            first_line,
        ) and not re.match(
            r"^\[[a-zA-Z0-9_-]+\] ",
            first_line,
        ):
            issues.append(f"WARN: Commit {short} message format: {first_line[:80]}")

        # 3c: claims vs diff consistency
        issues.extend(_check_commit_claims_vs_diff(root, commit_hash, msg))

    return issues


def _check_commit_claims_vs_diff(
    root: str,
    commit_hash: str,
    msg: str,
) -> list[str]:
    issues = []
    short = commit_hash[:10]

    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
            capture_output=True,
            text=True,
            cwd=root,
            check=True,
        )
        changed = {
            file_name.strip()
            for file_name in result.stdout.strip().split("\n")
            if file_name.strip()
        }
    except subprocess.CalledProcessError:
        return issues

    claim_patterns = [
        (r"(?i)remove\s+(\S+)", "removed"),
        (r"(?i)rename\s+(\S+)", "renamed"),
        (r"(?i)delete\s+(\S+)", "deleted"),
        (r"(?i)drop\s+(\S+)", "dropped"),
    ]

    diff_text = ""
    for f in changed:
        try:
            result = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "-p", commit_hash, "--", f],
                capture_output=True,
                text=True,
                cwd=root,
                check=True,
            )
            diff_text += result.stdout
        except subprocess.CalledProcessError:
            pass

    for pattern, action in claim_patterns:
        for match in re.finditer(pattern, msg):
            claimed = match.group(1).strip(",.")
            if claimed not in diff_text and claimed not in str(changed):
                issues.append(
                    f"WARN: Commit {short} claims '{action} {claimed}' "
                    f"but no matching change found in diff"
                )

    return issues


# =============================================================================
# Layer 4: General Hygiene
# =============================================================================


def check_hygiene(root: str, changed_files: list[str]) -> list[str]:
    issues = []

    SUSPICIOUS = [
        (r"\.DS_Store$", "macOS metadata file"),
        (r"\.idea/", "IDE config directory"),
        (r"\.vscode/", "IDE config directory"),
        (r"\.iml$", "IDE module file"),
        (r"\.class$", "compiled Java class"),
        (r"\.pyc$", "compiled Python bytecode"),
        (r"\b__pycache__/", "Python cache directory"),
        (r"\.so$", "compiled C extension"),
        (r"\.env$", "environment file (possibly with secrets)"),
        (r"\.env\.backend$", "backend environment file"),
        (r"^CLAUDE\.md$", "AI instructions file"),
        (r"^htmlcov/", "coverage HTML report"),
        (r"\.coverage$", "coverage data file"),
    ]

    for f in changed_files:
        for pattern, desc in SUSPICIOUS:
            if re.search(pattern, f):
                issues.append(
                    f"FAIL: Suspicious file ({desc}) should not be committed: {f}"
                )

        full_path = os.path.join(root, f)
        if not os.path.exists(full_path):
            continue

        # Merge conflict markers. Match only marker lines so ordinary strings
        # and comments containing these tokens do not produce false positives.
        content = read_file(full_path)
        markers = sorted(
            {
                match.group(1)
                for match in MERGE_CONFLICT_MARKER_PATTERN.finditer(content)
            }
        )
        if markers:
            issues.append(f"FAIL: Merge conflict markers {markers} found in: {f}")

        # Large files (>500KB)
        size = os.path.getsize(full_path)
        if size > 500_000:
            issues.append(f"WARN: Large file ({size / 1_000_000:.1f}MB): {f}")

    return issues


# =============================================================================
# Layer 5: Run Changed Tests
# =============================================================================


def check_run_tests(root: str, changed_files: list[str]) -> list[str]:
    issues = []

    test_files = [
        f
        for f in changed_files
        if f.endswith(".py") and ("/tests/" in f or f.startswith("tests/"))
    ]
    if not test_files:
        return issues

    try:
        result = run_project_command(
            root,
            ["python", "-c", "import tributo"],
        )
        if result.returncode != 0:
            issues.append(
                "WARN: tributo is not importable from the locked environment — "
                "run 'uv sync --extra dev --locked' in the worktree"
            )
            return issues
    except FileNotFoundError:
        return issues

    for tf in test_files:
        full_path = os.path.join(root, tf)
        if not os.path.exists(full_path):
            continue

        print(f"  Running: {tf} ...", end=" ", flush=True)
        try:
            result = run_project_command(
                root,
                [
                    "python",
                    "-m",
                    "pytest",
                    tf,
                    "-q",
                    "--tb=short",
                    "-m",
                    CHANGED_TEST_MARKER_FILTER,
                ],
                timeout=180,
            )
            if result.returncode == 0:
                print("PASS")
            elif result.returncode == 5:
                # Exit 5 = no tests collected: module fully deselected by the
                # marker filter or skipped via pytest.importorskip — not a failure.
                print("SKIP (no tests collected)")
            else:
                print("FAIL")
                failures = []
                for line in result.stdout.split("\n"):
                    if line.startswith("FAILED ") or line.startswith("ERROR "):
                        failures.append(line.strip())
                if failures:
                    issues.append(
                        f"FAIL: {tf}: {len(failures)} failure(s)\n"
                        + "\n".join(f"  {f}" for f in failures[:10])
                    )
                else:
                    tail = "\n".join(result.stdout.split("\n")[-15:])
                    issues.append(
                        f"FAIL: {tf} — pytest exit {result.returncode}\n{tail}"
                    )
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            issues.append(f"WARN: {tf}: test timeout (>180s)")
        except Exception as e:
            print("ERROR")
            issues.append(f"WARN: {tf}: could not run — {e}")

    return issues


def check_ci_parity_collection(root: str) -> list[str]:
    """Collect the CI unit suite in a dev-only environment.

    The CI unit-tests job installs only the ``dev`` extra, while a local
    worktree usually has the data/data-daft/postgresql extras installed.  A
    test module importing an optional dependency at import time passes local
    collection but breaks CI with a ``ModuleNotFoundError`` during
    collection.  This layer rebuilds the dev-only environment in a temporary
    venv (from the uv cache, offline) and runs ``pytest --collect-only`` with
    the CI marker filter.
    """
    issues: list[str] = []
    venv_dir = tempfile.mkdtemp(prefix="tributo-ci-parity-")
    try:
        sync = subprocess.run(
            ["uv", "sync", "--extra", "dev", "--locked", "--offline"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=600,
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": venv_dir},
        )
        if sync.returncode != 0:
            tail = "\n".join((sync.stdout or sync.stderr).splitlines()[-3:])
            issues.append(
                "WARN: dev-only parity sync failed — CI-parity collection "
                f"skipped ({tail})"
            )
            return issues
        python = os.path.join(venv_dir, "bin", "python")
        collect = subprocess.run(
            [
                python,
                "-m",
                "pytest",
                "tests/",
                "--collect-only",
                "-q",
                "-m",
                CHANGED_TEST_MARKER_FILTER,
            ],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=300,
        )
        if collect.returncode != 0:
            errors = [
                line.strip()
                for line in collect.stdout.splitlines()
                if "ERROR" in line or "ModuleNotFoundError" in line
            ]
            issues.append(
                "FAIL: CI-parity collection failed — a test module imports an "
                "optional dependency at import time:\n" + "\n".join(errors[:10])
            )
        return issues
    finally:
        shutil.rmtree(venv_dir, ignore_errors=True)


def check_docs_spelling(root: str) -> list[str]:
    """Run the CI docs spelling check when a docs environment is available."""
    makefile = os.path.join(root, "Makefile")
    sphinx_build = os.path.join(root, ".docs-venv", "bin", "sphinx-build")
    if not os.path.exists(makefile) or not os.path.exists(sphinx_build):
        return [
            "WARN: .docs-venv not found — docs spelling check skipped "
            "(run the docs build locally or rely on the CI docs job)"
        ]
    result = subprocess.run(
        ["make", "spelling", f"SPHINXBUILD={sphinx_build}"],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=300,
    )
    if result.returncode != 0:
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if "Spell check" in line or "misspelled" in line
        ]
        issues = ["FAIL: docs spelling found unknown words:"]
        issues.extend(lines[:10])
        return issues
    return []


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Tributo PR pre-push checks")
    parser.add_argument(
        "--worktree",
        default=DEFAULT_WORKTREE,
        help="Path to Tributo worktree (auto-detected if not specified)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip layer 5 (run changed tests)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help=(
            "Run the optional cross-platform dependency resolver against "
            "package indexes; the default check is offline"
        ),
    )
    args = parser.parse_args()

    root = args.worktree or detect_worktree()
    if root is None:
        print(
            "ERROR: Could not detect git worktree. Run from the worktree or use --worktree."
        )
        sys.exit(1)

    if not os.path.exists(os.path.join(root, "pyproject.toml")):
        print(
            f"ERROR: {root} doesn't look like a Tributo repo (missing pyproject.toml)"
        )
        sys.exit(1)

    print(f"Worktree: {root}\n")

    changed_files = get_changed_files(root)
    if not changed_files:
        print("No changed files detected.")
        sys.exit(0)

    py_files = [f for f in changed_files if f.endswith(".py")]
    other = len(changed_files) - len(py_files)
    print(
        f"Changed: {len(changed_files)} files ({len(py_files)} Python, {other} other)\n"
    )

    all_issues: list[str] = []

    # Layer 0
    print("=== Layer 0: Format & Lint ===")
    l0 = check_format_lint(root, changed_files)
    all_issues.extend(l0)
    for issue in l0:
        print(f"  {issue}")
    print(f"  {'PASS' if not l0 else 'ISSUES'} — {len(l0)} issue(s)\n")

    # Layer 0.5
    print("=== Layer 0.5: Dependency Resolution ===")
    l05 = check_dependency_resolution(root, allow_network=args.allow_network)
    all_issues.extend(l05)
    for issue in l05:
        print(f"  {issue}")
    print(f"  {'PASS' if not l05 else 'ISSUES'} — {len(l05)} issue(s)\n")

    # Layer 1
    print("=== Layer 1: API Stability ===")
    l1 = check_api_stability(root)
    all_issues.extend(l1)
    for issue in l1:
        print(f"  {issue}")
    print(f"  {'PASS' if not l1 else 'ISSUES'} — {len(l1)} issue(s)\n")

    # Layer 2
    print("=== Layer 2: Python Safety ===")
    l2 = check_python_safety(root, changed_files)
    all_issues.extend(l2)
    for issue in l2:
        print(f"  {issue}")
    print(f"  {'PASS' if not l2 else 'ISSUES'} — {len(l2)} issue(s)\n")

    # Layer 2.5
    print("=== Layer 2.5: Warning Suppressions ===")
    l25 = check_suppressions(root, changed_files)
    all_issues.extend(l25)
    for issue in l25:
        print(f"  {issue}")
    print(f"  {'PASS' if not l25 else 'ISSUES'} — {len(l25)} issue(s)\n")

    # Layer 3
    print("=== Layer 3: Commit Message ===")
    l3 = check_commit_messages(root)
    all_issues.extend(l3)
    for issue in l3:
        print(f"  {issue}")
    print(f"  {'PASS' if not l3 else 'ISSUES'} — {len(l3)} issue(s)\n")

    # Layer 4
    print("=== Layer 4: General Hygiene ===")
    l4 = check_hygiene(root, changed_files)
    all_issues.extend(l4)
    for issue in l4:
        print(f"  {issue}")
    print(f"  {'PASS' if not l4 else 'ISSUES'} — {len(l4)} issue(s)\n")

    # Layer 5
    if args.skip_tests:
        l5: list[str] = []
        print("=== Layer 5: Tests (SKIPPED) ===\n")
    else:
        print("=== Layer 5: Run Changed Tests ===")
        l5 = check_run_tests(root, changed_files)
        all_issues.extend(l5)
        for issue in l5:
            print(f"  {issue}")
        print(f"  {'PASS' if not l5 else 'ISSUES'} — {len(l5)} issue(s)\n")

    # Layer 5.5
    print("=== Layer 5.5: CI-parity Collection ===")
    l55 = check_ci_parity_collection(root)
    all_issues.extend(l55)
    for issue in l55:
        print(f"  {issue}")
    print(f"  {'PASS' if not l55 else 'ISSUES'} — {len(l55)} issue(s)\n")

    # Layer 5.6
    print("=== Layer 5.6: Docs Spelling ===")
    l56 = check_docs_spelling(root)
    all_issues.extend(l56)
    for issue in l56:
        print(f"  {issue}")
    print(f"  {'PASS' if not l56 else 'ISSUES'} — {len(l56)} issue(s)\n")

    errors = [i for i in all_issues if i.startswith("FAIL")]
    warns = [i for i in all_issues if i.startswith("WARN")]

    print("=== Summary ===")
    print(f"  Layer 0 (Format):        {'PASS' if not l0 else f'{len(l0)} issue(s)'}")
    print(f"  Layer 0.5 (Deps):        {'PASS' if not l05 else f'{len(l05)} issue(s)'}")
    print(f"  Layer 1 (API):           {'PASS' if not l1 else f'{len(l1)} issue(s)'}")
    print(f"  Layer 2 (Safety):        {'PASS' if not l2 else f'{len(l2)} issue(s)'}")
    print(f"  Layer 2.5 (Suppress):    {'PASS' if not l25 else f'{len(l25)} issue(s)'}")
    print(f"  Layer 3 (Commit):        {'PASS' if not l3 else f'{len(l3)} issue(s)'}")
    print(f"  Layer 4 (Hygiene):       {'PASS' if not l4 else f'{len(l4)} issue(s)'}")
    if args.skip_tests:
        print("  Layer 5 (Tests):         SKIPPED")
    else:
        print(
            f"  Layer 5 (Tests):         {'PASS' if not l5 else f'{len(l5)} issue(s)'}"
        )
    print(f"  Layer 5.5 (CI-parity):   {'PASS' if not l55 else f'{len(l55)} issue(s)'}")
    print(f"  Layer 5.6 (Docs):        {'PASS' if not l56 else f'{len(l56)} issue(s)'}")
    print()

    if errors:
        print(f"❌ {len(errors)} ERROR(S) — fix before pushing:")
        for e in errors:
            print(f"  {e}")
        print()

    if warns:
        print(f"⚠️  {len(warns)} WARNING(S) — review before pushing:")
        for w in warns:
            print(f"  {w}")
        print()

    if errors:
        sys.exit(1)

    if warns:
        print("RESULT: PASSED with warnings — review above before pushing.")
    else:
        print("✅ RESULT: ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
