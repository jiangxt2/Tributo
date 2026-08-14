#!/usr/bin/env python3
"""Plan and enforce Tributo's budgeted CI test tiers.

The manifest is the only test-orchestration source of truth.  This module keeps
manual and quarantined integration suites visible to change planning while
making them impossible to execute through the CI-safe runner.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("ci/test-suites.json")
TIERS = frozenset({"ci_fast", "ci_scheduled", "manual_external", "quarantine"})
CI_TIERS = frozenset({"ci_fast", "ci_scheduled"})
WORKFLOWS = frozenset({"matrix", "unit", "docs", "scheduled", "external", "none"})
CI_PYTEST_ENTRYPOINT = ("python", "-m", "pytest")
GOVERNED_MARKERS = frozenset(
    {
        "ci_safe",
        "integration",
        "slow",
        "distributed",
        "manual_it",
        "minio_compat",
        "quarantine",
        "ray_runtime_env",
        "s3_contract",
        "ingestion_conformance",
        "tributo_walking_skeleton",
    }
)
SHARED_SELECTOR_MARKERS = frozenset({"s3_contract"})
MARKER_ALLOWED_TIERS = {
    "ci_safe": frozenset({"ci_fast"}),
    "manual_it": frozenset({"manual_external", "quarantine"}),
    "quarantine": frozenset({"quarantine"}),
    "s3_contract": frozenset({"ci_fast"}),
}
FORBIDDEN_CI_REQUIREMENTS = frozenset(
    {
        "docker",
        "external_credentials",
        "external_model_download",
        "gpu",
        "mlflow",
        "minio",
        "multi_node_ray",
        "ray_jobs",
        "ray_serve",
        "shared_mlflow_at_fixed_port",
        "unowned_clickhouse",
        "unowned_mlflow",
    }
)


class ManifestError(ValueError):
    """Raised when the test-suite manifest violates its schema or policy."""


@dataclass(frozen=True)
class Suite:
    """One validated suite declaration."""

    suite_id: str
    owner: str
    domain: str
    tier: str
    workflow: str
    entrypoint: tuple[str, ...]
    args: tuple[str, ...]
    extras: tuple[str, ...]
    test_paths: tuple[str, ...]
    trigger_domains: tuple[str, ...]
    trigger_paths: tuple[str, ...]
    requires: tuple[str, ...]
    markers: tuple[str, ...]
    budget_seconds: int
    ci_allowed: bool
    forbid_skips: bool
    owns_tests: bool
    default_test_owner: bool
    always_run: bool
    all_extras: bool
    log_contract: str
    rationale: str


@dataclass(frozen=True)
class Manifest:
    """Validated top-level test policy."""

    root: Path
    path: Path
    budgets: dict[str, int]
    path_rules: tuple[dict[str, Any], ...]
    suites: tuple[Suite, ...]

    def suite(self, suite_id: str) -> Suite:
        for suite in self.suites:
            if suite.suite_id == suite_id:
                return suite
        raise ManifestError(f"unknown test suite: {suite_id}")


def _strings(value: object, field: str, suite_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(
            f"suite {suite_id!r} field {field!r} must be a string array"
        )
    return tuple(value)


def _required_string(payload: dict[str, Any], field: str, suite_id: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"suite {suite_id!r} field {field!r} must be non-empty")
    return value


def load_manifest(
    root: Path | str = ROOT,
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> Manifest:
    """Load and structurally validate the declarative test manifest."""
    repository = Path(root).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = repository / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load test manifest {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ManifestError("test manifest schema_version must be 1")

    budgets = payload.get("budgets")
    if not isinstance(budgets, dict):
        raise ManifestError("test manifest budgets must be an object")
    required_budgets = {
        "ci_fast_suite_seconds",
        "ci_scheduled_suite_seconds",
        "nightly_total_seconds",
    }
    normalized_budgets: dict[str, int] = {}
    for name in required_budgets:
        value = budgets.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ManifestError(f"budget {name!r} must be a positive integer")
        normalized_budgets[name] = value

    raw_rules = payload.get("path_rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ManifestError("test manifest path_rules must be a non-empty array")
    path_rules: list[dict[str, Any]] = []
    rule_names: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ManifestError("each path rule must be an object")
        name = raw_rule.get("name")
        if not isinstance(name, str) or not name or name in rule_names:
            raise ManifestError(
                f"path rule name must be unique and non-empty: {name!r}"
            )
        rule_names.add(name)
        patterns = _strings(raw_rule.get("patterns"), "patterns", name)
        domains = _strings(raw_rule.get("domains"), "domains", name)
        if not patterns or not domains:
            raise ManifestError(f"path rule {name!r} needs patterns and domains")
        path_rules.append({"name": name, "patterns": patterns, "domains": domains})

    raw_suites = payload.get("suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ManifestError("test manifest suites must be a non-empty array")
    suites: list[Suite] = []
    suite_ids: set[str] = set()
    for raw_suite in raw_suites:
        if not isinstance(raw_suite, dict):
            raise ManifestError("each suite must be an object")
        suite_id = raw_suite.get("id")
        if not isinstance(suite_id, str) or not suite_id or suite_id in suite_ids:
            raise ManifestError(f"suite id must be unique and non-empty: {suite_id!r}")
        suite_ids.add(suite_id)
        tier = _required_string(raw_suite, "tier", suite_id)
        workflow = _required_string(raw_suite, "workflow", suite_id)
        if tier not in TIERS:
            raise ManifestError(f"suite {suite_id!r} has invalid tier {tier!r}")
        if workflow not in WORKFLOWS:
            raise ManifestError(f"suite {suite_id!r} has invalid workflow {workflow!r}")
        budget_seconds = raw_suite.get("budget_seconds")
        if (
            not isinstance(budget_seconds, int)
            or isinstance(budget_seconds, bool)
            or budget_seconds < 0
            or (tier != "quarantine" and budget_seconds == 0)
        ):
            raise ManifestError(
                f"suite {suite_id!r} budget_seconds must be positive when executable"
            )
        suite = Suite(
            suite_id=suite_id,
            owner=_required_string(raw_suite, "owner", suite_id),
            domain=_required_string(raw_suite, "domain", suite_id),
            tier=tier,
            workflow=workflow,
            entrypoint=_strings(raw_suite.get("entrypoint"), "entrypoint", suite_id),
            args=_strings(raw_suite.get("args"), "args", suite_id),
            extras=_strings(raw_suite.get("extras"), "extras", suite_id),
            test_paths=_strings(raw_suite.get("test_paths"), "test_paths", suite_id),
            trigger_domains=_strings(
                raw_suite.get("trigger_domains"), "trigger_domains", suite_id
            ),
            trigger_paths=_strings(
                raw_suite.get("trigger_paths"), "trigger_paths", suite_id
            ),
            requires=_strings(raw_suite.get("requires"), "requires", suite_id),
            markers=_strings(raw_suite.get("markers", []), "markers", suite_id),
            budget_seconds=budget_seconds,
            ci_allowed=raw_suite.get("ci_allowed") is True,
            forbid_skips=raw_suite.get("forbid_skips") is True,
            owns_tests=raw_suite.get("owns_tests", True) is True,
            default_test_owner=raw_suite.get("default_test_owner") is True,
            always_run=raw_suite.get("always_run") is True,
            all_extras=raw_suite.get("all_extras") is True,
            log_contract=_required_string(raw_suite, "log_contract", suite_id),
            rationale=_required_string(raw_suite, "rationale", suite_id),
        )
        suites.append(suite)
    return Manifest(
        root=repository,
        path=path,
        budgets=normalized_budgets,
        path_rules=tuple(path_rules),
        suites=tuple(suites),
    )


def _normalize_path(path: str) -> str:
    normalized = path.replace(os.sep, "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def path_matches(path: str, pattern: str) -> bool:
    """Match repository-relative paths with explicit recursive-directory rules."""
    normalized = _normalize_path(path)
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return normalized == prefix or normalized.startswith(f"{prefix}/")
    return fnmatch.fnmatchcase(normalized, pattern)


def _owning_suites(manifest: Manifest, path: str) -> list[Suite]:
    return [
        suite
        for suite in manifest.suites
        if suite.owns_tests and path in suite.test_paths
    ]


def suite_for_test_path(manifest: Manifest, path: str) -> Suite:
    """Resolve the one owning suite for a collected test module."""
    owners = _owning_suites(manifest, path)
    if len(owners) > 1:
        raise ManifestError(
            f"test path {path!r} has multiple owners: "
            + ", ".join(suite.suite_id for suite in owners)
        )
    if owners:
        return owners[0]
    defaults = [suite for suite in manifest.suites if suite.default_test_owner]
    if len(defaults) != 1:
        raise ManifestError("the manifest must define exactly one default test owner")
    return defaults[0]


def markers_for_test_path(manifest: Manifest, path: str) -> tuple[str, ...]:
    """Return collection markers derived from the owning suite."""
    suite = suite_for_test_path(manifest, path)
    markers = list(suite.markers)
    if suite.tier == "manual_external":
        markers.append("manual_it")
    elif suite.tier == "quarantine":
        markers.extend(("manual_it", "quarantine"))
    return tuple(dict.fromkeys(markers))


def _attribute_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _declared_markers(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    markers: set[str] = set()
    for node in ast.walk(tree):
        name = _attribute_name(node)
        if name is not None and name.startswith("pytest.mark."):
            markers.add(name.rsplit(".", 1)[-1])
    return markers


def _requires_explicit_owner(root: Path, relative_path: str) -> bool:
    path = Path(relative_path)
    parts = set(path.parts)
    if "integration" in parts or "integrations" in parts:
        return True
    if "integration" in path.name:
        return True
    owner_only_markers = GOVERNED_MARKERS - SHARED_SELECTOR_MARKERS
    return bool(_declared_markers(root / path) & owner_only_markers)


def _pytest_marker_expression(suite: Suite) -> str | None:
    for index, argument in enumerate(suite.args):
        if argument == "-m" and index + 1 < len(suite.args):
            return suite.args[index + 1]
        if argument.startswith("-m="):
            return argument[3:]
    return None


def _positive_marker_names(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return set()
    positive: set[str] = set()

    def visit(node: ast.AST, negated: bool = False) -> None:
        if isinstance(node, ast.Expression):
            visit(node.body, negated)
        elif isinstance(node, ast.Name):
            if not negated:
                positive.add(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            visit(node.operand, not negated)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                visit(value, negated)

    visit(tree)
    return positive


def _marker_expression_matches(expression: str, markers: set[str]) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    def evaluate(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name):
            return node.id in markers
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return all(evaluate(value) for value in node.values)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return any(evaluate(value) for value in node.values)
        raise ValueError("unsupported pytest marker expression")

    try:
        return evaluate(tree)
    except ValueError:
        return False


def _marker_expression_is_supported(expression: str) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    supported = (
        ast.Expression,
        ast.Name,
        ast.Load,
        ast.UnaryOp,
        ast.Not,
        ast.BoolOp,
        ast.And,
        ast.Or,
    )
    return all(isinstance(node, supported) for node in ast.walk(tree))


def _marker_expression_can_match(
    expression: str, fixed_markers: dict[str, bool]
) -> bool:
    if not _marker_expression_is_supported(expression):
        return False
    tree = ast.parse(expression, mode="eval")

    def possible(node: ast.AST) -> set[bool]:
        if isinstance(node, ast.Expression):
            return possible(node.body)
        if isinstance(node, ast.Name):
            if node.id in fixed_markers:
                return {fixed_markers[node.id]}
            return {False, True}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return {not value for value in possible(node.operand)}
        if isinstance(node, ast.BoolOp):
            values = {True} if isinstance(node.op, ast.And) else {False}
            for child in node.values:
                child_values = possible(child)
                if isinstance(node.op, ast.And):
                    values = {
                        left and right for left in values for right in child_values
                    }
                else:
                    values = {
                        left or right for left in values for right in child_values
                    }
            return values
        return set()

    return True in possible(tree)


def _suite_selects_path(suite: Suite, relative_path: str) -> bool:
    normalized = _normalize_path(relative_path)
    return any(
        _normalize_path(argument.split("::", 1)[0]) == normalized
        for argument in suite.args
        if not argument.startswith("-")
    )


def _shared_marker_selectors(
    manifest: Manifest, relative_path: str, marker: str
) -> list[Suite]:
    selectors: list[Suite] = []
    active_markers = _declared_markers(manifest.root / relative_path)
    try:
        active_markers.update(markers_for_test_path(manifest, relative_path))
    except ManifestError:
        pass
    for suite in manifest.suites:
        expression = _pytest_marker_expression(suite)
        if (
            suite.tier in CI_TIERS
            and suite.ci_allowed
            and _suite_selects_path(suite, relative_path)
            and expression is not None
            and _positive_marker_names(expression) == {marker}
            and _marker_expression_matches(expression, active_markers)
        ):
            selectors.append(suite)
    return selectors


def _pytest_test_targets(suite: Suite) -> tuple[str, ...]:
    targets: list[str] = []
    for argument in suite.args:
        if argument.startswith("-"):
            continue
        candidate = _normalize_path(argument.split("::", 1)[0]).rstrip("/")
        if candidate == "tests" or candidate.startswith("tests/"):
            targets.append(candidate)
    return tuple(dict.fromkeys(targets))


def _ci_suite_selection_error(manifest: Manifest, suite: Suite) -> str | None:
    expression = _pytest_marker_expression(suite)
    if expression is not None:
        if not _marker_expression_is_supported(expression):
            return f"CI suite {suite.suite_id} has an invalid marker expression"
        if _marker_expression_can_match(expression, {"manual_it": True}):
            return f"CI suite {suite.suite_id} can select manual_it tests"

    targets = _pytest_test_targets(suite)
    if not targets:
        return f"CI suite {suite.suite_id} has no controlled test target"
    for target in targets:
        if not (manifest.root / target).exists():
            return f"CI suite {suite.suite_id} references missing target {target}"
    if expression is None:
        for target in targets:
            path = manifest.root / target
            if path.is_dir():
                return (
                    f"CI suite {suite.suite_id} selects a test directory without "
                    "a marker safety expression"
                )
            try:
                owner = suite_for_test_path(manifest, target)
            except ManifestError as exc:
                return f"CI suite {suite.suite_id} has an unsafe test target: {exc}"
            if owner.tier not in CI_TIERS:
                return (
                    f"CI suite {suite.suite_id} directly selects non-CI test "
                    f"target {target}"
                )
            active_markers = _declared_markers(path)
            active_markers.update(markers_for_test_path(manifest, target))
            if active_markers & {"manual_it", "quarantine"}:
                return (
                    f"CI suite {suite.suite_id} directly selects non-CI marked "
                    f"target {target}"
                )
    return None


def _workflow_run_commands(text: str) -> tuple[str, ...]:
    """Extract active YAML run commands while excluding comments and labels."""
    commands: list[str] = []
    block: list[str] | None = None
    run_indent = -1

    def flush() -> None:
        nonlocal block
        if block:
            commands.append(" ".join(block))
        block = None

    for line in text.splitlines():
        while block is not None:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped and indent <= run_indent:
                flush()
                continue
            if stripped and not stripped.startswith("#"):
                block.append(stripped)
            break
        else:
            match = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", line)
            if match is None:
                continue
            value = match.group(2).strip()
            if re.fullmatch(r"[>|](?:[+-]?\d?|\d[+-]?)", value):
                block = []
                run_indent = len(match.group(1))
            elif value and not value.startswith("#"):
                commands.append(value)
    flush()
    return tuple(commands)


def _uses_planner(commands: Sequence[str], subcommand: str) -> bool:
    pattern = re.compile(
        rf"\bpython3?\s+scripts/ci_test_plan\.py\s+{re.escape(subcommand)}\b"
    )
    return any(pattern.search(command) for command in commands)


def _runs_planner_suite(commands: Sequence[str], suite_pattern: str) -> bool:
    pattern = re.compile(
        r"\bpython3?\s+scripts/ci_test_plan\.py\s+run\b"
        rf".*--suite\s+{suite_pattern}(?:\s|$)"
    )
    return any(pattern.search(command) for command in commands)


def _workflow_errors(manifest: Manifest) -> list[str]:
    errors: list[str] = []
    workflow_dir = manifest.root / ".github" / "workflows"
    workflow_paths = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
    workflows = {
        path.relative_to(manifest.root).as_posix(): path.read_text(encoding="utf-8")
        for path in workflow_paths
    }
    active_commands = {
        path: _workflow_run_commands(text) for path, text in workflows.items()
    }
    active_combined = "\n".join(
        command for commands in active_commands.values() for command in commands
    )
    if re.search(r"run_[^\s\"']*_it\.sh", active_combined):
        errors.append("GitHub Actions contains an external IT runner reference")
    if re.search(r"tools/tributo_it\.py\s+run-", active_combined):
        errors.append("GitHub Actions directly invokes an external IT lifecycle")
    forbidden_paths = {
        path
        for suite in manifest.suites
        if suite.tier in {"manual_external", "quarantine"}
        for path in suite.test_paths
    }
    for path in sorted(forbidden_paths):
        if path in active_combined:
            errors.append(f"GitHub Actions references non-CI suite path: {path}")

    pr_path = ".github/workflows/pr-test-suite.yml"
    nightly_path = ".github/workflows/nightly-test.yml"
    pr = workflows.get(pr_path, "")
    nightly = workflows.get(nightly_path, "")
    pr_commands = active_commands.get(pr_path, ())
    nightly_commands = active_commands.get(nightly_path, ())
    pr_structure = {
        "merge_group event": r"(?m)^  merge_group:\s*$",
        "ci-safe-integration job": r"(?m)^  ci-safe-integration:\s*$",
        "core-gate job": r"(?m)^  core-gate:\s*$",
        "always-run gate": r"(?m)^    if:\s*always\(\)\s*$",
    }
    for label, pattern in pr_structure.items():
        if re.search(pattern, pr) is None:
            errors.append(f"{pr_path} is missing required {label}")
    if not _uses_planner(pr_commands, "audit"):
        errors.append(f"{pr_path} is missing an active manifest audit command")
    if not _uses_planner(pr_commands, "plan"):
        errors.append(f"{pr_path} is missing an active manifest plan command")
    if not _runs_planner_suite(pr_commands, "unit"):
        errors.append(f"{pr_path} is missing the active unit suite runner command")
    matrix_pattern = r'["\']?\$\{\{\s*matrix\.suite\s*\}\}["\']?'
    if not _runs_planner_suite(pr_commands, matrix_pattern):
        errors.append(f"{pr_path} is missing the active matrix suite runner command")
    for forbidden in (
        "data-ingestion-distributed:",
        "lance-vector-distributed:",
        "inference-distributed:",
        "core-walking-skeleton:",
        "docker",
    ):
        if forbidden in pr:
            errors.append(
                f"{pr_path} contains forbidden heavy-CI fragment: {forbidden}"
            )
    if re.search(r"(?m)^    timeout-minutes:\s*30\s*$", nightly) is None:
        errors.append(f"{nightly_path} is missing the scheduled 30-minute timeout")
    if not _uses_planner(nightly_commands, "audit"):
        errors.append(f"{nightly_path} is missing an active manifest audit command")
    if not _uses_planner(nightly_commands, "plan"):
        errors.append(f"{nightly_path} is missing an active manifest plan command")
    if not _runs_planner_suite(nightly_commands, matrix_pattern):
        errors.append(
            f"{nightly_path} is missing the active scheduled suite runner command"
        )
    for forbidden in ("docker", "run_data_ingestion_it.sh", "manual_external"):
        if forbidden in nightly:
            errors.append(
                f"{nightly_path} contains forbidden external-IT fragment: {forbidden}"
            )
    return errors


def audit_repository(manifest: Manifest) -> list[str]:
    """Return every manifest, inventory, budget, marker, and workflow violation."""
    errors: list[str] = []
    default_owners = [
        suite.suite_id for suite in manifest.suites if suite.default_test_owner
    ]
    if len(default_owners) != 1:
        errors.append(
            "manifest must define exactly one default test owner; found "
            + repr(default_owners)
        )

    fast_budget = manifest.budgets["ci_fast_suite_seconds"]
    scheduled_budget = manifest.budgets["ci_scheduled_suite_seconds"]
    scheduled_total = 0
    for suite in manifest.suites:
        if suite.tier in CI_TIERS and not suite.ci_allowed:
            errors.append(f"CI suite {suite.suite_id} must set ci_allowed=true")
        if suite.tier not in CI_TIERS and suite.ci_allowed:
            errors.append(f"non-CI suite {suite.suite_id} must set ci_allowed=false")
        if suite.tier == "ci_fast" and suite.budget_seconds > fast_budget:
            errors.append(f"ci_fast suite {suite.suite_id} exceeds {fast_budget}s")
        if suite.tier == "ci_scheduled":
            scheduled_total += suite.budget_seconds
            if suite.budget_seconds > scheduled_budget:
                errors.append(
                    f"ci_scheduled suite {suite.suite_id} exceeds {scheduled_budget}s"
                )
        if suite.tier in CI_TIERS and set(suite.requires) & FORBIDDEN_CI_REQUIREMENTS:
            errors.append(
                f"CI suite {suite.suite_id} declares forbidden infrastructure: "
                + ", ".join(sorted(set(suite.requires) & FORBIDDEN_CI_REQUIREMENTS))
            )
        if suite.tier in CI_TIERS | {"manual_external"} and not suite.entrypoint:
            errors.append(f"executable suite {suite.suite_id} has no entrypoint")
        if suite.tier in CI_TIERS and suite.entrypoint != CI_PYTEST_ENTRYPOINT:
            errors.append(
                f"CI suite {suite.suite_id} must use the controlled pytest entrypoint"
            )
        if suite.tier in CI_TIERS:
            selection_error = _ci_suite_selection_error(manifest, suite)
            if selection_error is not None:
                errors.append(selection_error)
        for marker in suite.markers:
            allowed_tiers = MARKER_ALLOWED_TIERS.get(marker)
            if allowed_tiers is not None and suite.tier not in allowed_tiers:
                errors.append(
                    f"suite {suite.suite_id} marker {marker} is incompatible with "
                    f"tier {suite.tier}"
                )
        if suite.tier == "quarantine" and suite.workflow != "none":
            errors.append(f"quarantine suite {suite.suite_id} must use workflow=none")
        for relative_path in suite.test_paths:
            if not (manifest.root / relative_path).exists():
                errors.append(
                    f"suite {suite.suite_id} references missing path: {relative_path}"
                )
    if scheduled_total > manifest.budgets["nightly_total_seconds"]:
        errors.append(
            f"scheduled suite budget total {scheduled_total}s exceeds nightly limit "
            f"{manifest.budgets['nightly_total_seconds']}s"
        )

    test_files = sorted(
        path.relative_to(manifest.root).as_posix()
        for path in (manifest.root / "tests").glob("**/test_*.py")
    )
    for relative_path in test_files:
        owners = _owning_suites(manifest, relative_path)
        declared_markers = _declared_markers(manifest.root / relative_path)
        if len(owners) > 1:
            errors.append(
                f"test module {relative_path} has multiple explicit owners: "
                + ", ".join(suite.suite_id for suite in owners)
            )
        if _requires_explicit_owner(manifest.root, relative_path) and not owners:
            errors.append(
                f"integration-sensitive test module lacks an explicit suite: {relative_path}"
            )
        if len(owners) == 1:
            owner = owners[0]
            for marker in sorted(declared_markers):
                allowed_tiers = MARKER_ALLOWED_TIERS.get(marker)
                if allowed_tiers is not None and owner.tier not in allowed_tiers:
                    errors.append(
                        f"test module {relative_path} marker {marker} is incompatible "
                        f"with owner tier {owner.tier}"
                    )
        for marker in sorted(declared_markers & SHARED_SELECTOR_MARKERS):
            selectors = _shared_marker_selectors(manifest, relative_path, marker)
            if len(selectors) != 1:
                errors.append(
                    f"test module {relative_path} marker {marker} must have exactly "
                    "one controlled CI selector; found "
                    + repr([suite.suite_id for suite in selectors])
                )
        try:
            suite_for_test_path(manifest, relative_path)
        except ManifestError as exc:
            errors.append(str(exc))

    runners = sorted(
        path.relative_to(manifest.root).as_posix()
        for path in (manifest.root / "scripts").glob("run_*_it.sh")
    )
    for runner in runners:
        owners = _owning_suites(manifest, runner)
        if len(owners) != 1:
            errors.append(
                f"IT runner {runner} must have exactly one explicit owner; found "
                + repr([suite.suite_id for suite in owners])
            )
        elif owners[0].tier != "manual_external":
            errors.append(f"IT runner {runner} must be manual_external")

    try:
        import tomllib

        pyproject = tomllib.loads(
            (manifest.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        registered = "\n".join(
            pyproject["tool"]["pytest"]["ini_options"].get("markers", [])
        )
        addopts = pyproject["tool"]["pytest"]["ini_options"].get("addopts", "")
        for marker in MARKER_ALLOWED_TIERS:
            if f"{marker}:" not in registered:
                errors.append(f"pytest marker is not registered: {marker}")
        for expression in ("not manual_it", "not quarantine"):
            if expression not in addopts:
                errors.append(f"default pytest addopts is missing {expression!r}")
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"cannot validate pytest marker policy: {exc}")

    errors.extend(_workflow_errors(manifest))
    return errors


def _domains_for_path(manifest: Manifest, path: str) -> tuple[set[str], str | None]:
    for rule in manifest.path_rules:
        if any(path_matches(path, pattern) for pattern in rule["patterns"]):
            return set(rule["domains"]), str(rule["name"])
    return set(), None


def _suite_triggered(suite: Suite, changed_paths: Sequence[str]) -> bool:
    return any(
        path_matches(path, pattern)
        for path in changed_paths
        for pattern in suite.trigger_paths
    )


def build_plan(
    manifest: Manifest,
    *,
    event: str,
    mode: str,
    changed_paths: Sequence[str],
) -> dict[str, Any]:
    """Build a deterministic CI matrix and external-validation impact report."""
    normalized_paths = tuple(
        sorted({_normalize_path(path) for path in changed_paths if path})
    )
    if mode == "scheduled":
        selected = {
            tier: [suite.suite_id for suite in manifest.suites if suite.tier == tier]
            if tier == "ci_scheduled"
            else []
            for tier in TIERS
        }
        scheduled = [manifest.suite(suite_id) for suite_id in selected["ci_scheduled"]]
        return {
            "event": event,
            "mode": mode,
            "changed_paths": list(normalized_paths),
            "matched_rules": {},
            "unknown_paths": [],
            "selected": selected,
            "matrix": {"include": [{"suite": suite.suite_id} for suite in scheduled]},
            "run_unit": False,
            "run_docs": False,
        }
    if mode != "pr":
        raise ManifestError(f"unknown planning mode: {mode}")

    select_all = event == "merge_group"
    domains: set[str] = set()
    unknown_paths: list[str] = []
    matched_rules: dict[str, str] = {}
    for path in normalized_paths:
        path_domains, rule_name = _domains_for_path(manifest, path)
        if rule_name is None:
            unknown_paths.append(path)
        else:
            domains.update(path_domains)
            matched_rules[path] = rule_name
    if not normalized_paths and event not in {"merge_group"}:
        unknown_paths.append("<unresolved-diff>")
    fallback_all = select_all or bool(unknown_paths)

    selected: dict[str, list[str]] = {tier: [] for tier in TIERS}
    for suite in manifest.suites:
        is_selected = False
        if suite.tier == "ci_fast":
            is_selected = (
                suite.always_run
                or fallback_all
                or bool(set(suite.trigger_domains) & domains)
            )
        elif suite.tier in {"manual_external", "quarantine"}:
            is_selected = fallback_all or _suite_triggered(suite, normalized_paths)
        if is_selected:
            selected[suite.tier].append(suite.suite_id)

    fast = [manifest.suite(suite_id) for suite_id in selected["ci_fast"]]
    matrix_suites = [suite for suite in fast if suite.workflow == "matrix"]
    return {
        "event": event,
        "mode": mode,
        "changed_paths": list(normalized_paths),
        "matched_rules": matched_rules,
        "unknown_paths": unknown_paths,
        "selected": selected,
        "matrix": {"include": [{"suite": suite.suite_id} for suite in matrix_suites]},
        "run_unit": any(suite.workflow == "unit" for suite in fast),
        "run_docs": any(suite.workflow == "docs" for suite in fast),
    }


def git_changed_paths(root: Path, base: str, head: str) -> tuple[list[str], bool]:
    """Resolve changed paths, returning a fail-safe flag when Git cannot do so."""
    if not base or not head or set(base) == {"0"}:
        return [], False
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACDMR", base, head],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [], False
    return [line for line in result.stdout.splitlines() if line], True


def external_report(manifest: Manifest, plan: dict[str, Any]) -> str:
    """Render external and quarantined impacts without executable CI commands."""
    lines = ["## Test impact plan", ""]
    changed = plan["changed_paths"]
    lines.append(f"Changed paths: {len(changed)}")
    if plan["unknown_paths"]:
        lines.append(
            "Fail-safe fallback: "
            + ", ".join(str(path) for path in plan["unknown_paths"])
        )
    lines.extend(("", "### CI-safe suites", ""))
    ci_suites = plan["selected"]["ci_fast"] + plan["selected"]["ci_scheduled"]
    if ci_suites:
        lines.extend(f"- `{suite_id}`" for suite_id in ci_suites)
    else:
        lines.append("- None")

    lines.extend(("", "### Required external validation", ""))
    manual = plan["selected"]["manual_external"]
    if manual:
        for suite_id in manual:
            suite = manifest.suite(suite_id)
            command = " ".join((*suite.entrypoint, *suite.args))
            lines.append(
                f"- `{suite_id}` ({suite.owner}): {suite.rationale} "
                f"Declared entry point: `{command}`."
            )
        lines.append("")
        lines.append(
            "These commands are a validation ledger requirement, not CI execution "
            "authorization."
        )
    else:
        lines.append("- None")

    lines.extend(("", "### Quarantine impacts", ""))
    quarantined = plan["selected"]["quarantine"]
    if quarantined:
        for suite_id in quarantined:
            suite = manifest.suite(suite_id)
            lines.append(f"- `{suite_id}`: {suite.rationale}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(path: Path, plan: dict[str, Any]) -> None:
    values = {
        "run_unit": str(plan["run_unit"]).lower(),
        "run_docs": str(plan["run_docs"]).lower(),
        "run_matrix": str(bool(plan["matrix"]["include"])).lower(),
        "matrix": json.dumps(plan["matrix"], separators=(",", ":")),
        "manual_count": str(len(plan["selected"]["manual_external"])),
        "quarantine_count": str(len(plan["selected"]["quarantine"])),
    }
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            output.write(f"{name}={value}\n")


def _pytest_skips(path: Path) -> int:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        return int(root.attrib.get("skipped", "0"))
    return sum(
        int(suite.attrib.get("skipped", "0")) for suite in root.findall("testsuite")
    )


def run_suite(manifest: Manifest, suite_id: str, *, prepare: bool) -> int:
    """Execute one CI-allowed suite with argv-only commands and a hard budget."""
    suite = manifest.suite(suite_id)
    if suite.tier not in CI_TIERS or not suite.ci_allowed:
        raise ManifestError(
            f"suite {suite_id!r} is {suite.tier} and cannot run through the CI runner"
        )
    if suite.entrypoint != CI_PYTEST_ENTRYPOINT:
        raise ManifestError(
            f"suite {suite_id!r} does not use the controlled pytest entrypoint"
        )
    selection_error = _ci_suite_selection_error(manifest, suite)
    if selection_error is not None:
        raise ManifestError(selection_error)
    if prepare:
        sync = ["uv", "sync"]
        if suite.all_extras:
            sync.append("--all-extras")
        else:
            for extra in suite.extras:
                sync.extend(("--extra", extra))
        sync.append("--locked")
        completed = subprocess.run(sync, cwd=manifest.root, check=False)
        if completed.returncode != 0:
            return completed.returncode

    command = [
        "uv",
        "run",
        "--locked",
        "--no-sync",
        *suite.entrypoint,
        *suite.args,
    ]
    try:
        if suite.forbid_skips:
            with tempfile.TemporaryDirectory(
                prefix=f"tributo-{suite.suite_id}-"
            ) as tmp:
                report = Path(tmp) / "junit.xml"
                completed = subprocess.run(
                    [*command, f"--junitxml={report}"],
                    cwd=manifest.root,
                    check=False,
                    timeout=suite.budget_seconds,
                )
                if completed.returncode == 0:
                    try:
                        skipped = _pytest_skips(report) if report.is_file() else None
                    except (ET.ParseError, OSError, ValueError) as exc:
                        print(
                            f"suite {suite.suite_id} produced invalid JUnit evidence: "
                            f"{exc}",
                            file=sys.stderr,
                        )
                        return 1
                    if skipped is None or skipped > 0:
                        print(
                            f"suite {suite.suite_id} skipped required tests or produced "
                            "no JUnit evidence",
                            file=sys.stderr,
                        )
                        return 1
                return completed.returncode
        completed = subprocess.run(
            command,
            cwd=manifest.root,
            check=False,
            timeout=suite.budget_seconds,
        )
        return completed.returncode
    except subprocess.TimeoutExpired:
        print(
            f"suite {suite.suite_id} exceeded its {suite.budget_seconds}s budget",
            file=sys.stderr,
        )
        return 124


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="validate inventory, budgets, and workflows")

    plan = subparsers.add_parser("plan", help="select budgeted and external suites")
    plan.add_argument("--event", required=True)
    plan.add_argument("--mode", choices=("pr", "scheduled"), default="pr")
    plan.add_argument("--base", default="")
    plan.add_argument("--head", default="")
    plan.add_argument("--changed-path", action="append", default=[])
    plan.add_argument("--github-output", type=Path)
    plan.add_argument("--summary", type=Path)
    plan.add_argument("--json-output", type=Path)

    run = subparsers.add_parser("run", help="run one CI-allowed suite")
    run.add_argument("--suite", required=True)
    run.add_argument("--prepare", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.root, args.manifest)
        errors = audit_repository(manifest)
        if errors:
            print("CI test policy audit failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        if args.command == "audit":
            print(
                f"CI test policy audit passed: {len(manifest.suites)} suites, "
                f"{len(manifest.path_rules)} ordered path rules"
            )
            return 0
        if args.command == "run":
            return run_suite(manifest, args.suite, prepare=args.prepare)

        changed_paths = list(args.changed_path)
        if not changed_paths and args.mode == "pr" and args.event != "merge_group":
            changed_paths, resolved = git_changed_paths(
                manifest.root, args.base, args.head
            )
            if not resolved:
                changed_paths = []
        plan = build_plan(
            manifest,
            event=args.event,
            mode=args.mode,
            changed_paths=changed_paths,
        )
        rendered = json.dumps(plan, indent=2, sort_keys=True)
        print(rendered)
        if args.json_output:
            args.json_output.write_text(rendered + "\n", encoding="utf-8")
        if args.github_output:
            _write_outputs(args.github_output, plan)
        if args.summary:
            with args.summary.open("a", encoding="utf-8") as summary:
                summary.write(external_report(manifest, plan))
        return 0
    except (ManifestError, OSError) as exc:
        print(f"CI test policy error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
