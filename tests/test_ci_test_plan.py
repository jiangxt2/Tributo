"""Contracts for the declarative CI test policy and impact planner."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import ci_test_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest() -> ci_test_plan.Manifest:
    return ci_test_plan.load_manifest(ROOT)


def _replace_suite(
    manifest: ci_test_plan.Manifest,
    suite_id: str,
    **changes: object,
) -> ci_test_plan.Manifest:
    return replace(
        manifest,
        suites=tuple(
            replace(suite, **changes) if suite.suite_id == suite_id else suite
            for suite in manifest.suites
        ),
    )


def test_repository_policy_audit_is_clean(
    manifest: ci_test_plan.Manifest,
) -> None:
    assert ci_test_plan.audit_repository(manifest) == []


def test_manifest_rejects_non_positive_suite_budget(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "ci" / "test-suites.json").read_text(encoding="utf-8"))
    payload["suites"][0]["budget_seconds"] = 0
    path = tmp_path / "test-suites.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ci_test_plan.ManifestError, match="positive when executable"):
        ci_test_plan.load_manifest(ROOT, path)


def test_every_test_module_and_it_runner_has_one_effective_owner(
    manifest: ci_test_plan.Manifest,
) -> None:
    for path in sorted((ROOT / "tests").glob("**/test_*.py")):
        relative_path = path.relative_to(ROOT).as_posix()
        assert ci_test_plan.suite_for_test_path(manifest, relative_path)

    runner_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").glob("run_*_it.sh")
    )
    assert runner_paths
    for runner_path in runner_paths:
        suite = ci_test_plan.suite_for_test_path(manifest, runner_path)
        assert suite.tier == "manual_external"
        assert suite.workflow == "external"


def test_dot_github_path_keeps_leading_dot_and_uses_first_rule(
    manifest: ci_test_plan.Manifest,
) -> None:
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["./.github/workflows/pr-test-suite.yml"],
    )

    assert plan["changed_paths"] == [".github/workflows/pr-test-suite.yml"]
    assert plan["matched_rules"][".github/workflows/pr-test-suite.yml"] == "ci-policy"
    assert plan["unknown_paths"] == []


def test_documentation_only_change_avoids_unit_and_storage_suites(
    manifest: ci_test_plan.Manifest,
) -> None:
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["README.md"],
    )

    assert plan["selected"]["ci_fast"] == ["policy", "documentation-api"]
    assert plan["run_unit"] is False
    assert plan["run_docs"] is True
    assert plan["selected"]["manual_external"] == []


def test_public_api_generator_change_selects_documentation_gate(
    manifest: ci_test_plan.Manifest,
) -> None:
    path = "tools/generate_public_api_reference.py"
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=[path],
    )

    assert plan["matched_rules"][path] == "documentation"
    assert plan["selected"]["ci_fast"] == ["policy", "documentation-api"]
    assert plan["run_unit"] is False
    assert plan["run_docs"] is True
    assert manifest.suite("documentation-api").extras == (
        "dev",
        "grpc",
        "training",
        "vector-index",
    )


def test_storage_change_selects_bounded_ci_and_reports_external_validation(
    manifest: ci_test_plan.Manifest,
) -> None:
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["src/tributo/exporting/manager.py"],
    )

    assert plan["selected"]["ci_fast"] == [
        "policy",
        "unit",
        "unit-integration-contracts",
        "documentation-api",
        "s3-contract",
    ]
    assert "model-export-cluster" in plan["selected"]["manual_external"]
    assert plan["run_unit"] is True
    assert plan["run_docs"] is True
    assert {entry["suite"] for entry in plan["matrix"]["include"]} == {
        "policy",
        "unit-integration-contracts",
        "s3-contract",
    }


def test_unknown_path_fails_safe_to_every_pr_relevant_suite(
    manifest: ci_test_plan.Manifest,
) -> None:
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["new-unclassified-area/value.txt"],
    )

    assert plan["unknown_paths"] == ["new-unclassified-area/value.txt"]
    for tier in ("ci_fast", "manual_external", "quarantine"):
        expected = [suite.suite_id for suite in manifest.suites if suite.tier == tier]
        assert plan["selected"][tier] == expected
    assert plan["selected"]["ci_scheduled"] == []


def test_merge_group_runs_all_fast_suites_without_running_external_suites(
    manifest: ci_test_plan.Manifest,
) -> None:
    plan = ci_test_plan.build_plan(
        manifest,
        event="merge_group",
        mode="pr",
        changed_paths=[],
    )

    expected_fast = [
        suite.suite_id for suite in manifest.suites if suite.tier == "ci_fast"
    ]
    assert plan["selected"]["ci_fast"] == expected_fast
    assert plan["unknown_paths"] == []
    assert plan["run_unit"] is True
    assert plan["run_docs"] is True


def test_scheduled_plan_contains_only_budgeted_scheduled_shards(
    manifest: ci_test_plan.Manifest,
) -> None:
    plan = ci_test_plan.build_plan(
        manifest,
        event="schedule",
        mode="scheduled",
        changed_paths=[],
    )

    scheduled_ids = [
        suite.suite_id for suite in manifest.suites if suite.tier == "ci_scheduled"
    ]
    assert plan["selected"]["ci_scheduled"] == scheduled_ids
    assert plan["matrix"] == {
        "include": [{"suite": suite_id} for suite_id in scheduled_ids]
    }
    assert all(
        not plan["selected"][tier]
        for tier in ("ci_fast", "manual_external", "quarantine")
    )
    assert (
        sum(manifest.suite(value).budget_seconds for value in scheduled_ids)
        <= (manifest.budgets["nightly_total_seconds"])
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "tests/integration/test_walking_skeleton.py",
            ("manual_it",),
        ),
        (
            "tests/registry/test_integration.py",
            ("manual_it", "quarantine"),
        ),
        ("tests/test_config.py", ()),
    ],
)
def test_collection_markers_come_from_the_owning_suite(
    manifest: ci_test_plan.Manifest,
    path: str,
    expected: tuple[str, ...],
) -> None:
    assert ci_test_plan.markers_for_test_path(manifest, path) == expected


def test_ci_runner_refuses_external_suite_before_spawning_process(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail(f"subprocess must not run: {args!r} {kwargs!r}")

    monkeypatch.setattr(ci_test_plan.subprocess, "run", unexpected_run)
    with pytest.raises(ci_test_plan.ManifestError, match="cannot run through"):
        ci_test_plan.run_suite(manifest, "data-ingestion-cluster", prepare=True)


def test_manifest_commands_are_argv_and_report_is_non_authorizing(
    manifest: ci_test_plan.Manifest,
) -> None:
    assert all(isinstance(suite.entrypoint, tuple) for suite in manifest.suites)
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["src/tributo/data/reader.py"],
    )
    report = ci_test_plan.external_report(manifest, plan)
    assert "Required external validation" in report
    assert "not CI execution authorization" in report


def test_ci_entrypoint_allowlist_is_enforced_before_spawning(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = _replace_suite(
        manifest,
        "policy",
        entrypoint=("bash", "-c"),
        args=("echo unsafe",),
    )

    assert any(
        "policy must use the controlled pytest entrypoint" in error
        for error in ci_test_plan.audit_repository(unsafe)
    )

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail(f"subprocess must not run: {args!r} {kwargs!r}")

    monkeypatch.setattr(ci_test_plan.subprocess, "run", unexpected_run)
    with pytest.raises(ci_test_plan.ManifestError, match="controlled pytest"):
        ci_test_plan.run_suite(unsafe, "policy", prepare=True)


@pytest.mark.parametrize("expression", ["manual_it", "manual_it and integration"])
def test_ci_marker_safety_is_enforced_before_spawning(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
    expression: str,
) -> None:
    unsafe = _replace_suite(
        manifest,
        "policy",
        args=("tests/", "-m", expression),
    )

    assert "CI suite policy can select manual_it tests" in (
        ci_test_plan.audit_repository(unsafe)
    )

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail(f"subprocess must not run: {args!r} {kwargs!r}")

    monkeypatch.setattr(ci_test_plan.subprocess, "run", unexpected_run)
    with pytest.raises(ci_test_plan.ManifestError, match="select manual_it"):
        ci_test_plan.run_suite(unsafe, "policy", prepare=True)


def test_governed_marker_modules_have_controlled_owners_or_selectors(
    manifest: ci_test_plan.Manifest,
    tmp_path: Path,
) -> None:
    shared_path = "tests/data/test_ds_parquet_csv.py"
    selectors = ci_test_plan._shared_marker_selectors(
        manifest, shared_path, "s3_contract"
    )
    assert [suite.suite_id for suite in selectors] == ["s3-contract"]
    assert "not s3_contract" in (
        ci_test_plan._pytest_marker_expression(manifest.suite("unit")) or ""
    )

    manual_test = tmp_path / "tests" / "test_manual_case.py"
    manual_test.parent.mkdir(parents=True)
    manual_test.write_text(
        "import pytest\npytestmark = pytest.mark.manual_it\n",
        encoding="utf-8",
    )
    assert ci_test_plan._requires_explicit_owner(tmp_path, "tests/test_manual_case.py")


def test_shared_s3_marker_requires_exactly_one_controlled_selector(
    manifest: ci_test_plan.Manifest,
) -> None:
    suite = manifest.suite("s3-contract")
    without_shared_path = _replace_suite(
        manifest,
        "s3-contract",
        args=tuple(
            argument
            for argument in suite.args
            if argument != "tests/data/test_ds_parquet_csv.py"
        ),
    )

    assert any(
        "tests/data/test_ds_parquet_csv.py marker s3_contract must have exactly "
        "one controlled CI selector" in error
        for error in ci_test_plan.audit_repository(without_shared_path)
    )

    marker_index = suite.args.index("-m")
    incompatible_expression = _replace_suite(
        manifest,
        "s3-contract",
        args=(
            *suite.args[: marker_index + 1],
            "s3_contract and ci_safe and not manual_it and not quarantine",
            *suite.args[marker_index + 2 :],
        ),
    )
    assert any(
        "tests/data/test_ds_parquet_csv.py marker s3_contract must have exactly "
        "one controlled CI selector" in error
        for error in ci_test_plan.audit_repository(incompatible_expression)
    )


def test_workflow_audit_rejects_wildcard_external_runner_in_yaml(
    manifest: ci_test_plan.Manifest,
    tmp_path: Path,
) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    source_dir = ROOT / ".github" / "workflows"
    for name in ("pr-test-suite.yml", "nightly-test.yml"):
        (workflow_dir / name).write_text(
            (source_dir / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (workflow_dir / "bypass.yaml").write_text(
        "jobs:\n  bypass:\n    steps:\n      - run: scripts/run_*_it.sh\n",
        encoding="utf-8",
    )

    errors = ci_test_plan._workflow_errors(replace(manifest, root=tmp_path))

    assert "GitHub Actions contains an external IT runner reference" in errors


def test_workflow_required_command_cannot_be_satisfied_by_comment(
    manifest: ci_test_plan.Manifest,
    tmp_path: Path,
) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    source_dir = ROOT / ".github" / "workflows"
    pr = (source_dir / "pr-test-suite.yml").read_text(encoding="utf-8")
    pr = pr.replace(
        "run: python3 scripts/ci_test_plan.py run --suite unit --prepare",
        "run: echo 'unit runner disabled'",
    )
    pr = pr.replace("  core-gate:\n", "  # core-gate:\n")
    (workflow_dir / "pr-test-suite.yml").write_text(pr, encoding="utf-8")
    (workflow_dir / "nightly-test.yml").write_text(
        (source_dir / "nightly-test.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    errors = ci_test_plan._workflow_errors(replace(manifest, root=tmp_path))

    assert (
        ".github/workflows/pr-test-suite.yml is missing the active unit suite "
        "runner command"
    ) in errors
    assert (
        ".github/workflows/pr-test-suite.yml is missing required core-gate job"
    ) in errors


def test_git_changed_paths_includes_deletions(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def changed(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="src/tributo/removed.py\ndocs/index.md\n",
            stderr="",
        )

    monkeypatch.setattr(ci_test_plan.subprocess, "run", changed)

    paths, resolved = ci_test_plan.git_changed_paths(ROOT, "base", "head")

    assert resolved is True
    assert paths == ["src/tributo/removed.py", "docs/index.md"]
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=paths,
    )
    assert plan["run_unit"] is True
    assert plan["run_docs"] is True
    assert commands == [
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACDMR",
            "base",
            "head",
        ]
    ]


def test_git_changed_paths_failure_is_unresolved_and_fails_safe(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 128, stdout="", stderr="bad ref")

    monkeypatch.setattr(ci_test_plan.subprocess, "run", failed)

    paths, resolved = ci_test_plan.git_changed_paths(ROOT, "base", "head")
    plan = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=paths,
    )

    assert resolved is False
    assert plan["unknown_paths"] == ["<unresolved-diff>"]
    assert plan["selected"]["ci_fast"] == [
        suite.suite_id for suite in manifest.suites if suite.tier == "ci_fast"
    ]


@pytest.mark.parametrize(
    ("junit", "expected"),
    [
        ('<testsuite tests="1" skipped="0"/>', 0),
        ('<testsuite tests="1" skipped="1"/>', 1),
        ("<testsuite>", 1),
        (None, 1),
    ],
)
def test_forbid_skips_requires_valid_skip_free_junit_evidence(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
    junit: str | None,
    expected: int,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def completed(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        report_argument = next(
            argument for argument in command if argument.startswith("--junitxml=")
        )
        if junit is not None:
            Path(report_argument.split("=", 1)[1]).write_text(junit, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ci_test_plan.subprocess, "run", completed)

    assert ci_test_plan.run_suite(manifest, "policy", prepare=False) == expected
    assert len(calls) == 1
    assert "env" not in calls[0][1]


def test_workflows_preserve_events_permissions_budgets_and_stable_gate() -> None:
    pr = (ROOT / ".github" / "workflows" / "pr-test-suite.yml").read_text(
        encoding="utf-8"
    )
    nightly = (ROOT / ".github" / "workflows" / "nightly-test.yml").read_text(
        encoding="utf-8"
    )

    for event in ("pull_request:", "push:", "merge_group:"):
        assert event in pr
    assert "concurrency:" in pr
    assert "cancel-in-progress: true" in pr
    assert "permissions:\n  contents: read" in pr
    assert "matrix: ${{ fromJSON(needs.test-plan.outputs.matrix) }}" in pr
    assert "name: core-gate" in pr
    assert "if: always()" in pr

    assert "schedule:" in nightly
    assert "workflow_dispatch:" in nightly
    assert "permissions:\n  contents: read" in nightly
    assert "timeout-minutes: 30" in nightly
    assert "matrix: ${{ fromJSON(needs.test-plan.outputs.matrix) }}" in nightly

    commands = ci_test_plan._workflow_run_commands(
        pr
    ) + ci_test_plan._workflow_run_commands(nightly)
    assert all("pytest " not in command for command in commands)
    combined = pr + nightly
    assert "timeout-minutes: 45" not in combined
    assert "timeout-minutes: 65" not in combined
    assert "timeout-minutes: 80" not in combined
