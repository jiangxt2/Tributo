"""Contracts for the declarative CI test policy and impact planner."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from scripts import ci_test_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize(
    ("branch", "accepted"),
    [
        ("master", True),
        ("release-1.1.0", True),
        ("release-1.1.1", True),
        ("version-policy", False),
        ("feature/example", False),
    ],
)
def test_release_branch_uses_existing_ci_workflow(
    event: str, branch: str, accepted: bool
) -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/pr-test-suite.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    patterns = workflow["on"][event]["branches"]

    assert any(fnmatchcase(branch, pattern) for pattern in patterns) is accepted


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


@pytest.mark.parametrize(
    "path",
    [
        "src/tributo/_common/logging.py",
        "src/tributo/_common/serve_utils.py",
        "src/tributo/inference/kernel.py",
        "src/tributo/integrations/algorithm_runtimes/ray_train_torch.py",
    ],
)
def test_logging_changes_select_existing_runtime_environment_gate(manifest, path):
    plan = ci_test_plan.build_plan(
        manifest, event="pull_request", mode="pr", changed_paths=[path]
    )
    assert "ray-runtime-environment" in plan["selected"]["manual_external"]


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
        if runner_path == "scripts/run_database_filter_it.sh":
            assert suite.suite_id == "database-filter"
            assert suite.tier == "ci_database"
            assert suite.entrypoint == ci_test_plan.DATABASE_ENTRYPOINT
            assert suite.workflow == "database"
        else:
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


@pytest.mark.parametrize(
    "path",
    (
        "pyproject.toml",
        "uv.lock",
        "tests/integration/test_dowhy_scipy_compat.py",
    ),
)
def test_dependency_change_selects_real_causal_ci_suite(
    manifest: ci_test_plan.Manifest, path: str
) -> None:
    plan = ci_test_plan.build_plan(
        manifest, event="pull_request", mode="pr", changed_paths=[path]
    )

    assert "causal-dependency-contract" in plan["selected"]["ci_fast"]
    assert {"suite": "causal-dependency-contract"} in plan["matrix"]["include"]
    suite = manifest.suite("causal-dependency-contract")
    assert suite.extras == ("dev", "causal")
    assert suite.forbid_skips
    assert plan["run_unit"] is True
    assert plan["run_docs"] is (path != "tests/integration/test_dowhy_scipy_compat.py")


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

    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", unexpected_run)
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

    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", unexpected_run)
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

    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", unexpected_run)
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

    def completed(command: list[str], **kwargs: object) -> int:
        calls.append((command, kwargs))
        report_argument = next(
            argument for argument in command if argument.startswith("--junitxml=")
        )
        if junit is not None:
            Path(report_argument.split("=", 1)[1]).write_text(junit, encoding="utf-8")
        return 0

    monkeypatch.setattr(ci_test_plan, "_run_ci_stage", completed)

    assert ci_test_plan.run_suite(manifest, "policy", prepare=False) == expected
    assert len(calls) == 1
    assert "env" not in calls[0][1]


def test_preparation_and_execution_share_one_deadline(
    manifest: ci_test_plan.Manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [10.0]
    waits: list[float] = []
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(ci_test_plan.time, "monotonic", lambda: now[0])

    def start(command: list[str], **kwargs: object) -> Mock:
        calls.append((command, kwargs))
        process = Mock()

        def wait(*, timeout: float) -> int:
            waits.append(timeout)
            now[0] += 2.0
            return 0

        process.wait.side_effect = wait
        return process

    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", start)
    bounded = _replace_suite(manifest, "unit", budget_seconds=5)
    assert ci_test_plan.run_suite(bounded, "unit", prepare=True) == 0
    assert waits == [5.0, 3.0]
    assert calls[0][0] == ["uv", "sync", "--extra", "dev", "--locked"]
    assert calls[1][0][0:3] == ["uv", "run", "--locked"]
    assert all("env" not in kwargs for _, kwargs in calls)
    assert all(
        kwargs["start_new_session"] == (os.name == "posix") for _, kwargs in calls
    )


def test_relative_log_directory_is_shared_with_the_child_working_directory(
    manifest: ci_test_plan.Manifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    def completed(command: list[str], **kwargs: object) -> int:
        assert kwargs["root"] == ROOT
        assert kwargs["log_dir"] == tmp_path / "logs"
        report = next(
            arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")
        )
        assert Path(report).is_absolute()
        Path(report).parent.mkdir()
        Path(report).write_text('<testsuite tests="1" skipped="0"/>')
        return 0

    monkeypatch.setattr(ci_test_plan, "_run_ci_stage", completed)
    assert (
        ci_test_plan.run_suite(manifest, "policy", prepare=False, log_dir=Path("logs"))
        == 0
    )


@pytest.mark.parametrize("status", [0, 7])
def test_no_test_process_after_failed_or_exhausted_preparation(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
) -> None:
    now = [10.0]
    monkeypatch.setattr(ci_test_plan.time, "monotonic", lambda: now[0])
    process = Mock()

    def wait(*, timeout: float) -> int:
        now[0] += timeout
        return status

    process.wait.side_effect = wait
    start = Mock(return_value=process)
    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", start)
    bounded = _replace_suite(manifest, "unit", budget_seconds=1)
    result = ci_test_plan.run_suite(bounded, "unit", prepare=True, log_dir=tmp_path)
    assert result == (status or 124)
    assert start.call_count == 1
    assert not (tmp_path / "tests.log").exists()
    events = [
        json.loads(line)
        for line in (tmp_path / "stages.jsonl").read_text().splitlines()
    ]
    assert events[0]["stage"] == "prepare"
    assert events[-1]["state"] == ("finished" if status else "timed_out")


@pytest.mark.parametrize("prepare", [True, False])
def test_stage_timeout_is_nonzero_and_cleans_the_owned_process(
    manifest: ci_test_plan.Manifest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepare: bool,
) -> None:
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired("controlled", 1)
    start = Mock(return_value=process)
    stop = Mock()
    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", start)
    monkeypatch.setattr(ci_test_plan, "_stop_ci_process", stop)
    assert (
        ci_test_plan.run_suite(manifest, "unit", prepare=prepare, log_dir=tmp_path)
        == 124
    )
    assert start.call_count == 1
    stop.assert_called_once_with(process)
    events = [
        json.loads(line)
        for line in (tmp_path / "stages.jsonl").read_text().splitlines()
    ]
    assert events[-1]["state"] == "timed_out"
    assert events[-1]["stage"] == ("prepare" if prepare else "tests")


def test_interrupted_stage_preserves_interruption_after_cleanup(
    manifest: ci_test_plan.Manifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = Mock()
    process.wait.side_effect = KeyboardInterrupt
    stop = Mock()
    monkeypatch.setattr(ci_test_plan.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(ci_test_plan, "_stop_ci_process", stop)
    with pytest.raises(KeyboardInterrupt):
        ci_test_plan.run_suite(manifest, "unit", prepare=False, log_dir=tmp_path)
    stop.assert_called_once_with(process)
    assert '"state": "interrupted"' in (tmp_path / "stages.jsonl").read_text()


def test_stage_retains_both_streams_and_failure_status(tmp_path: Path) -> None:
    status = ci_test_plan._run_ci_stage(
        [
            sys.executable,
            "-c",
            "import sys; print('last case'); print('details', file=sys.stderr); sys.exit(7)",
        ],
        root=ROOT,
        suite_id="unit",
        stage="tests",
        deadline=time.monotonic() + 5,
        log_dir=tmp_path,
    )
    assert status == 7
    text = (tmp_path / "tests.log").read_text()
    assert "last case" in text and "details" in text
    assert '"returncode": 7' in (tmp_path / "stages.jsonl").read_text()


def test_stage_start_failure_is_visible_and_not_success(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ci_test_plan._run_ci_stage(
            [str(tmp_path / "missing-installer")],
            root=ROOT,
            suite_id="unit",
            stage="prepare",
            deadline=time.monotonic() + 5,
            log_dir=tmp_path,
        )
    assert '"state": "start_failed"' in (tmp_path / "stages.jsonl").read_text()


def test_timeout_retains_the_child_thread_dump(tmp_path: Path) -> None:
    status = ci_test_plan._run_ci_stage(
        [
            sys.executable,
            "-u",
            "-c",
            "import faulthandler,time; faulthandler.dump_traceback_later(0.1); print('last case'); time.sleep(60)",
        ],
        root=ROOT,
        suite_id="unit",
        stage="tests",
        deadline=time.monotonic() + 1,
        log_dir=tmp_path,
    )
    assert status == 124
    text = (tmp_path / "tests.log").read_text()
    assert "last case" in text and "Timeout" in text and "Thread" in text


@pytest.mark.skipif(os.name != "posix", reason="CI process groups use POSIX sessions")
@pytest.mark.parametrize(
    ("parent_ignores_term", "use_uv"), [(False, False), (True, False), (True, True)]
)
def test_timeout_removes_term_resistant_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_ignores_term: bool,
    use_uv: bool,
) -> None:
    monkeypatch.setattr(ci_test_plan, "_PROCESS_STOP_GRACE_SECONDS", 0.1)
    ready = tmp_path / "child-ready"
    pids = tmp_path / "pids.json"
    child = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    parent = (
        "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
        + (
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            if parent_ignores_term
            else ""
        )
        + f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        + f"ready=Path({str(ready)!r}); "
        + "\nwhile not ready.exists(): time.sleep(0.01)\n"
        + f"Path({str(pids)!r}).write_text(json.dumps({{'parent':os.getpid(),'group':os.getpgrp(),'child':p.pid}})); "
        + "time.sleep(60)"
    )
    command = [sys.executable, "-c", parent]
    if use_uv:
        uv = shutil.which("uv")
        assert uv is not None, "The CI controller requires uv on PATH"
        command = [
            uv,
            "run",
            "--no-cache",
            "--offline",
            "--no-project",
            "--no-python-downloads",
            "--python",
            sys.executable,
            *command,
        ]
    owned_group: int | None = None

    def running_owned_pids() -> list[str]:
        identities = json.loads(pids.read_text())
        result = subprocess.run(
            [
                "ps",
                "-p",
                ",".join(str(pid) for pid in set(identities.values())),
                "-o",
                "pid=,stat=",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode in (0, 1) and not result.stderr, result.stderr
        # Reparented zombies are already dead; their reaper is outside this stage.
        return [
            line
            for line in result.stdout.splitlines()
            if not line.split()[1].startswith("Z")
        ]

    try:
        assert (
            ci_test_plan._run_ci_stage(
                command,
                root=ROOT,
                suite_id="unit",
                stage="tests",
                deadline=time.monotonic() + 2,
                log_dir=tmp_path,
            )
            == 124
        )
        owned_group = json.loads(pids.read_text())["group"]
        assert owned_group != os.getpgrp()
        deadline = time.monotonic() + 2
        while running_owned_pids() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not running_owned_pids(), "owned descendants still run after timeout"
    finally:
        if owned_group is None and pids.exists():
            owned_group = json.loads(pids.read_text())["group"]
        if owned_group is not None and running_owned_pids():
            try:
                os.killpg(owned_group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_unit_workflow_retains_diagnostics_without_changing_its_budget() -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/pr-test-suite.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    unit = workflow["jobs"]["unit-tests"]
    assert unit["timeout-minutes"] == "20"
    assert unit["strategy"]["matrix"]["python-version"] == ["3.12", "3.13"]
    run = next(
        s for s in unit["steps"] if s.get("name") == "Run manifest-owned unit suite"
    )
    assert "--prepare --log-dir" in run["run"]
    assert run["env"]["PYTHONUNBUFFERED"] == "1"
    assert run["env"]["PYTEST_ADDOPTS"] == "-o faulthandler_timeout=60"
    upload = next(
        s for s in unit["steps"] if s.get("name") == "Upload unit diagnostics"
    )
    assert upload["if"] == "always()"
    assert upload["with"]["path"].endswith("unit-${{ matrix.python-version }}/")
    assert upload["with"]["if-no-files-found"] == "warn"


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


def test_dco_check_skips_merge_commits_but_requires_signed_off_by() -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/pr-test-suite.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    dco_step = next(
        step
        for step in workflow["jobs"]["dco"]["steps"]
        if step.get("name") == "Check Signed-off-by"
    )

    assert (
        "git log --no-merges --format='%H' \"$BASE_SHA..$HEAD_SHA\"" in dco_step["run"]
    )
    assert 'grep -qE "^Signed-off-by: [^<]+ <[^>]+>$"' in dco_step["run"]
