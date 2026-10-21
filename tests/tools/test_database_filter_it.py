"""Reject incomplete execution evidence and unsafe CI database ownership."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import yaml

from scripts import ci_test_plan
from tools import database_filter_it
from tools.database_filter_it import validate_evidence
from tools.tributo_it import TributoITError


def _evidence(directory: Path) -> None:
    (directory / "junit.xml").write_text(
        '<testsuites><testsuite tests="1"><testcase name="real-read"/></testsuite></testsuites>'
    )
    rows = [
        {
            "database": "doris",
            "mode": mode,
            "ray_nodes": 1,
            "baseline_rows": 10_000,
            "source_rows": 100,
            "final_rows": 100,
            "residual_ordinals": [],
            "queries": [
                {
                    "stage": "real_driver_execute_succeeded",
                    "sql": "SELECT id FROM fixture WHERE id >= 9900",
                }
            ],
        }
        for mode in ("mysql", "flight")
    ]
    (directory / "evidence.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )


def test_complete_evidence_is_accepted(tmp_path: Path) -> None:
    _evidence(tmp_path)
    validate_evidence(tmp_path, "doris")


@pytest.mark.parametrize("source", [None, "local-mirror", "canonical"])
def test_image_source_reaches_runtime_and_database_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    if source is None:
        monkeypatch.delenv("TRIBUTO_FILTER_IMAGE_SOURCE", raising=False)
    else:
        monkeypatch.setenv("TRIBUTO_FILTER_IMAGE_SOURCE", source)
    mode = database_filter_it.it.RuntimeBuildInputMode(source or "local-mirror")
    profile = object()
    prepared = SimpleNamespace(source="local")
    prepare = Mock(return_value=prepared)
    execute = Mock()
    monkeypatch.setattr(database_filter_it, "_image_diagnostics", Mock())
    monkeypatch.setattr(database_filter_it, "_run_database_filter", execute)
    monkeypatch.setattr(database_filter_it.it, "load_profile", lambda _: profile)
    monkeypatch.setattr(database_filter_it.it, "docker_platform", lambda: "linux/amd64")
    monkeypatch.setattr(database_filter_it.it, "prepare_runtime", prepare)

    database_filter_it.run_database_filter("clickhouse", log_dir=tmp_path)

    prepare.assert_called_once_with(
        profile, platform="linux/amd64", build_input_mode=mode
    )
    execute.assert_called_once_with(
        "clickhouse", log_dir=tmp_path.resolve(), prepared=prepared, image_source=mode
    )


def test_invalid_image_source_preserves_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIBUTO_FILTER_IMAGE_SOURCE", "invalid")
    prepare = Mock()
    monkeypatch.setattr(database_filter_it, "_image_diagnostics", Mock())
    monkeypatch.setattr(database_filter_it.it, "prepare_runtime", prepare)

    with pytest.raises(TributoITError, match="TRIBUTO_FILTER_IMAGE_SOURCE"):
        database_filter_it.run_database_filter("doris", log_dir=tmp_path)

    prepare.assert_not_called()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["passed"] is False
    assert result["phase"] == "preparation"
    assert "TRIBUTO_FILTER_IMAGE_SOURCE" in (tmp_path / "lifecycle.log").read_text()


@pytest.mark.parametrize("database", ["clickhouse", "doris"])
@pytest.mark.parametrize("source", list(database_filter_it.it.RuntimeBuildInputMode))
def test_database_service_images_use_the_selected_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database: str,
    source: database_filter_it.it.RuntimeBuildInputMode,
) -> None:
    images = {
        "clickhouse_image": "clickhouse/server:1@sha256:" + "a" * 64,
        "doris_fe_image": "apache/doris:fe-1@sha256:" + "b" * 64,
        "doris_be_image": "apache/doris:be-1@sha256:" + "c" * 64,
    }
    monkeypatch.setattr(database_filter_it.it, "_assert_project_absent", Mock())
    monkeypatch.setattr(
        database_filter_it.it, "_capture_image_diagnostic_baseline", Mock()
    )
    monkeypatch.setattr(
        database_filter_it.it, "_capture_container_diagnostic_baseline", Mock()
    )
    monkeypatch.setattr(
        database_filter_it.it,
        "load_profile",
        lambda _: SimpleNamespace(definition=images),
    )
    ensure = Mock()
    monkeypatch.setattr(database_filter_it.it, "ensure_digest_image", ensure)
    monkeypatch.setattr(database_filter_it.it, "_local_digest_reference", lambda x: x)
    monkeypatch.setattr(
        database_filter_it,
        "_available_subnet",
        Mock(side_effect=RuntimeError("stop after image preparation")),
    )

    with pytest.raises(RuntimeError, match="stop after image preparation"):
        database_filter_it._run_database_filter(
            database,
            log_dir=tmp_path,
            prepared=SimpleNamespace(),
            image_source=source,
        )

    names = (
        ("clickhouse_image",)
        if database == "clickhouse"
        else ("doris_fe_image", "doris_be_image")
    )
    assert ensure.call_args_list == [
        call(
            images[name],
            use_mirror=source
            == database_filter_it.it.RuntimeBuildInputMode.LOCAL_MIRROR,
        )
        for name in names
    ]


def test_database_workflow_selects_canonical_images() -> None:
    path = ci_test_plan.ROOT / ".github/workflows/database-filter.yml"
    workflow = yaml.safe_load(path.read_text())
    controlled_step = next(
        step
        for step in workflow["jobs"]["database"]["steps"]
        if "--suite database-filter" in step.get("run", "")
    )
    assert controlled_step["env"]["TRIBUTO_FILTER_IMAGE_SOURCE"] == "canonical"


@pytest.mark.parametrize("source", ["local", "concurrent-local", "registry"])
def test_runtime_cleanup_preserves_reused_images(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    run = Mock()
    monkeypatch.setattr(database_filter_it.it, "_run", run)
    database_filter_it._cleanup_built_runtime(SimpleNamespace(source=source))
    run.assert_not_called()


@pytest.mark.parametrize("failure", ["changed_tag", "in_use", None])
def test_new_runtime_cleanup_checks_identity_and_ownership(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from contextlib import nullcontext

    prepared = SimpleNamespace(
        source="build",
        image_id="sha256:owned",
        identity=SimpleNamespace(local_tag="tributo-filter-runtime:owned"),
    )
    monkeypatch.setattr(database_filter_it.it, "runtime_lock", lambda _: nullcontext())
    monkeypatch.setattr(
        database_filter_it.it,
        "_image_inspect",
        lambda _: {
            "Id": "sha256:other" if failure == "changed_tag" else "sha256:owned"
        },
    )
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [], 0, "busy" if failure == "in_use" else "", ""
        )
    )
    monkeypatch.setattr(database_filter_it.it, "_run", run)
    if failure:
        with pytest.raises(TributoITError):
            database_filter_it._cleanup_built_runtime(prepared)
        assert all("rm" not in call.args[0] for call in run.call_args_list)
    else:
        database_filter_it._cleanup_built_runtime(prepared)
        assert run.call_args.args == (
            ["docker", "image", "rm", prepared.identity.local_tag],
        )


def test_builtin_networks_without_ipam_configuration_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["network", "ls"]:
            output = "host\nbridge\n"
        else:
            output = json.dumps(
                [
                    {"IPAM": {"Config": None}},
                    {"IPAM": {"Config": [{"Subnet": "172.30.0.0/24"}]}},
                ]
            )
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(database_filter_it.it, "_run", run)
    assert str(database_filter_it._available_subnet()) == "172.30.1.0/24"


@pytest.mark.parametrize(
    "failure",
    [
        "skip",
        "no_tests",
        "missing_mode",
        "missing_query",
        "ray_filter",
        "multiple_nodes",
        "unfiltered_source",
    ],
)
def test_incomplete_proof_cannot_pass(tmp_path: Path, failure: str) -> None:
    _evidence(tmp_path)
    if failure == "skip":
        (tmp_path / "junit.xml").write_text(
            "<testsuites><testcase><skipped/></testcase></testsuites>"
        )
    elif failure == "no_tests":
        (tmp_path / "junit.xml").write_text("<testsuites/>")
    else:
        rows = [
            json.loads(line)
            for line in (tmp_path / "evidence.jsonl").read_text().splitlines()
        ]
        if failure == "missing_mode":
            rows.pop()
        elif failure == "missing_query":
            rows[0]["queries"] = []
        elif failure == "ray_filter":
            rows[0]["residual_ordinals"] = [0]
        elif failure == "multiple_nodes":
            rows[0]["ray_nodes"] = 2
        else:
            rows[0]["source_rows"] = 10_000
        (tmp_path / "evidence.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    with pytest.raises(TributoITError):
        validate_evidence(tmp_path, "doris")


def test_database_ci_is_selected_for_bindings_but_not_docs() -> None:
    manifest = ci_test_plan.load_manifest()
    changed = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["src/tributo/data/bindings/ray_doris.py"],
    )
    docs = ci_test_plan.build_plan(
        manifest,
        event="pull_request",
        mode="pr",
        changed_paths=["docs/data/user-guides/read-data.md"],
    )
    assert changed["selected"]["ci_database"] == ["database-filter"]
    assert docs["selected"]["ci_database"] == []
    assert manifest.suite("data-ingestion-cluster").tier == "manual_external"


@pytest.mark.parametrize(
    "requires", [("docker", "multi_node_ray"), ("docker", "external_credentials")]
)
def test_database_tier_cannot_authorize_unowned_infrastructure(
    requires: tuple[str, ...],
) -> None:
    suite = replace(
        ci_test_plan.load_manifest().suite("database-filter"), requires=requires
    )
    assert ci_test_plan._database_suite_error(suite)


def test_database_workflow_requires_gate_and_failure_evidence(tmp_path: Path) -> None:
    manifest = ci_test_plan.load_manifest()
    directory = tmp_path / ".github/workflows"
    directory.mkdir(parents=True)
    for source in (manifest.root / ".github/workflows").glob("*.yml"):
        content = source.read_text()
        if source.name == "database-filter.yml":
            content = content.replace("\n    if: always()", "\n    if: success()")
        (directory / source.name).write_text(content)
    errors = ci_test_plan._workflow_errors(replace(manifest, root=tmp_path))
    assert any(
        "database-filter" in error and "if: always()" in error for error in errors
    )
