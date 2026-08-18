"""Unit tests for the Docker IT lifecycle helper."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import pytest

from tools import tributo_it
from tributo._common.runtime_env import find_project_root


def _write_profile(root: Path) -> tributo_it.RuntimeProfile:
    integration_dir = root / "tests" / "integrations"
    integration_dir.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n")
    (root / "uv.lock").write_text("version = 1\n")
    dockerfile = integration_dir / "Dockerfile.data-ingestion"
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = "c" * 64
    digest_d = "d" * 64
    dockerfile.write_text(
        f"FROM example:1@sha256:{digest_a}\nRUN uv sync --extra data-daft\n"
    )
    (root / ".dockerignore").write_text("root-only\n")
    dockerfile.with_name(f"{dockerfile.name}.dockerignore").write_text("src/**\n")
    definition = {
        "base_image": f"example:1@sha256:{digest_a}",
        "dockerfile": "tests/integrations/Dockerfile.data-ingestion",
        "extras": ["data-daft"],
        "minio_image": f"example:2@sha256:{digest_b}",
        "python_version": "3.12",
        "runtime_repository": "tributo-it-runtime",
        "tool_image": f"python:3@sha256:{digest_c}",
        "uv_image": f"uv:1@sha256:{digest_d}",
        "version_contract": {"daft_prefix": "0.7.", "ray": "2.55.1"},
    }
    (integration_dir / "runtime-profiles.json").write_text(
        json.dumps({"schema_version": 1, "profiles": {"data-ingestion": definition}})
    )
    return tributo_it.load_profile("data-ingestion", root=root)


def _hold_runtime_lock(
    identity: tributo_it.RuntimeIdentity,
    ready: multiprocessing.synchronize.Event,
) -> None:
    with tributo_it.runtime_lock(identity):
        ready.set()
        time.sleep(0.4)


def _prepare_runtime_worker(
    profile: tributo_it.RuntimeProfile,
    ready: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    ready.wait(timeout=2)
    prepared = tributo_it.prepare_runtime(profile, platform="linux/amd64")
    results.put(prepared.source)


def test_runtime_key_ignores_source_but_tracks_dependency_inputs(
    tmp_path: Path,
) -> None:
    profile = _write_profile(tmp_path)
    source = tmp_path / "src" / "example.py"
    source.write_text("value = 1\n")
    original = tributo_it.runtime_identity(profile, "linux/amd64")

    source.write_text("value = 2\n")
    source_changed = tributo_it.runtime_identity(profile, "linux/amd64")
    assert source_changed.runtime_key == original.runtime_key

    (tmp_path / ".dockerignore").write_text("root-changed\n")
    unused_root_ignore_changed = tributo_it.runtime_identity(profile, "linux/amd64")
    assert unused_root_ignore_changed.runtime_key == original.runtime_key

    dockerfile_ignore = profile.dockerfile.with_name(
        f"{profile.dockerfile.name}.dockerignore"
    )
    dockerfile_ignore.write_text("src/**\ntests/**\n")
    ignore_changed = tributo_it.runtime_identity(profile, "linux/amd64")
    assert ignore_changed.runtime_key != original.runtime_key

    (tmp_path / "uv.lock").write_text("version = 2\n")
    dependency_changed = tributo_it.runtime_identity(profile, "linux/amd64")
    assert dependency_changed.runtime_key != ignore_changed.runtime_key

    profile.definition["minio_image"] = "example:3@sha256:changed"
    infrastructure_changed = tributo_it.runtime_identity(profile, "linux/amd64")
    assert infrastructure_changed.runtime_key == dependency_changed.runtime_key


def test_profile_requires_named_digest_pinned_minio_image(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path)
    profile_file = tmp_path / "tests" / "integrations" / "runtime-profiles.json"
    payload = json.loads(profile_file.read_text())
    del payload["profiles"][profile.name]["minio_image"]
    profile_file.write_text(json.dumps(payload))

    with pytest.raises(tributo_it.TributoITError, match="missing fields.*minio_image"):
        tributo_it.load_profile(profile.name, root=tmp_path)


def test_profile_rejects_mutable_minio_image(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path)
    profile_file = tmp_path / "tests" / "integrations" / "runtime-profiles.json"
    payload = json.loads(profile_file.read_text())
    payload["profiles"][profile.name]["minio_image"] = "minio/minio:latest"
    profile_file.write_text(json.dumps(payload))

    with pytest.raises(tributo_it.TributoITError, match="readable tag@sha256"):
        tributo_it.load_profile(profile.name, root=tmp_path)


def test_domestic_mirror_reference_maps_supported_registries() -> None:
    digest = "sha256:" + "a" * 64

    assert tributo_it._domestic_mirror_reference(f"minio/minio:latest@{digest}") == (
        f"docker.m.daocloud.io/minio/minio:latest@{digest}"
    )
    assert (
        tributo_it._domestic_mirror_reference(f"ghcr.io/astral-sh/uv:0.11.23@{digest}")
        == f"ghcr.m.daocloud.io/astral-sh/uv:0.11.23@{digest}"
    )
    custom = f"quay.io/example/image:1@{digest}"
    assert tributo_it._domestic_mirror_reference(custom) == custom


def test_docker_command_environment_removes_pandafan_proxy_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10080")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:10080")

    assert all(
        variable not in tributo_it._docker_environment()
        for variable in tributo_it.DOCKER_PROXY_VARIABLES
    )


@pytest.mark.parametrize(
    "project",
    [
        "tributo-ingestion-Run-1",
        "tributo-ingestion-run.1",
        "tributo-ingestion-",
    ],
)
def test_project_validation_rejects_names_compose_cannot_use(project: str) -> None:
    with pytest.raises(tributo_it.TributoITError, match="COMPOSE_PROJECT_NAME"):
        tributo_it._validate_project(project)


def test_project_validation_accepts_generated_and_ci_names() -> None:
    tributo_it._validate_project("tributo-ingestion-123-1-deadbeef")
    tributo_it._validate_project("tributo-ingestion-ci_run-1")


def test_source_snapshot_uses_allowlist_manifest_and_completion_marker(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkout"
    destination = tmp_path / "snapshot"
    (source / "ci").mkdir(parents=True)
    (source / "src" / "tributo" / "__pycache__").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "tests").mkdir()
    (source / ".git").mkdir()
    (source / "pyproject.toml").write_text(
        """[project]
name = "tributo"
version = "1.2.3"
requires-python = ">=3.12"

[project.scripts]
tributo = "tributo.cli:main"
"""
    )
    (source / "uv.lock").write_text("version = 1\n")
    (source / "ci" / "test-suites.json").write_text("{}\n")
    module = source / "src" / "tributo" / "module.py"
    module.write_text("uncommitted = True\n")
    (source / "src" / "tributo" / "__pycache__" / "module.pyc").write_bytes(b"x")
    (source / "src" / "tributo" / "module-link.py").symlink_to("module.py")
    (source / "tests" / ".env.local").write_text("TOKEN=secret\n")
    (source / "tests" / "test_example.py").write_text("def test_example(): pass\n")
    (source / "scripts" / "ci_test_plan.py").write_text("MARKER = True\n")
    (source / ".git" / "config").write_text("secret\n")

    digest = tributo_it.create_source_snapshot(
        source,
        destination,
        owner_uid=os.getuid(),
        owner_gid=os.getgid(),
    )

    copied_module = destination / "src" / "tributo" / "module.py"
    assert copied_module.is_file()
    assert (
        destination / "scripts" / "ci_test_plan.py"
    ).read_text() == "MARKER = True\n"
    assert (destination / "src" / "tributo" / "module-link.py").is_symlink()
    assert not (destination / "src" / "tributo" / "__pycache__").exists()
    assert not (destination / "tests" / ".env.local").exists()
    assert not (destination / ".git").exists()
    assert find_project_root(destination / "src" / "tributo") == destination
    distributions = list(importlib.metadata.distributions(path=[str(destination)]))
    tributo_distribution = next(
        distribution
        for distribution in distributions
        if distribution.metadata["Name"] == "tributo"
    )
    assert tributo_distribution.version == "1.2.3"
    assert {
        (entry_point.group, entry_point.name, entry_point.value)
        for entry_point in tributo_distribution.entry_points
    } == {
        ("console_scripts", "tributo", "tributo.cli:main"),
    }
    module.write_text("changed after snapshot\n")
    assert copied_module.read_text() == "uncommitted = True\n"
    assert (destination / tributo_it.SNAPSHOT_READY).read_text().strip() == digest
    manifest_bytes = (destination / tributo_it.SNAPSHOT_MANIFEST).read_bytes()
    assert (
        tributo_it._canonical_json(json.loads(manifest_bytes)) + b"\n" == manifest_bytes
    )
    assert (
        hashlib.sha256(manifest_bytes).hexdigest()
        == (destination / tributo_it.SNAPSHOT_DIGEST).read_text().strip()
        == digest
    )


def test_source_snapshot_rejects_symlink_that_escapes_checkout(tmp_path: Path) -> None:
    source = tmp_path / "checkout"
    destination = tmp_path / "snapshot"
    (source / "ci").mkdir(parents=True)
    (source / "src").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "tests").mkdir()
    (source / "pyproject.toml").write_text("[project]\n")
    (source / "uv.lock").write_text("version = 1\n")
    (source / "ci" / "test-suites.json").write_text("{}\n")
    (source / "scripts" / "ci_test_plan.py").write_text("MARKER = True\n")
    (source / "src" / "outside").symlink_to("../../outside.txt")
    (tmp_path / "outside.txt").write_text("secret\n")

    with pytest.raises(tributo_it.TributoITError, match="escapes source snapshot"):
        tributo_it.create_source_snapshot(
            source,
            destination,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
    assert not (destination / tributo_it.SNAPSHOT_READY).exists()


def test_runtime_lock_has_bounded_wait_and_releases_with_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _write_profile(tmp_path / "repository")
    identity = tributo_it.runtime_identity(profile, "linux/amd64")
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(mode=0o700)
    monkeypatch.setattr(tributo_it, "_lock_directory", lambda: lock_dir)
    monkeypatch.setattr(tributo_it, "_docker_daemon_identity", lambda: "daemon")
    monkeypatch.setattr(tributo_it, "_image_inspect", lambda _reference: None)
    monkeypatch.setenv("TRIBUTO_IT_RUNTIME_LOCK_TIMEOUT_SECONDS", "0.1")
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(target=_hold_runtime_lock, args=(identity, ready))
    process.start()
    assert ready.wait(timeout=2)
    try:
        with pytest.raises(tributo_it.TributoITError, match="timed out"):
            with tributo_it.runtime_lock(identity):
                pytest.fail("a second process must not acquire the held lock")
    finally:
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
    assert process.exitcode == 0
    with tributo_it.runtime_lock(identity):
        pass


def test_concurrent_prepare_has_exactly_one_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _write_profile(tmp_path / "repository")
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(mode=0o700)
    context = multiprocessing.get_context("fork")
    built = context.Value("i", 0)
    build_count = context.Value("i", 0)
    start = context.Event()
    results = context.Queue()
    image_id = "sha256:" + "d" * 64

    def fake_inspect(_reference: str) -> dict[str, str] | None:
        return {"Id": image_id} if built.value else None

    def fake_build(_identity: tributo_it.RuntimeIdentity) -> str:
        with build_count.get_lock():
            build_count.value += 1
        time.sleep(0.2)
        built.value = 1
        return image_id

    monkeypatch.setattr(tributo_it, "_lock_directory", lambda: lock_dir)
    monkeypatch.setattr(tributo_it, "_docker_daemon_identity", lambda: "daemon")
    monkeypatch.setattr(tributo_it, "_image_inspect", fake_inspect)
    monkeypatch.setattr(tributo_it, "_build_runtime", fake_build)
    monkeypatch.setattr(
        tributo_it, "validate_runtime_image", lambda _identity: image_id
    )
    processes = [
        context.Process(
            target=_prepare_runtime_worker,
            args=(profile, start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=3)
        assert process.exitcode == 0

    assert build_count.value == 1
    assert {results.get(timeout=1) for _ in processes} == {
        "build",
        "concurrent-local",
    }


def test_compose_contract_rejects_service_level_build(tmp_path: Path) -> None:
    profile = _write_profile(tmp_path)
    identity = tributo_it.runtime_identity(profile, "linux/amd64")
    runtime = tributo_it.PreparedRuntime(identity, "sha256:image", "local")
    config = {
        "services": {
            "ray-head": {"build": {"context": "."}},
        }
    }
    with pytest.raises(tributo_it.TributoITError, match="must not define build"):
        tributo_it.validate_compose_contract(config, runtime, profile)


def test_registry_runtime_is_resolved_and_pulled_by_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    commands: list[list[str]] = []

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if "imagetools" in args:
            return subprocess.CompletedProcess(args, 0, f"Digest: {digest}\n", "")
        return subprocess.CompletedProcess(args, 0, "pulled\n", "")

    monkeypatch.setattr(tributo_it, "_run", fake_run)
    monkeypatch.setattr(tributo_it, "_image_inspect", lambda _reference: None)
    result, pinned = tributo_it._pull_registry_runtime(
        "ghcr.io/example/runtime:data-key", wait_seconds=0
    )

    assert result.returncode == 0
    assert pinned == f"ghcr.io/example/runtime:data-key@{digest}"
    assert commands[-1] == ["docker", "pull", pinned]


def test_registry_miss_accepts_exact_ghcr_not_found_response() -> None:
    reference = "ghcr.io/example/runtime:data-key"

    assert tributo_it._registry_miss(
        f"ERROR: {reference}: not found\n",
        reference,
    )


@pytest.mark.parametrize(
    "marker",
    ("manifest unknown", "manifest not found", "name unknown", "404 not found"),
)
def test_registry_miss_accepts_oci_marker_for_exact_reference(marker: str) -> None:
    reference = "ghcr.io/example/runtime:data-key"

    assert tributo_it._registry_miss(f"ERROR: {reference}: {marker}\n", reference)


@pytest.mark.parametrize(
    "output",
    (
        "ERROR: failed to authorize: unexpected status: 403 Forbidden",
        "ERROR: denied: permission_denied: write_package",
        "ERROR: failed to do request: dial tcp: network is unreachable",
        "ERROR: ghcr.io/example/other:data-key: not found",
        "ERROR: ghcr.io/example/other:data-key: manifest unknown",
        (
            "Inspecting ghcr.io/example/runtime:data-key\n"
            "ERROR: ghcr.io/example/other:data-key: manifest unknown"
        ),
        (
            "ERROR: ghcr.io/example/runtime:data-key: manifest unknown\n"
            "ERROR: failed to authorize: unexpected status: 403 Forbidden"
        ),
    ),
)
def test_registry_miss_rejects_non_missing_failures(output: str) -> None:
    assert not tributo_it._registry_miss(
        output,
        "ghcr.io/example/runtime:data-key",
    )


def test_publish_runtime_builds_and_pushes_exact_missing_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _write_profile(tmp_path)
    identity = tributo_it.runtime_identity(profile, "linux/amd64")
    registry = "ghcr.io/example/runtime"
    remote = f"{registry}:{profile.name}-{identity.runtime_key}"
    digest = "sha256:" + "a" * 64
    prepared = tributo_it.PreparedRuntime(identity, "sha256:image", "build")
    pulls = iter(
        (
            (
                subprocess.CompletedProcess(
                    ["docker", "buildx"], 1, "", f"ERROR: {remote}: not found\n"
                ),
                None,
            ),
            (
                subprocess.CompletedProcess(["docker", "pull"], 0, "pulled", ""),
                f"{remote}@{digest}",
            ),
        )
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(
        tributo_it,
        "_pull_registry_runtime",
        lambda _reference, *, wait_seconds: next(pulls),
    )
    monkeypatch.setattr(tributo_it, "prepare_runtime", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        tributo_it,
        "validate_runtime_image",
        lambda _identity, _reference=None: "sha256:image",
    )
    monkeypatch.setattr(
        tributo_it,
        "_run",
        lambda args, **_kwargs: (
            commands.append(args) or subprocess.CompletedProcess(args, 0, "", "")
        ),
    )

    tributo_it.publish_runtime(
        profile,
        platform="linux/amd64",
        registry=registry,
    )

    assert commands == [
        ["docker", "tag", identity.local_tag, remote],
        ["docker", "push", remote],
    ]


def test_digest_image_gets_readable_tag_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "example/tool:1@sha256:" + "b" * 64
    readable_tag = reference.split("@", 1)[0]
    image_id = "sha256:" + "c" * 64
    tagged = False

    def fake_inspect(target: str) -> dict[str, object] | None:
        if target == reference:
            return {
                "Id": image_id,
                "RepoDigests": [reference],
            }
        if target == readable_tag and tagged:
            return {"Id": image_id}
        return None

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal tagged
        assert args == ["docker", "tag", image_id, readable_tag]
        tagged = True
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(tributo_it, "_image_inspect", fake_inspect)
    monkeypatch.setattr(tributo_it, "_run", fake_run)

    assert tributo_it.ensure_digest_image(reference) == image_id
    assert tagged


def test_digest_image_refuses_to_move_existing_readable_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "example/tool:1@sha256:" + "e" * 64
    wrong = "example/tool:1@sha256:" + "f" * 64

    monkeypatch.setattr(
        tributo_it,
        "_image_inspect",
        lambda reference: (
            {"Id": "sha256:wrong", "RepoDigests": [wrong]}
            if reference == "example/tool:1"
            else None
        ),
    )

    with pytest.raises(tributo_it.TributoITError, match="refusing to overwrite"):
        tributo_it.ensure_digest_image(expected)


def test_run_data_ingestion_cli_dispatches_without_platform_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _write_profile(tmp_path)
    called: list[tributo_it.RuntimeProfile] = []
    monkeypatch.setattr(tributo_it, "load_profile", lambda _name: profile)
    monkeypatch.setattr(tributo_it, "run_data_ingestion", called.append)

    assert tributo_it.main(["run-data-ingestion"]) == 0
    assert called == [profile]


def test_container_states_capture_ownership_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64
    unlabeled_id = "b" * 64
    output = (
        f"{container_id}\trunning\tray-head\ttributo-other-run\n"
        f"{unlabeled_id}\texited\tstandalone\t\n"
    )
    monkeypatch.setattr(
        tributo_it,
        "_run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )

    assert tributo_it._container_states() == {
        container_id: tributo_it.ContainerSnapshot(
            state="running",
            name="ray-head",
            compose_project="tributo-other-run",
        ),
        unlabeled_id: tributo_it.ContainerSnapshot(
            state="exited",
            name="standalone",
            compose_project="",
        ),
    }


def test_container_diagnostic_baseline_failure_is_advisory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> dict[str, tributo_it.ContainerSnapshot]:
        raise tributo_it.TributoITError("diagnostic unavailable")

    monkeypatch.setattr(tributo_it, "_container_states", fail)

    assert tributo_it._capture_container_diagnostic_baseline() is None
    assert "owned-project checks remain authoritative" in capsys.readouterr().err


def test_external_container_activity_is_reported_without_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    container_id = "c" * 64
    before = {
        container_id: tributo_it.ContainerSnapshot(
            state="created",
            name="other-ray-worker",
            compose_project="tributo-other-run",
        )
    }
    monkeypatch.setattr(
        tributo_it,
        "_container_states",
        lambda: {
            container_id: tributo_it.ContainerSnapshot(
                state="running",
                name="other-ray-worker",
                compose_project="tributo-other-run",
            )
        },
    )

    tributo_it._report_external_container_activity(
        "tributo-lance-vector-current",
        before,
    )

    diagnostic = capsys.readouterr().err
    assert "Concurrent external Docker activity detected and ignored" in diagnostic
    assert "tributo-other-run" in diagnostic
    assert "tributo-lance-vector-current" in diagnostic


def test_new_external_container_is_reported_without_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    new_id = "e" * 64
    monkeypatch.setattr(
        tributo_it,
        "_container_states",
        lambda: {
            new_id: tributo_it.ContainerSnapshot(
                state="running",
                name="unrelated-service",
                compose_project="another-project",
            )
        },
    )

    tributo_it._report_external_container_activity(
        "tributo-lance-vector-current",
        {},
    )

    diagnostic = capsys.readouterr().err
    assert new_id in diagnostic
    assert '"before": "<missing>"' in diagnostic
    assert "another-project" in diagnostic


def test_external_container_diagnostic_failure_does_not_fail_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = {
        "d" * 64: tributo_it.ContainerSnapshot(
            state="running",
            name="other-service",
            compose_project="tributo-other-run",
        )
    }

    def fail() -> dict[str, tributo_it.ContainerSnapshot]:
        raise tributo_it.TributoITError("diagnostic unavailable")

    monkeypatch.setattr(tributo_it, "_container_states", fail)

    tributo_it._report_external_container_activity(
        "tributo-lance-vector-current",
        before,
    )

    assert "owned-project checks remain authoritative" in capsys.readouterr().err


def test_image_diagnostics_exclude_pulled_digest_only_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dangling_id = "sha256:" + "1" * 64
    pulled_id = "sha256:" + "2" * 64

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "dangling=true" in args:
            return subprocess.CompletedProcess(args, 0, f"{dangling_id}\n", "")
        if "inspect" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                '["example/runtime@sha256:' + "3" * 64 + '"]',
                "",
            )
        return subprocess.CompletedProcess(
            args,
            0,
            f"example/runtime\t<none>\t{pulled_id}\n",
            "",
        )

    monkeypatch.setattr(tributo_it, "_run", fake_run)

    assert tributo_it._diagnostic_image_ids() == {dangling_id}


def test_new_image_artifacts_are_advisory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    new_id = "sha256:" + "4" * 64
    monkeypatch.setattr(
        tributo_it,
        "_diagnostic_image_ids",
        lambda: {new_id},
    )

    tributo_it._report_new_image_artifacts(
        "tributo-lance-vector-current",
        set(),
    )

    diagnostic = capsys.readouterr().err
    assert "detected and ignored" in diagnostic
    assert new_id in diagnostic


def test_image_diagnostic_failures_are_advisory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> set[str]:
        raise tributo_it.TributoITError("diagnostic unavailable")

    monkeypatch.setattr(tributo_it, "_diagnostic_image_ids", fail)

    assert tributo_it._capture_image_diagnostic_baseline() is None
    tributo_it._report_new_image_artifacts(
        "tributo-lance-vector-current",
        set(),
    )
    diagnostic = capsys.readouterr().err
    assert diagnostic.count("owned-project checks remain authoritative") == 2


def test_project_resources_are_queried_by_exact_compose_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "owned-resource-id\n", "")

    monkeypatch.setattr(tributo_it, "_run", fake_run)

    assert tributo_it._project_resource_ids("tributo-lance-vector-current") == {
        "containers": ["owned-resource-id"],
        "networks": ["owned-resource-id"],
        "volumes": ["owned-resource-id"],
    }
    assert len(commands) == 3
    assert all(
        "label=com.docker.compose.project=tributo-lance-vector-current" in command
        for command in commands
    )


def test_owned_project_residue_remains_a_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tributo_it,
        "_project_resource_ids",
        lambda _project: {
            "containers": [],
            "networks": ["owned-network-id"],
            "volumes": [],
        },
    )

    with pytest.raises(tributo_it.TributoITError, match="still owns resources"):
        tributo_it._assert_project_absent("tributo-lance-vector-current")
