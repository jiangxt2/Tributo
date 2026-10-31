"""Tests for the shared host-uv requirements export contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from tests.support.runtime_versions import RAY_VERSION
from tools.host_uv_export import HostUVExportError, export_locked_requirements


def _runner_factory(
    root: Path,
    *,
    content: bytes,
    mutate_lock: bool = False,
):
    commands: list[list[str]] = []

    def run(args: list[str], cwd: Path) -> CompletedProcess[str]:
        commands.append(args)
        if args[1:] == ["--version"]:
            return CompletedProcess(args, 0, "uv 0.11.23\n", "")
        if args[1:3] == ["lock", "--check"]:
            return CompletedProcess(args, 0, "", "")
        output = Path(args[args.index("--output-file") + 1])
        output.write_bytes(content)
        if mutate_lock:
            (root / "uv.lock").write_bytes(b"changed")
        return CompletedProcess(args, 0, "", "")

    return commands, run


def test_export_locked_requirements_is_hashed_and_lock_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"locked")
    content = f"ray=={RAY_VERSION} --hash=sha256:".encode() + b"a" * 64 + b"\n"
    commands, run = _runner_factory(tmp_path, content=content)
    monkeypatch.setattr(
        "tools.host_uv_export.shutil.which", lambda _name: "/usr/bin/uv"
    )

    exported = export_locked_requirements(
        root=tmp_path,
        extras=("dev", "training"),
        baseline_uv_version="0.11.23",
        run=run,
    )

    assert exported.content == content
    assert exported.sha256 == hashlib.sha256(content).hexdigest()
    assert lock.read_bytes() == b"locked"
    export_command = commands[-1]
    first_extra = export_command.index("--extra")
    second_extra = export_command.index("--extra", first_extra + 1)
    assert export_command[first_extra + 1] == "dev"
    assert export_command[second_extra + 1] == "training"


def test_export_locked_requirements_accepts_hashed_source_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked")
    content = (
        b"ray-clickhouse @ https://github.com/jiangxt2/ray-clickhouse/archive/"
        b"857ef213e2579d9fff144ffaf3f0de359bd8d8bc.tar.gz --hash=sha256:"
        + b"a" * 64
        + b"\n"
    )
    _commands, run = _runner_factory(tmp_path, content=content)
    monkeypatch.setattr(
        "tools.host_uv_export.shutil.which", lambda _name: "/usr/bin/uv"
    )

    exported = export_locked_requirements(
        root=tmp_path,
        extras=("clickhouse",),
        baseline_uv_version="0.11.23",
        run=run,
    )

    assert exported.content == content


def test_export_locked_requirements_rejects_lock_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked")
    _commands, run = _runner_factory(
        tmp_path,
        content=f"ray=={RAY_VERSION} --hash=sha256:".encode() + b"a" * 64 + b"\n",
        mutate_lock=True,
    )
    monkeypatch.setattr(
        "tools.host_uv_export.shutil.which", lambda _name: "/usr/bin/uv"
    )

    with pytest.raises(HostUVExportError, match="changed uv.lock"):
        export_locked_requirements(
            root=tmp_path,
            extras=("dev",),
            baseline_uv_version="0.11.23",
            run=run,
        )


def test_export_locked_requirements_rejects_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked")
    _commands, run = _runner_factory(
        tmp_path,
        content=f"-e .\nray=={RAY_VERSION} --hash=sha256:".encode() + b"a" * 64 + b"\n",
    )
    monkeypatch.setattr(
        "tools.host_uv_export.shutil.which", lambda _name: "/usr/bin/uv"
    )

    with pytest.raises(HostUVExportError, match="host paths"):
        export_locked_requirements(
            root=tmp_path,
            extras=("dev",),
            baseline_uv_version="0.11.23",
            run=run,
        )


def _cpu_profile(root: Path) -> Path:
    (root / "pyproject.toml").write_text("# sdk project\n")
    (root / "uv.lock").write_text("# sdk lock\n")
    profile = root / "cpu"
    profile.mkdir()
    (profile / "pyproject.toml").write_text("# cpu project\n")
    (profile / "README.md").write_text("cpu\n")
    url = "https://download.pytorch.org/torch.whl"
    (profile / "uv.lock").write_text(
        '[[package]]\nname="torch"\nversion="2.13.0+cpu"\n'
        f'source={{url="{url}"}}\n'
        'wheels=[{hash="sha256:' + "c" * 64 + '"}]\n'
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "platform": "linux/arm64",
        "python_version": "3.12",
        "torch_version": "2.13.0+cpu",
        "torch_url": url,
        "torch_sha256": "c" * 64,
        "inputs": {name: digest(root / name) for name in ("pyproject.toml", "uv.lock")},
        "files": {
            name: digest(profile / name)
            for name in ("pyproject.toml", "uv.lock", "README.md")
        },
    }
    (profile / "profile.json").write_text(json.dumps(manifest))
    return profile


def test_cpu_export_uses_separate_lock_and_explicit_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _cpu_profile(tmp_path)
    content = b"torch==2.13.0+cpu --hash=sha256:" + b"c" * 64 + b"\n"
    commands, runner = _runner_factory(tmp_path, content=content)
    working_directories = []

    def run(args: list[str], cwd: Path) -> CompletedProcess[str]:
        working_directories.append(cwd)
        return runner(args, cwd)

    monkeypatch.setattr("tools.host_uv_export.shutil.which", lambda _: "/usr/bin/uv")
    exported = export_locked_requirements(
        root=tmp_path,
        extras=("graph",),
        groups=("dev",),
        baseline_uv_version="0.11.23",
        run=run,
        cpu_profile=profile,
        platform="linux/arm64",
    )
    assert exported.content == content
    assert working_directories[1:] == [profile, profile]
    assert commands[-1][commands[-1].index("--group") + 1] == "dev"
    assert (tmp_path / "uv.lock").read_text() == "# sdk lock\n"


@pytest.mark.parametrize("target", [None, "linux/amd64", "darwin/arm64"])
def test_cpu_export_rejects_other_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str | None
) -> None:
    profile = _cpu_profile(tmp_path)
    _, run = _runner_factory(tmp_path, content=b"unused")
    monkeypatch.setattr("tools.host_uv_export.shutil.which", lambda _: "/usr/bin/uv")
    with pytest.raises(HostUVExportError, match="target platform"):
        export_locked_requirements(
            root=tmp_path,
            extras=(),
            baseline_uv_version="0.11.23",
            run=run,
            cpu_profile=profile,
            platform=target,
        )


@pytest.mark.parametrize("changed", ["uv.lock", "cpu/uv.lock", "cpu/pyproject.toml"])
def test_cpu_export_rejects_stale_or_modified_lock_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    profile = _cpu_profile(tmp_path)
    (tmp_path / changed).write_text("changed")
    _, run = _runner_factory(tmp_path, content=b"unused")
    monkeypatch.setattr("tools.host_uv_export.shutil.which", lambda _: "/usr/bin/uv")
    with pytest.raises(HostUVExportError, match="stale or modified"):
        export_locked_requirements(
            root=tmp_path,
            extras=(),
            baseline_uv_version="0.11.23",
            run=run,
            cpu_profile=profile,
            platform="linux/arm64",
        )


def test_cpu_export_rejects_wheel_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _cpu_profile(tmp_path)
    manifest = json.loads((profile / "profile.json").read_text())
    manifest["torch_sha256"] = "d" * 64
    (profile / "profile.json").write_text(json.dumps(manifest))
    _, run = _runner_factory(tmp_path, content=b"unused")
    monkeypatch.setattr("tools.host_uv_export.shutil.which", lambda _: "/usr/bin/uv")
    with pytest.raises(HostUVExportError, match="exact CPU Wheel"):
        export_locked_requirements(
            root=tmp_path,
            extras=(),
            baseline_uv_version="0.11.23",
            run=run,
            cpu_profile=profile,
            platform="linux/arm64",
        )
