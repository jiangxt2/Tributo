"""Tests for the shared host-uv requirements export contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from subprocess import CompletedProcess

import pytest

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
    content = b"ray==2.55.1 --hash=sha256:" + b"a" * 64 + b"\n"
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


def test_export_locked_requirements_rejects_lock_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"locked")
    _commands, run = _runner_factory(
        tmp_path,
        content=b"ray==2.55.1 --hash=sha256:" + b"a" * 64 + b"\n",
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
        content=b"-e .\nray==2.55.1 --hash=sha256:" + b"a" * 64 + b"\n",
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
