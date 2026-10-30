"""Validate generated CPU locks without network access or package installation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import generate_cpu_profile as generator


@pytest.mark.parametrize("failure", [None, "hash", "version"])
def test_cpu_profile_preserves_sdk_inputs_and_rejects_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    project = (
        '[project]\nname="tributo"\nversion="1.3.0.dev0"\n'
        'requires-python = ">=3.12,<3.14"\n[tool.uv.sources]\n'
    )
    original_lock = (
        '[[package]]\nname="torch"\nversion="2.13.0"\n'
        '[[package]]\nname="filelock"\nversion="3.20.0"\n'
    )
    (tmp_path / "pyproject.toml").write_text(project)
    (tmp_path / "uv.lock").write_text(original_lock)
    (tmp_path / "README.md").write_text("SDK\n")
    profile = tmp_path / "cpu"
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator, "PROFILE", profile)

    def resolve(args: list[str], *, cwd: Path, check: bool) -> None:
        assert args[:2] == ["uv", "lock"] and cwd == tmp_path and check
        digest = "0" * 64 if failure == "hash" else generator.TORCH_SHA256
        version = "3.21.0" if failure == "version" else "3.20.0"
        (profile / "uv.lock").write_text(
            '[[package]]\nname="torch"\nversion="2.13.0+cpu"\n'
            'wheels=[{hash="sha256:' + digest + '"}]\n'
            f'[[package]]\nname="filelock"\nversion="{version}"\n'
        )

    monkeypatch.setattr(subprocess, "run", resolve)
    if failure:
        with pytest.raises(ValueError, match="hash differs|version drift"):
            generator.main()
        assert not (profile / "profile.json").exists()
    else:
        generator.main()
        manifest = json.loads((profile / "profile.json").read_text())
        assert manifest["platform"] == "linux/arm64"
        assert manifest["torch_sha256"] == generator.TORCH_SHA256
        assert "package = false" in (profile / "pyproject.toml").read_text()
    assert (tmp_path / "pyproject.toml").read_text() == project
    assert (tmp_path / "uv.lock").read_text() == original_lock
