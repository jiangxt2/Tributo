"""Core broker CLI isolation tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from tributo.cli import main


def test_broker_config_is_json_only_and_provider_owned(tmp_path) -> None:
    path = tmp_path / "broker.json"
    path.write_text(json.dumps({"opaque_provider_field": True}), encoding="utf-8")
    result = CliRunner().invoke(
        main, ["broker", "validate", "--broker", "missing", "--config", str(path)]
    )
    assert result.exit_code != 0
    assert "Unknown broker" in result.output


def test_normal_cli_does_not_require_redis() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "broker" in result.output


def test_broker_list_without_provider_is_empty(monkeypatch) -> None:
    monkeypatch.setattr("tributo.plugin._iter_entry_points", lambda _group: iter(()))
    result = CliRunner().invoke(main, ["broker", "list"])
    assert result.exit_code == 0
    assert result.output == ""


def test_broker_consume_without_provider_fails_closed(tmp_path) -> None:
    path = tmp_path / "broker.json"
    path.write_text("{}", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        ["broker", "consume", "--broker", "missing", "--config", str(path)],
    )
    assert result.exit_code != 0
    assert "Unknown broker" in result.output


def test_importing_core_cli_does_not_import_broker_module() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1] / "src"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tributo.cli; assert 'tributo.cli_broker' not in sys.modules",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
