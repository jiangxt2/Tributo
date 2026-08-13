"""CLI surface tests for the thin Ray Jobs explainability entry point."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from tributo.cli import main


def test_explain_submits_json_config(tmp_path) -> None:
    config = tmp_path / "explain.json"
    config.write_text("{}", encoding="utf-8")
    with patch(
        "tributo.explainability.job_runner.submit_explainability_job",
        return_value="job-explain-1",
    ) as submit:
        result = CliRunner().invoke(
            main,
            ["explain", "--config", str(config), "--address", "http://ray:8265"],
        )
    assert result.exit_code == 0
    assert "job-explain-1" in result.output
    submit.assert_called_once_with(str(config), dashboard_url="http://ray:8265")


def test_explain_rejects_missing_config() -> None:
    result = CliRunner().invoke(main, ["explain", "--config", "/missing.json"])
    assert result.exit_code != 0
