"""Tests for the ``tributo algo`` CLI command group."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from tributo.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# algo list
# ---------------------------------------------------------------------------


class TestAlgoList:
    def test_list_default(self, runner) -> None:
        result = runner.invoke(main, ["algo", "list"])
        assert result.exit_code == 0
        # Should list at least xgboost (registered at import)
        assert "xgboost" in result.output
        assert "dnn" in result.output

    def test_list_json_output(self, runner) -> None:
        result = runner.invoke(main, ["algo", "list", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        names = [item["name"] for item in parsed]
        assert "xgboost" in names

    def test_list_filter_family(self, runner) -> None:
        result = runner.invoke(main, ["algo", "list", "--family", "classification"])
        assert result.exit_code == 0
        # xgboost supports classification
        assert "xgboost" in result.output

    def test_list_filter_modality(self, runner) -> None:
        result = runner.invoke(main, ["algo", "list", "--modality", "tabular"])
        assert result.exit_code == 0
        assert "xgboost" in result.output

    def test_list_no_results(self, runner) -> None:
        """Filter combination that nothing matches."""
        result = runner.invoke(main, ["algo", "list", "--family", "clustering"])
        assert result.exit_code == 0
        assert "No algorithms found" in result.output

    def test_list_include_deprecated(self, runner) -> None:
        result = runner.invoke(main, ["algo", "list", "--deprecated"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# algo info
# ---------------------------------------------------------------------------


class TestAlgoInfo:
    def test_info_known_algorithm(self, runner) -> None:
        result = runner.invoke(main, ["algo", "info", "xgboost"])
        assert result.exit_code == 0
        assert "xgboost" in result.output
        assert "Problem Types" in result.output
        assert "Data Modality" in result.output

    def test_info_unknown_algorithm(self, runner) -> None:
        result = runner.invoke(main, ["algo", "info", "nonexistent_algo_xyz"])
        assert result.exit_code == 1
        assert "Error" in result.output


# ---------------------------------------------------------------------------
# algo config-schema
# ---------------------------------------------------------------------------


class TestAlgoConfigSchema:
    def test_schema_for_xgboost(self, runner) -> None:
        result = runner.invoke(main, ["algo", "config-schema", "xgboost"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "properties" in schema

    def test_schema_no_config_model_exits_1(self, runner) -> None:
        """Algorithm without config_model → error exit 1."""
        from tributo.training.algorithm_spec import AlgorithmSpec
        from tributo.training.registry import _registry

        name = "_test_no_cfg_schema"
        _registry.register(
            name, AlgorithmSpec(name=name, trainer_cls=type("Fake", (), {}))
        )
        try:
            result = runner.invoke(main, ["algo", "config-schema", name])
            assert result.exit_code == 1
            assert "Error" in result.output
        finally:
            _registry._store.pop(name, None)


# ---------------------------------------------------------------------------
# algo validate
# ---------------------------------------------------------------------------


class TestAlgoValidate:
    def test_valid_config_with_source(self, runner, tmp_path) -> None:
        """Full config including data.source → exit 0."""
        config_file = tmp_path / "valid.json"
        config_file.write_text(
            json.dumps(
                {
                    "data": {
                        "source": {
                            "type": "parquet",
                            "path": "/tmp/data.parquet",
                        }
                    },
                    "model": {"objective": "binary:logistic"},
                    "training": {"num_rounds": 50},
                }
            )
        )
        result = runner.invoke(
            main,
            ["algo", "validate", "--algo", "xgboost", "--config", str(config_file)],
        )
        assert result.exit_code == 0

    def test_valid_config_missing_source_rejected(self, runner, tmp_path) -> None:
        """Config without data.source → exit 1 (execution validation fails)."""
        config_file = tmp_path / "no_source.json"
        config_file.write_text(
            json.dumps(
                {
                    "model": {"objective": "binary:logistic"},
                    "training": {"num_rounds": 50},
                }
            )
        )
        result = runner.invoke(
            main,
            [
                "algo",
                "validate",
                "--algo",
                "xgboost",
                "--config",
                str(config_file),
            ],
        )
        assert result.exit_code == 1

    def test_unknown_algorithm(self, runner, tmp_path) -> None:
        config_file = tmp_path / "cfg.json"
        config_file.write_text("{}")
        result = runner.invoke(
            main,
            [
                "algo",
                "validate",
                "--algo",
                "nonexistent_xyz",
                "--config",
                str(config_file),
            ],
        )
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_no_config_model_exit_2(self, runner, tmp_path) -> None:
        """Algorithm without config_model → exit code 2."""
        from tributo.training.algorithm_spec import AlgorithmSpec
        from tributo.training.registry import _registry

        name = "_test_no_cfg_validate"
        _registry.register(
            name, AlgorithmSpec(name=name, trainer_cls=type("Fake", (), {}))
        )
        config_file = tmp_path / "cfg.json"
        config_file.write_text("{}")
        try:
            result = runner.invoke(
                main,
                ["algo", "validate", "--algo", name, "--config", str(config_file)],
            )
            assert result.exit_code == 2
        finally:
            _registry._store.pop(name, None)

    def test_invalid_json_config(self, runner, tmp_path) -> None:
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json")
        # json.load will raise, caught as click error or json.JSONDecodeError
        result = runner.invoke(
            main,
            [
                "algo",
                "validate",
                "--algo",
                "xgboost",
                "--config",
                str(config_file),
            ],
        )
        assert result.exit_code != 0
