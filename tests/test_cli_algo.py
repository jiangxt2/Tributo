"""Tests for the ``tributo algo`` CLI command group."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

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
        # Should list at least the bootstrapped first-party XGBoost descriptor.
        assert "xgboost" in result.output
        assert "dnn" in result.output
        assert "MODALITY" in result.output
        assert "GPU REQ" in result.output
        assert "STATUS" in result.output
        assert "STABILITY" in result.output
        assert "AVAILABLE" in result.output
        assert "TESTED" in result.output
        assert "SUPPORTED" in result.output

    def test_list_json_output(self, runner) -> None:
        result = runner.invoke(main, ["algo", "list", "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, list)
        names = [item["name"] for item in parsed]
        assert "xgboost" in names
        by_name = {item["name"]: item for item in parsed}
        assert by_name["xgboost"]["execution_kind"] == "train"
        assert by_name["xgboost"]["supported_tasks"] == ["fit"]
        assert by_name["xgboost"]["capabilities"] == [
            "tunable",
            "exportable",
            "distributed",
        ]
        assert by_name["xgboost"]["data_loading"] == "canonical_driver"
        assert by_name["xgboost"]["stability"] == "alpha"
        assert by_name["xgboost"]["available"] is True
        assert by_name["xgboost"]["compatibility_only"] is False
        assert by_name["xgboost"]["tested"] is True
        assert by_name["xgboost"]["supported"] is True
        assert by_name["xgboost"]["native_migration_complete"] is True
        assert by_name["xgboost"]["implementation_ids"] == [
            "tributo.xgboost.framework_native",
            "tributo.xgboost.legacy_trainer",
        ]
        assert by_name["xgboost"]["distribution_strategies"] == ["framework_native"]
        assert by_name["xgboost"]["execution_profiles"] == [
            "cluster",
            "local",
        ]
        assert by_name["xgboost"]["validated_execution_profiles"] == [
            "cluster",
            "local",
        ]

        if "dnn" in by_name:
            assert "distributed" in by_name["dnn"]["capabilities"]
        if "pu" in by_name:
            assert "distributed" in by_name["pu"]["capabilities"]

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
        assert "Execution Kind" in result.output
        assert "Capabilities" in result.output
        assert "distributed" in result.output
        assert "Stability:      alpha" in result.output
        assert "tributo.xgboost.framework_native" in result.output
        assert "Distribution:   ['framework_native']" in result.output
        assert "Profiles:       ['cluster', 'local']" in result.output
        assert "Validated:      ['cluster', 'local']" in result.output

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
        schema = json.loads(result.stdout)
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
            _registry.unregister(name)


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
            _registry.unregister(name)

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


class TestAlgoRun:
    def test_run_builds_one_formal_request_from_json(self, runner, tmp_path) -> None:
        from tributo.algorithms.api import (
            AlgorithmExecutionResult,
            DistributionStrategy,
            ExecutionProfile,
            ExecutionReceipt,
            StateCoordination,
            StateCoordinationEvidence,
            WorkerExecutionEvidence,
            WorkerResources,
        )

        config_file = tmp_path / "execution.json"
        config_file.write_text(
            json.dumps(
                {
                    "algorithm": "multinomial_nb",
                    "profile": "local",
                    "worker_count": 1,
                    "input": {
                        "ingestion": {
                            "source": {"type": "parquet", "path": "/tmp/data.parquet"},
                            "engine": "ray",
                        },
                        "features": ["f1", "f2"],
                        "label": "label",
                    },
                    "algorithm_config": {"output": {"bundle_uri": "/tmp/bundles"}},
                    "local_runtime": {"num_cpus": 1, "num_gpus": 0},
                }
            )
        )
        receipt = ExecutionReceipt(
            run_id="run-1",
            plan_id="a" * 64,
            requested_algorithm="multinomial_nb",
            canonical_algorithm="multinomial_nb",
            profile=ExecutionProfile.LOCAL,
            strategy=DistributionStrategy.RAY_MAP_REDUCE,
            requested_worker_count=1,
            distributed_min_workers=2,
            requested_resources_per_worker=WorkerResources(),
            workers=(
                WorkerExecutionEvidence(
                    worker_id="worker-0",
                    node_id="node-0",
                    rank=0,
                    world_size=1,
                    shard_id="shard-0",
                    resources=WorkerResources(),
                    rows_processed=2,
                ),
            ),
            input_complete=True,
            state=StateCoordinationEvidence(
                coordination=StateCoordination.ASSOCIATIVE_REDUCE,
                synchronized=True,
                bounded=True,
                global_model_digest="b" * 64,
            ),
        )
        fake_result = SimpleNamespace(
            run_id="run-1",
            plan_id="a" * 64,
            execution=AlgorithmExecutionResult(
                status="succeeded",
                metrics={"row_count": 2},
                outputs={"bundle_uri": "/tmp/bundles/bundle"},
            ),
            execution_receipt=receipt,
        )
        fake_dispatcher = SimpleNamespace(execute=lambda *args, **kwargs: fake_result)

        with patch(
            "tributo.algorithms.composition.build_algorithm_dispatcher",
            return_value=fake_dispatcher,
        ) as build_dispatcher:
            result = runner.invoke(main, ["algo", "run", "--config", str(config_file)])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["status"] == "succeeded"
        assert payload["execution_receipt"]["execution_profile"] == "local"
        manager = build_dispatcher.call_args.kwargs["runtime_manager"]
        assert manager._default_local_options.num_cpus == 1.0

    def test_run_rejects_local_runtime_for_cluster(self, runner, tmp_path) -> None:
        config_file = tmp_path / "execution.json"
        config_file.write_text(
            json.dumps(
                {
                    "algorithm": "multinomial_nb",
                    "profile": "cluster",
                    "worker_count": 2,
                    "input": {
                        "ingestion": {
                            "source": {"type": "parquet", "path": "/tmp/data.parquet"},
                            "engine": "ray",
                        },
                        "features": ["f1"],
                        "label": "label",
                    },
                    "algorithm_config": {},
                    "local_runtime": {"num_cpus": 2},
                }
            )
        )

        result = runner.invoke(main, ["algo", "run", "--config", str(config_file)])

        assert result.exit_code == 1
        assert "local_runtime is valid only" in result.output
