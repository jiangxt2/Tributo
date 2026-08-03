"""Integration tests — catalog integrity, pipeline E2E, snapshot freshness.

These tests validate that the full integration stack works correctly:
1. Integrity gate fires on bad replacement graphs at import time.
2. AlgorithmCatalog sees live registrations (snapshot freshness).
3. build_effective_config → validate_and_normalize → validate_execution
   pipeline produces configs accepted by real trainers.
4. DataLoadingMode correctly routes data-source resolution.
"""

from __future__ import annotations

import pytest

from tributo._common.registry import Registry
from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    AlgorithmStatus,
    DataLoadingMode,
    ProblemType,
)
from tributo.training.catalog import AlgorithmCatalog, get_algorithm_catalog
from tributo.training.config import (
    build_effective_config,
    resolve_data_source,
    validate_execution_config,
)
from tributo.training.registry import _registry


class FakeTrainer:
    pass


# ---------------------------------------------------------------------------
# Integrity gate — replacement graph validation
# ---------------------------------------------------------------------------


class TestIntegrityGate:
    def test_integrity_rejects_missing_replacement(self) -> None:
        """A catalog with a DEPRECATED→missing chain should fail."""
        r: Registry[str, AlgorithmSpec] = Registry("test")
        r.register(
            "old",
            AlgorithmSpec(
                name="old",
                trainer_cls=FakeTrainer,
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="nonexistent",
            ),
        )
        cat = AlgorithmCatalog(r)
        with pytest.raises(JobConfigurationError, match="not found in registry"):
            cat.validate_integrity()

    def test_integrity_rejects_cycle(self) -> None:
        r: Registry[str, AlgorithmSpec] = Registry("test")
        r.register(
            "a",
            AlgorithmSpec(
                name="a",
                trainer_cls=FakeTrainer,
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="b",
            ),
        )
        r.register(
            "b",
            AlgorithmSpec(
                name="b",
                trainer_cls=FakeTrainer,
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="a",
            ),
        )
        cat = AlgorithmCatalog(r)
        with pytest.raises(JobConfigurationError, match="cycle"):
            cat.validate_integrity()

    def test_shared_registry_is_valid_on_import(self) -> None:
        """The shared registry passes integrity as loaded by __init__.py."""
        cat = get_algorithm_catalog()
        cat.validate_integrity()  # must not raise


# ---------------------------------------------------------------------------
# Snapshot freshness
# ---------------------------------------------------------------------------


class TestSnapshotFreshness:
    def test_new_registration_visible_to_catalog(self) -> None:
        """Registering after catalog creation should be immediately visible."""
        name = "_test_freshness"
        spec = AlgorithmSpec(
            name=name,
            trainer_cls=FakeTrainer,
            problem_types=(ProblemType.CLUSTERING,),
        )
        _registry.register(name, spec)
        try:
            cat = get_algorithm_catalog()
            assert name in cat.list(problem_type=ProblemType.CLUSTERING)
        finally:
            _registry._store.pop(name, None)

    def test_catalog_list_calls_snapshot_each_time(self) -> None:
        """Each list() call gets a fresh snapshot."""
        cat = get_algorithm_catalog()
        names_before = cat.list()
        assert "xgboost" in names_before
        # Register a new one, then list again
        name = "_test_snapshot"
        _registry.register(name, AlgorithmSpec(name=name, trainer_cls=FakeTrainer))
        try:
            names_after = cat.list()
            assert name in names_after
        finally:
            _registry._store.pop(name, None)


# ---------------------------------------------------------------------------
# Pipeline E2E — build_effective_config with real trainer specs
# ---------------------------------------------------------------------------


class TestPipelineE2E:
    def test_xgboost_pipeline_accepts_minimal_config(self) -> None:
        """Minimal valid XGBoost config passes the full pipeline."""
        spec = get_algorithm_catalog().get_spec("xgboost")
        # All sections have defaults, empty config is valid
        cfg = build_effective_config(spec, {}, datasets_supplied=True)
        assert "model" in cfg
        assert "training" in cfg

    def test_xgboost_pipeline_rejects_typo_in_top_level(self) -> None:
        spec = get_algorithm_catalog().get_spec("xgboost")
        with pytest.raises(JobConfigurationError, match="Config validation failed"):
            build_effective_config(spec, {"typo_field": 123})

    def test_dnn_pipeline_accepts_minimal_config(self) -> None:
        spec = get_algorithm_catalog().get_spec("dnn")
        cfg = build_effective_config(spec, {}, datasets_supplied=True)
        assert isinstance(cfg, dict)

    def test_default_config_overridable(self) -> None:
        """User config overrides defaults in the pipeline."""
        spec = get_algorithm_catalog().get_spec("xgboost")
        cfg = build_effective_config(
            spec,
            {"training": {"num_rounds": 42, "seed": 7}},
            datasets_supplied=True,
        )
        assert cfg["training"]["num_rounds"] == 42
        assert cfg["training"]["seed"] == 7

    def test_dot_overrides_applied(self) -> None:
        spec = get_algorithm_catalog().get_spec("xgboost")
        cfg = build_effective_config(
            spec,
            {},
            dot_overrides={"training.num_rounds": 77},
            datasets_supplied=True,
        )
        assert cfg["training"]["num_rounds"] == 77

    def test_execution_check_blocks_missing_source(self) -> None:
        """When datasets_supplied=False, CANONICAL_DRIVER requires data.source."""
        spec = get_algorithm_catalog().get_spec("xgboost")
        with pytest.raises(JobConfigurationError, match="requires 'data.source'"):
            build_effective_config(spec, {})


# ---------------------------------------------------------------------------
# DataLoadingMode routing
# ---------------------------------------------------------------------------


class TestDataLoadingRouting:
    def test_xgboost_is_canonical_driver(self) -> None:
        spec = get_algorithm_catalog().get_spec("xgboost")
        assert spec.data_loading == DataLoadingMode.CANONICAL_DRIVER

    def test_dnn_is_canonical_driver(self) -> None:
        spec = get_algorithm_catalog().get_spec("dnn")
        assert spec.data_loading == DataLoadingMode.CANONICAL_DRIVER

    def test_pu_is_canonical_trainer(self) -> None:
        spec = get_algorithm_catalog().get_spec("pu")
        assert spec.data_loading == DataLoadingMode.CANONICAL_TRAINER

    def test_validate_execution_respects_canonical_trainer(self) -> None:
        """CANONICAL_TRAINER always requires data.source, even with datasets_supplied."""
        spec = get_algorithm_catalog().get_spec("pu")
        with pytest.raises(JobConfigurationError):
            validate_execution_config(
                spec, {"training": {"num_rounds": 10}}, datasets_supplied=True
            )

    def test_validate_execution_skips_for_legacy(self) -> None:
        """LEGACY_DRIVER skips source checks entirely."""
        from tributo.training.algorithm_spec import AlgorithmSpec

        name = "_test_legacy"
        spec = AlgorithmSpec(
            name=name,
            trainer_cls=FakeTrainer,
            data_loading=DataLoadingMode.LEGACY_DRIVER,
        )
        _registry.register(name, spec)
        try:
            # Should not raise even without data section
            validate_execution_config(
                spec, {"anything": "goes"}, datasets_supplied=False
            )
        finally:
            _registry._store.pop(name, None)


# ---------------------------------------------------------------------------
# Data source resolution integration
# ---------------------------------------------------------------------------


class TestResolveDataSourceIntegration:
    def test_canonical_parquet_source(self) -> None:
        spec = get_algorithm_catalog().get_spec("xgboost")
        config = {
            "data": {
                "source": {
                    "type": "parquet",
                    "path": "/tmp/data.parquet",
                }
            }
        }
        result = resolve_data_source(spec, config)
        assert result["type"] == "parquet"
        assert result["path"] == "/tmp/data.parquet"

    def test_canonical_csv_source(self) -> None:
        spec = get_algorithm_catalog().get_spec("xgboost")
        config = {"data": {"source": {"type": "csv", "path": "/tmp/data.csv"}}}
        result = resolve_data_source(spec, config)
        assert result["type"] == "csv"

    def test_canonical_sql_source(self) -> None:
        """Canonical SQL source: type='sql' with dialect='clickhouse'."""
        spec = get_algorithm_catalog().get_spec("xgboost")
        config = {
            "data": {
                "source": {
                    "type": "sql",
                    "dialect": "clickhouse",
                    "host": "127.0.0.1",
                    "sql": "SELECT 1",
                }
            }
        }
        result = resolve_data_source(spec, config)
        assert result["type"] == "sql"
        assert result["sql"] == "SELECT 1"
        assert result["dialect"] == "clickhouse"

    def test_canonical_iceberg_source(self) -> None:
        spec = get_algorithm_catalog().get_spec("xgboost")
        config = {
            "data": {
                "source": {
                    "type": "iceberg",
                    "catalog": "default",
                    "table": "db.tbl",
                }
            }
        }
        result = resolve_data_source(spec, config)
        assert result["type"] == "iceberg"
