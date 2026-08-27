"""Integration tests for the algorithm catalog and legacy config pipeline.

The Core test suite owns the framework contract, not any concrete algorithm.
Every executable algorithm fixture in this module is therefore registered by
the test itself, exactly as a third-party compatibility Wheel would register it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pydantic import BaseModel, ConfigDict, Field

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
    """Minimal compatibility Trainer identity used by contract fixtures."""


class _TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_rounds: int = 10
    seed: int = 0


class _DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: dict[str, object] | None = None


class _AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: _DataConfig = Field(default_factory=_DataConfig)
    training: _TrainingConfig = Field(default_factory=_TrainingConfig)


def _spec(
    name: str = "test_pipeline_algorithm",
    *,
    data_loading: DataLoadingMode = DataLoadingMode.CANONICAL_DRIVER,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=FakeTrainer,
        problem_types=(ProblemType.REGRESSION,),
        data_modality=("tabular",),
        config_model=_AlgorithmConfig,
        data_loading=data_loading,
    )


@contextmanager
def _registered(spec: AlgorithmSpec) -> Iterator[AlgorithmSpec]:
    _registry.register(spec.name, spec)
    try:
        yield spec
    finally:
        _registry.unregister(spec.name)


class TestIntegrityGate:
    def test_integrity_rejects_missing_replacement(self) -> None:
        registry: Registry[str, AlgorithmSpec] = Registry("test")
        registry.register(
            "old",
            AlgorithmSpec(
                name="old",
                trainer_cls=FakeTrainer,
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="nonexistent",
            ),
        )
        catalog = AlgorithmCatalog(registry)
        with pytest.raises(JobConfigurationError, match="not found in registry"):
            catalog.validate_integrity()

    def test_integrity_rejects_cycle(self) -> None:
        registry: Registry[str, AlgorithmSpec] = Registry("test")
        registry.register(
            "a",
            AlgorithmSpec(
                name="a",
                trainer_cls=FakeTrainer,
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="b",
            ),
        )
        registry.register(
            "b",
            AlgorithmSpec(
                name="b",
                trainer_cls=FakeTrainer,
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="a",
            ),
        )
        catalog = AlgorithmCatalog(registry)
        with pytest.raises(JobConfigurationError, match="cycle"):
            catalog.validate_integrity()

    def test_shared_registry_is_valid_on_import(self) -> None:
        get_algorithm_catalog().validate_integrity()


class TestSnapshotFreshness:
    def test_new_registration_is_immediately_visible(self) -> None:
        registry: Registry[str, AlgorithmSpec] = Registry("test")
        catalog = AlgorithmCatalog(registry)
        registry.register("fresh", _spec("fresh"))

        assert catalog.list(problem_type=ProblemType.REGRESSION) == ["fresh"]

    def test_shared_catalog_observes_programmatic_registration(self) -> None:
        spec = _spec("_test_snapshot")
        with _registered(spec):
            assert spec.name in get_algorithm_catalog().list()


class TestPipeline:
    def test_minimal_config_uses_declared_defaults(self) -> None:
        config = build_effective_config(_spec(), {}, datasets_supplied=True)

        assert config["training"] == {"num_rounds": 10, "seed": 0}

    def test_unknown_top_level_field_is_rejected(self) -> None:
        with pytest.raises(JobConfigurationError, match="Config validation failed"):
            build_effective_config(_spec(), {"typo_field": 123})

    def test_user_config_overrides_defaults(self) -> None:
        config = build_effective_config(
            _spec(),
            {"training": {"num_rounds": 42, "seed": 7}},
            datasets_supplied=True,
        )

        assert config["training"] == {"num_rounds": 42, "seed": 7}

    def test_dot_overrides_are_applied(self) -> None:
        config = build_effective_config(
            _spec(),
            {},
            dot_overrides={"training.num_rounds": 77},
            datasets_supplied=True,
        )

        assert config["training"]["num_rounds"] == 77

    def test_canonical_driver_requires_source_without_supplied_dataset(self) -> None:
        with pytest.raises(JobConfigurationError, match="requires 'data.source'"):
            build_effective_config(_spec(), {})


class TestDataLoadingRouting:
    def test_canonical_driver_accepts_preloaded_dataset(self) -> None:
        validate_execution_config(_spec(), {}, datasets_supplied=True)

    def test_canonical_trainer_always_requires_source(self) -> None:
        spec = _spec(data_loading=DataLoadingMode.CANONICAL_TRAINER)
        with pytest.raises(JobConfigurationError, match="source"):
            validate_execution_config(spec, {}, datasets_supplied=True)

    def test_legacy_driver_skips_source_validation(self) -> None:
        spec = _spec(data_loading=DataLoadingMode.LEGACY_DRIVER)
        validate_execution_config(
            spec,
            {"anything": "goes"},
            datasets_supplied=False,
        )


class TestResolveDataSource:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            (
                {"type": "parquet", "path": "/tmp/data.parquet"},
                {"type": "parquet", "path": "/tmp/data.parquet"},
            ),
            (
                {"type": "csv", "path": "/tmp/data.csv"},
                {"type": "csv", "path": "/tmp/data.csv"},
            ),
            (
                {
                    "type": "sql",
                    "dialect": "clickhouse",
                    "host": "127.0.0.1",
                    "sql": "SELECT 1",
                },
                {
                    "type": "sql",
                    "dialect": "clickhouse",
                    "host": "127.0.0.1",
                    "sql": "SELECT 1",
                },
            ),
            (
                {"type": "iceberg", "catalog": "default", "table": "db.tbl"},
                {"type": "iceberg", "catalog": "default", "table": "db.tbl"},
            ),
        ],
    )
    def test_canonical_source_is_returned_without_algorithm_branching(
        self,
        source: dict[str, object],
        expected: dict[str, object],
    ) -> None:
        result = resolve_data_source(_spec(), {"data": {"source": source}})

        assert all(result[key] == value for key, value in expected.items())
