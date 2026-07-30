"""Tests for tributo.training.config — merge, validation, and effective config pipeline."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import Field, ValidationError

from tributo._common.config import StrictConfigModel
from tributo._common.immutable import deep_thaw
from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    DataLoadingMode,
)
from tributo.training.config import (
    TrainingDataConfig,
    apply_dot_overrides,
    build_effective_config,
    merge_nested,
    resolve_data_source,
    validate_and_normalize_config,
    validate_execution_config,
)

# ---------------------------------------------------------------------------
# Dummy StrictConfigModel for testing
# ---------------------------------------------------------------------------


class SimpleModelConfig(StrictConfigModel):
    """Minimal model for testing the config pipeline."""

    training: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)


class FakeTrainer:
    pass


def _make_spec(
    name: str = "test",
    default_config: dict[str, Any] | None = None,
    config_model: type | None = SimpleModelConfig,
    data_loading: DataLoadingMode = DataLoadingMode.CANONICAL_DRIVER,
    **kwargs: Any,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=FakeTrainer,
        default_config=default_config or {},
        config_model=config_model,
        data_loading=data_loading,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# merge_nested
# ---------------------------------------------------------------------------


class TestMergeNested:
    def test_shallow_merge(self) -> None:
        result = merge_nested({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_replaces_leaf(self) -> None:
        result = merge_nested({"a": 1}, {"a": 99})
        assert result == {"a": 99}

    def test_nested_merge(self) -> None:
        result = merge_nested(
            {"training": {"lr": 0.01, "epochs": 10}},
            {"training": {"lr": 0.05}},
        )
        assert result == {"training": {"lr": 0.05, "epochs": 10}}

    def test_deeply_nested(self) -> None:
        result = merge_nested(
            {"a": {"b": {"c": 1, "d": 2}}},
            {"a": {"b": {"c": 99}}},
        )
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_override_adds_new_key_in_nested(self) -> None:
        result = merge_nested(
            {"training": {"lr": 0.01}},
            {"training": {"batch_size": 32}},
        )
        assert result == {"training": {"lr": 0.01, "batch_size": 32}}

    def test_base_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        base_snapshot = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = merge_nested(base, override)
        assert base == base_snapshot
        assert result == {"a": {"b": 1, "c": 2}}

    def test_override_not_mutated(self) -> None:
        override = {"a": {"b": 99}}
        override_snapshot = {"a": {"b": 99}}
        result = merge_nested({"a": {"c": 1}}, override)
        assert override == override_snapshot
        assert result == {"a": {"b": 99, "c": 1}}

    def test_no_shared_references(self) -> None:
        """Verify nested result dicts are independent from base."""
        base = {"a": {"b": [1, 2, 3]}}
        result = merge_nested(base, {})
        result["a"]["b"].append(4)
        assert base["a"]["b"] == [1, 2, 3]  # not mutated

    def test_empty_base(self) -> None:
        result = merge_nested({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override_is_copy(self) -> None:
        base = {"x": {"y": 1}}
        result = merge_nested(base, {})
        assert result == base
        assert result is not base  # different object


# ---------------------------------------------------------------------------
# apply_dot_overrides
# ---------------------------------------------------------------------------


class TestApplyDotOverrides:
    def test_simple_path(self) -> None:
        result = apply_dot_overrides({"a": 1}, {"a": 99})
        assert result == {"a": 99}

    def test_nested_path(self) -> None:
        result = apply_dot_overrides(
            {"training": {"lr": 0.01}},
            {"training.lr": 0.05},
        )
        assert result == {"training": {"lr": 0.05}}

    def test_deeply_nested_path(self) -> None:
        result = apply_dot_overrides(
            {"model": {"gradient_boosted_trees": {"max_depth": 3}}},
            {"model.gradient_boosted_trees.max_depth": 7},
        )
        assert result == {"model": {"gradient_boosted_trees": {"max_depth": 7}}}

    def test_creates_intermediate_dicts(self) -> None:
        result = apply_dot_overrides(
            {},
            {"a.b.c": 42},
        )
        assert result == {"a": {"b": {"c": 42}}}

    def test_input_not_mutated(self) -> None:
        config = {"x": 1}
        apply_dot_overrides(config, {"y": 2})
        assert config == {"x": 1}

    def test_multiple_overrides(self) -> None:
        result = apply_dot_overrides(
            {"a": {"x": 1, "y": 2}},
            {"a.x": 10, "a.y": 20},
        )
        assert result == {"a": {"x": 10, "y": 20}}

    def test_empty_dot_path_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="Invalid dot-path"):
            apply_dot_overrides({}, {"": 1})

    def test_double_dot_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="empty segment"):
            apply_dot_overrides({}, {"a..b": 1})

    def test_leading_dot_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="empty segment"):
            apply_dot_overrides({}, {".a": 1})

    def test_trailing_dot_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="empty segment"):
            apply_dot_overrides({}, {"a.": 1})

    def test_scalar_intermediate_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="Cannot descend"):
            apply_dot_overrides({"x": 42}, {"x.y": 99})


# ---------------------------------------------------------------------------
# validate_and_normalize_config
# ---------------------------------------------------------------------------


class TestValidateAndNormalizeConfig:
    def test_passes_valid_config(self) -> None:
        spec = _make_spec()
        result = validate_and_normalize_config(
            spec, {"training": {"lr": 0.01}, "model": {"type": "xgboost"}}
        )
        assert result["training"] == {"lr": 0.01}

    def test_rejects_unknown_field(self) -> None:
        spec = _make_spec()
        with pytest.raises(JobConfigurationError, match="Config validation failed"):
            validate_and_normalize_config(spec, {"typo_field": 1})

    def test_legacy_no_config_model(self) -> None:
        """When config_model is None, raw dict is returned as-is (thawed)."""
        spec = _make_spec(config_model=None)
        result = validate_and_normalize_config(spec, {"anything": "goes"})
        assert result == {"anything": "goes"}

    def test_input_not_mutated(self) -> None:
        spec = _make_spec()
        config = {"training": {"lr": 0.01}}
        original = deep_thaw(config)
        validate_and_normalize_config(spec, config)
        assert config == original


# ---------------------------------------------------------------------------
# validate_execution_config
# ---------------------------------------------------------------------------


class TestValidateExecutionConfig:
    def test_canonical_driver_requires_source(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(JobConfigurationError, match="requires 'data' section"):
            validate_execution_config(spec, {"training": {}}, datasets_supplied=False)

    def test_canonical_driver_skips_when_datasets_supplied(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        # Should not raise
        validate_execution_config(spec, {"training": {}}, datasets_supplied=True)

    def test_canonical_trainer_always_requires_source(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_TRAINER)
        with pytest.raises(JobConfigurationError):
            validate_execution_config(spec, {"training": {}}, datasets_supplied=True)

    def test_legacy_driver_skips_check(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.LEGACY_DRIVER)
        # Should not raise even without data section
        validate_execution_config(spec, {"training": {}}, datasets_supplied=False)

    def test_data_is_not_dict_raises(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(JobConfigurationError, match="requires 'data' section"):
            validate_execution_config(
                spec, {"data": "not_a_dict"}, datasets_supplied=False
            )

    def test_source_is_none_raises(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(JobConfigurationError, match="requires 'data.source'"):
            validate_execution_config(
                spec,
                {"data": {"source": None}},
                datasets_supplied=False,
            )


# ---------------------------------------------------------------------------
# build_effective_config (integration of the pipeline)
# ---------------------------------------------------------------------------


class TestBuildEffectiveConfig:
    def test_no_overrides_no_datasets(self) -> None:
        spec = _make_spec(
            default_config={"training": {"lr": 0.01}},
            data_loading=DataLoadingMode.LEGACY_DRIVER,
        )
        result = build_effective_config(spec, {"training": {"epochs": 10}})
        assert result["training"]["lr"] == 0.01
        assert result["training"]["epochs"] == 10

    def test_with_dot_overrides(self) -> None:
        spec = _make_spec(
            default_config={"training": {"lr": 0.01, "epochs": 10}},
            data_loading=DataLoadingMode.LEGACY_DRIVER,
        )
        result = build_effective_config(
            spec,
            {},
            dot_overrides={"training.lr": 0.05},
        )
        assert result["training"]["lr"] == 0.05
        assert result["training"]["epochs"] == 10

    def test_validation_catches_extra_field(self) -> None:
        spec = _make_spec()
        with pytest.raises(JobConfigurationError, match="Config validation failed"):
            build_effective_config(spec, {"unknown_section": {}})


# ---------------------------------------------------------------------------
# resolve_data_source
# ---------------------------------------------------------------------------


class TestResolveDataSource:
    def test_canonical_returns_validated_dict(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        config = {
            "data": {
                "source": {
                    "type": "parquet",
                    "path": "s3://bucket/data.parquet",
                }
            }
        }
        result = resolve_data_source(spec, config)
        assert result["type"] == "parquet"
        assert result["path"] == "s3://bucket/data.parquet"

    def test_canonical_missing_source_raises(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(JobConfigurationError, match="requires 'data.source'"):
            resolve_data_source(spec, {"data": {"not_source": True}})

    def test_canonical_invalid_source_raises(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(JobConfigurationError, match="Invalid data.source"):
            resolve_data_source(
                spec,
                {"data": {"source": {"type": "unknown_type", "path": ""}}},
            )

    def test_legacy_normalizes_flat_data(self) -> None:
        """Legacy flat data dict → SourceConfig model."""
        spec = _make_spec(data_loading=DataLoadingMode.LEGACY_DRIVER)
        result = resolve_data_source(
            spec,
            {"data": {"type": "s3", "uri": "s3://bkt/d.parquet"}},
        )
        assert result["type"] == "parquet"
        assert result["path"] == "s3://bkt/d.parquet"

    def test_missing_data_raises(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(
            JobConfigurationError, match="must contain a 'data' section"
        ):
            resolve_data_source(spec, {"training": {}})

    def test_data_is_not_dict_raises(self) -> None:
        spec = _make_spec(data_loading=DataLoadingMode.CANONICAL_DRIVER)
        with pytest.raises(
            JobConfigurationError, match="must contain a 'data' section"
        ):
            resolve_data_source(spec, {"data": "not_a_dict"})


# ---------------------------------------------------------------------------
# TrainingDataConfig
# ---------------------------------------------------------------------------


class TestTrainingDataConfig:
    def test_default(self) -> None:
        cfg = TrainingDataConfig()
        assert cfg.source is None

    def test_with_source(self) -> None:
        cfg = TrainingDataConfig(
            source={"type": "parquet", "path": "/tmp/data.parquet"}
        )
        assert cfg.source is not None

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrainingDataConfig(typo=1)  # type: ignore[call-arg]

    def test_rejects_non_source(self) -> None:
        """Non-discriminated dicts fail validation."""
        with pytest.raises(ValidationError):
            TrainingDataConfig(source={"type": "invalid"})


# ---------------------------------------------------------------------------
# Edge-case: regression test for deep_thaw usage in merge_nested
# ---------------------------------------------------------------------------


class TestMergeNestedEdgeCases:
    def test_list_value_in_nested(self) -> None:
        result = merge_nested(
            {"features": {"columns": ["a", "b"]}},
            {"features": {"columns": ["a", "b", "c"]}},
        )
        assert result == {"features": {"columns": ["a", "b", "c"]}}

    def test_none_value_override(self) -> None:
        result = merge_nested({"key": "original"}, {"key": None})
        assert result == {"key": None}
