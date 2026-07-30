"""Tests for tributo.training.tune_space module (IR + adapters)."""

from __future__ import annotations

import json

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.training.tune_space import (
    SearchParamSpec,
    SearchSpaceSpec,
    parse_search_space,
    resolve_local_overrides,
    to_ray_param_space,
    validate_search_targets,
)


@pytest.fixture
def sample_json(tmp_path):
    content = {
        "search_space": {
            "training.learning_rate": {
                "type": "loguniform",
                "lower": 0.001,
                "upper": 0.1,
            },
            "model.max_depth": {"type": "choice", "values": [3, 5, 7, 9]},
            "training.n_estimators": {"type": "randint", "lower": 50, "upper": 500},
            "training.subsample": {"type": "uniform", "lower": 0.5, "upper": 1.0},
        }
    }
    p = tmp_path / "space.json"
    p.write_text(json.dumps(content))
    return str(p)


class TestParseSearchSpace:
    def test_parse_all_sampling_types(self, sample_json):
        result = parse_search_space(sample_json)
        assert isinstance(result, SearchSpaceSpec)
        paths = {p.path for p in result.parameters}
        assert "training.learning_rate" in paths
        assert "model.max_depth" in paths
        assert "training.n_estimators" in paths
        assert "training.subsample" in paths

    def test_parse_grid_search(self, tmp_path):
        p = tmp_path / "grid.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "training.batch_size": {
                            "type": "grid_search",
                            "values": [16, 32, 64],
                        }
                    }
                }
            )
        )
        result = parse_search_space(str(p))
        paths = {p.path for p in result.parameters}
        assert "training.batch_size" in paths

    def test_missing_search_space_key(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"other_key": {}}))
        with pytest.raises(JobConfigurationError, match="must contain 'search_space'"):
            parse_search_space(str(p))

    def test_rejects_yaml(self, tmp_path):
        p = tmp_path / "space.yaml"
        p.write_text("test")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_search_space(str(p))

    def test_unsupported_sampling_type(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {"search_space": {"param": {"type": "unsupported_type", "value": 1}}}
            )
        )
        with pytest.raises(JobConfigurationError, match="unknown type"):
            parse_search_space(str(p))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_search_space("/nonexistent.json")

    def test_missing_lower_upper(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps({"search_space": {"lr": {"type": "uniform", "lower": 0.01}}})
        )
        with pytest.raises(JobConfigurationError, match="requires 'lower' and 'upper'"):
            parse_search_space(str(p))

    def test_lower_gte_upper(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "lr": {"type": "uniform", "lower": 0.5, "upper": 0.1}
                    }
                }
            )
        )
        with pytest.raises(JobConfigurationError, match="lower"):
            parse_search_space(str(p))

    def test_quantized_missing_q(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "batch_size": {"type": "qrandint", "lower": 1, "upper": 100}
                    }
                }
            )
        )
        with pytest.raises(JobConfigurationError, match="requires 'q'"):
            parse_search_space(str(p))

    def test_choice_empty_values(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps({"search_space": {"lr": {"type": "choice", "values": []}}})
        )
        with pytest.raises(JobConfigurationError, match="non-empty"):
            parse_search_space(str(p))


class TestSearchSpaceValidation:
    def test_rejects_data_path(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "data.source.type": {"type": "choice", "values": ["csv"]}
                    }
                }
            )
        )
        with pytest.raises(JobConfigurationError, match="data"):
            parse_search_space(str(p))

    def test_rejects_output_path(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "output.onnx_path": {"type": "choice", "values": ["a.onnx"]}
                    }
                }
            )
        )
        with pytest.raises(JobConfigurationError, match="output"):
            parse_search_space(str(p))

    def test_rejects_search_prefix_conflict(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "model": {"type": "choice", "values": [{}]},
                        "model.max_depth": {"type": "choice", "values": [3]},
                    }
                }
            )
        )
        with pytest.raises(JobConfigurationError):
            parse_search_space(str(p))


class TestValidateSearchTargets:
    def test_rejects_scalar_parent(self):
        """Search path targeting inside a scalar should fail."""
        with pytest.raises(JobConfigurationError, match="cannot descend"):
            validate_search_targets(
                {"model": "not_a_dict"},
                SearchSpaceSpec(
                    parameters=(_param("model.max_depth", "choice", values=[3]),)
                ),
            )


class TestResolveLocalOverrides:
    def test_uses_explicit_default(self):
        space = SearchSpaceSpec(
            parameters=(
                _param(
                    "training.lr", "loguniform", lower=0.001, upper=0.1, default=0.01
                ),
            )
        )
        overrides = resolve_local_overrides(space, {})
        assert overrides == {"training.lr": 0.01}

    def test_falls_back_to_effective_base(self):
        space = SearchSpaceSpec(
            parameters=(_param("training.lr", "loguniform", lower=0.001, upper=0.1),)
        )
        overrides = resolve_local_overrides(space, {"training": {"lr": 0.05}})
        assert overrides == {"training.lr": 0.05}

    def test_no_default_or_base_raises(self):
        space = SearchSpaceSpec(
            parameters=(_param("training.lr", "loguniform", lower=0.001, upper=0.1),)
        )
        with pytest.raises(JobConfigurationError, match="no default"):
            resolve_local_overrides(space, {})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# validate_search_targets — Phase 3 additions
# ---------------------------------------------------------------------------


class TestValidateSearchTargetsComprehensive:
    def test_valid_path_passes(self) -> None:
        """Non-mapping leaf in an existing config tree is valid."""
        validate_search_targets(
            {"training": {"lr": 0.01}},
            SearchSpaceSpec(
                parameters=(_param("training.lr", "uniform", lower=0.001, upper=0.1),)
            ),
        )

    def test_entire_section_is_mapping_raises(self) -> None:
        """Search param targeting a mapping node should fail."""
        with pytest.raises(JobConfigurationError, match="target is a mapping"):
            validate_search_targets(
                {"model": {"gbm": {"depth": 3}}},
                SearchSpaceSpec(parameters=(_param("model", "choice", values=[{}]),)),
            )

    def test_entire_section_is_mapping_at_leaf_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="target is a mapping"):
            validate_search_targets(
                {"model": {"gbm": {"depth": 3}}},
                SearchSpaceSpec(
                    parameters=(_param("model.gbm", "choice", values=[{}]),)
                ),
            )

    def test_segment_not_found_raises(self) -> None:
        with pytest.raises(JobConfigurationError, match="not found"):
            validate_search_targets(
                {"training": {"lr": 0.01}},
                SearchSpaceSpec(
                    parameters=(
                        _param("nonexistent.lr", "uniform", lower=0.001, upper=0.1),
                    )
                ),
            )


# ---------------------------------------------------------------------------
# warn_search_space_conflicts
# ---------------------------------------------------------------------------


class TestWarnSearchSpaceConflicts:
    def test_no_conflict_when_not_in_config(self, caplog) -> None:
        from tributo.training.tune_space import warn_search_space_conflicts

        with caplog.at_level("WARNING"):
            warn_search_space_conflicts(
                {"something_else": 42},
                SearchSpaceSpec(
                    parameters=(
                        _param("training.lr", "uniform", lower=0.001, upper=0.1),
                    )
                ),
            )
        assert "Search param" not in caplog.text

    def test_warns_when_in_config(self, caplog) -> None:
        from tributo.training.tune_space import warn_search_space_conflicts

        with caplog.at_level("WARNING"):
            warn_search_space_conflicts(
                {"training": {"lr": 0.01}},
                SearchSpaceSpec(
                    parameters=(
                        _param("training.lr", "uniform", lower=0.001, upper=0.1),
                    )
                ),
            )
        assert "also set in training config" in caplog.text


# ---------------------------------------------------------------------------
# to_ray_param_space — lazy import, RAy Tune objects
# ---------------------------------------------------------------------------


class TestToRayParamSpace:
    def test_returns_dict(self) -> None:
        space = SearchSpaceSpec(
            parameters=(
                _param("training.lr", "loguniform", lower=0.001, upper=0.1),
                _param("model.max_depth", "choice", values=[3, 5, 7]),
            )
        )
        result = to_ray_param_space(space)
        assert isinstance(result, dict)
        assert "training.lr" in result
        assert "model.max_depth" in result

    def test_ray_objects_are_domain_or_dict(self) -> None:
        """Ray domain objects are either sample.Domain or dict (grid_search)."""
        space = SearchSpaceSpec(
            parameters=(
                _param("lr", "uniform", lower=0.0, upper=1.0),
                _param("bs", "grid_search", values=(16, 32, 64)),
            )
        )
        result = to_ray_param_space(space)
        # Continuous domains are sample.Domain objects; grid_search is a dict.
        assert hasattr(result["lr"], "sampler")
        assert isinstance(result["bs"], dict)

    def test_all_kinds_convert(self) -> None:
        """Smoke-test all 10 search kinds."""
        params = (
            SearchParamSpec(path="p.uniform", kind="uniform", lower=0.0, upper=1.0),
            SearchParamSpec(
                path="p.loguniform", kind="loguniform", lower=0.001, upper=1.0
            ),
            SearchParamSpec(
                path="p.quniform", kind="quniform", lower=1.0, upper=10.0, q=1.0
            ),
            SearchParamSpec(
                path="p.qloguniform", kind="qloguniform", lower=0.1, upper=1.0, q=0.1
            ),
            SearchParamSpec(path="p.randint", kind="randint", lower=1, upper=100),
            SearchParamSpec(path="p.lograndint", kind="lograndint", lower=1, upper=100),
            SearchParamSpec(
                path="p.qrandint", kind="qrandint", lower=1, upper=100, q=5
            ),
            SearchParamSpec(
                path="p.qlograndint", kind="qlograndint", lower=1, upper=100, q=5
            ),
            SearchParamSpec(path="p.choice", kind="choice", values=(1, 2, 3)),
            SearchParamSpec(path="p.grid_search", kind="grid_search", values=(16, 32)),
        )
        space = SearchSpaceSpec(parameters=params)
        result = to_ray_param_space(space)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# resolve_local_overrides — extended
# ---------------------------------------------------------------------------


class TestResolveLocalOverridesExtended:
    def test_mixed_default_and_base(self) -> None:
        space = SearchSpaceSpec(
            parameters=(
                _param("a.x", "uniform", lower=0.0, upper=1.0, default=0.5),
                _param("a.y", "uniform", lower=0.0, upper=1.0),
            )
        )
        overrides = resolve_local_overrides(space, {"a": {"y": 0.99}})
        assert overrides == {"a.x": 0.5, "a.y": 0.99}

    def test_deeply_nested_base_fallback(self) -> None:
        space = SearchSpaceSpec(
            parameters=(
                _param("model.params.lr", "loguniform", lower=0.001, upper=0.1),
            )
        )
        overrides = resolve_local_overrides(
            space, {"model": {"params": {"lr": 0.01, "bs": 32}}}
        )
        assert overrides == {"model.params.lr": 0.01}


def _param(path, kind, **kwargs):
    from tributo.training.tune_space import SearchParamSpec

    return SearchParamSpec(path=path, kind=kind, **kwargs)
