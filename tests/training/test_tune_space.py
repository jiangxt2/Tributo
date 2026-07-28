"""Tests for tributo.training.tune_space module."""

from __future__ import annotations

import json

import pytest

from tributo.exceptions import JobConfigurationError
from tributo.training.tune_space import parse_search_space


@pytest.fixture
def sample_json(tmp_path):
    content = {
        "search_space": {
            "learning_rate": {"type": "loguniform", "lower": 0.001, "upper": 0.1},
            "max_depth": {"type": "choice", "values": [3, 5, 7, 9]},
            "n_estimators": {"type": "randint", "lower": 50, "upper": 500},
            "subsample": {"type": "uniform", "lower": 0.5, "upper": 1.0},
        }
    }
    p = tmp_path / "space.json"
    p.write_text(json.dumps(content))
    return str(p)


class TestParseSearchSpace:
    def test_parse_all_sampling_types(self, sample_json):
        result = parse_search_space(sample_json)
        assert "learning_rate" in result
        assert "max_depth" in result
        assert "n_estimators" in result
        assert "subsample" in result

    def test_parse_grid_search(self, tmp_path):
        p = tmp_path / "grid.json"
        p.write_text(
            json.dumps(
                {
                    "search_space": {
                        "batch_size": {"type": "grid_search", "values": [16, 32, 64]}
                    }
                }
            )
        )
        result = parse_search_space(str(p))
        assert "batch_size" in result

    def test_missing_search_space_key(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"other_key": {}}))
        with pytest.raises(JobConfigurationError, match="must contain 'search_space'"):
            parse_search_space(str(p))

    def test_rejects_yaml(self, tmp_path):
        p = tmp_path / "space.yaml"
        p.write_text("test")
        with pytest.raises(ValueError, match="YAML"):
            parse_search_space(str(p))

    def test_unsupported_sampling_type(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {"search_space": {"param": {"type": "unsupported_type", "value": 1}}}
            )
        )
        with pytest.raises(JobConfigurationError, match="Unsupported sampling type"):
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
        with pytest.raises(JobConfigurationError, match="requires lower < upper"):
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
        with pytest.raises(JobConfigurationError, match="requires positive 'q'"):
            parse_search_space(str(p))

    def test_choice_empty_values(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps({"search_space": {"lr": {"type": "choice", "values": []}}})
        )
        with pytest.raises(JobConfigurationError, match="requires non-empty list"):
            parse_search_space(str(p))
