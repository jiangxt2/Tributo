"""Tests for tributo.training.tune_config module."""

from __future__ import annotations

import pytest

from tributo.training.tune_config import TuneSearchConfig


class TestTuneSearchConfig:
    """Tests for TuneSearchConfig Pydantic model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TuneSearchConfig()
        assert config.metric == "loss"
        assert config.mode == "min"
        assert config.num_samples == 1
        assert config.max_concurrent_trials is None
        assert config.time_budget_s is None
        assert config.search_alg == "random"
        assert config.scheduler == "fifo"
        assert config.fail_fast is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TuneSearchConfig(
            metric="accuracy",
            mode="max",
            num_samples=10,
            max_concurrent_trials=4,
            time_budget_s=3600.0,
            search_alg="bayesopt",
            scheduler="asha",
            fail_fast=True,
        )
        assert config.metric == "accuracy"
        assert config.mode == "max"
        assert config.num_samples == 10
        assert config.max_concurrent_trials == 4
        assert config.time_budget_s == 3600.0
        assert config.search_alg == "bayesopt"
        assert config.scheduler == "asha"
        assert config.fail_fast is True

    def test_invalid_mode(self):
        """Test validation error for invalid mode."""
        with pytest.raises(ValueError):
            TuneSearchConfig(mode="invalid")

    def test_invalid_search_alg(self):
        """Test validation error for invalid search algorithm."""
        with pytest.raises(ValueError):
            TuneSearchConfig(search_alg="invalid")

    def test_invalid_scheduler(self):
        """Test validation error for invalid scheduler."""
        with pytest.raises(ValueError):
            TuneSearchConfig(scheduler="invalid")

    def test_invalid_num_samples(self):
        """Test validation error for num_samples < 1."""
        with pytest.raises(ValueError):
            TuneSearchConfig(num_samples=0)

    def test_invalid_max_concurrent_trials(self):
        """Test validation error for max_concurrent_trials < 1."""
        with pytest.raises(ValueError):
            TuneSearchConfig(max_concurrent_trials=0)

    def test_invalid_time_budget(self):
        """Test validation error for time_budget_s <= 0."""
        with pytest.raises(ValueError):
            TuneSearchConfig(time_budget_s=-1.0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            TuneSearchConfig(unknown_field="value")

    def test_model_dump(self):
        """Test model serialization."""
        config = TuneSearchConfig(metric="accuracy", num_samples=5)
        data = config.model_dump()
        assert data["metric"] == "accuracy"
        assert data["num_samples"] == 5
        assert data["search_alg"] == "random"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
