"""Tests for tributo.training.tune_runner module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.training.base import BaseTrainer
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_runner import TuneRunner, extract_best_params


class MockTrainer(BaseTrainer):
    """Mock trainer for testing."""

    def setup(self) -> None:
        pass

    def training_loop(self) -> Any:
        return {"metrics": {"loss": 0.5, "accuracy": 0.9}}

    def export_model(self, checkpoint: Any, output_path: str) -> None:
        self._summary["output_path"] = output_path
        if isinstance(checkpoint, dict):
            self._summary.update(checkpoint)


@pytest.fixture
def trainer_spec():
    """Create a mock AlgorithmSpec."""
    return AlgorithmSpec(
        name="mock",
        trainer_cls=MockTrainer,
        default_config={"learning_rate": 0.01},
        supported_tasks=["train"],
    )


@pytest.fixture
def tune_config():
    """Create a TuneSearchConfig for testing."""
    return TuneSearchConfig(
        metric="loss",
        mode="min",
        num_samples=2,
        search_alg="random",
        scheduler="fifo",
    )


@pytest.fixture
def search_space():
    """Create a simple SearchSpaceSpec."""
    from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec

    return SearchSpaceSpec(
        parameters=(
            SearchParamSpec(
                path="training.learning_rate",
                kind="uniform",
                lower=0.001,
                upper=0.1,
            ),
        )
    )


@pytest.fixture
def effective_config():
    """Minimal effective config."""
    return {"training": {"learning_rate": 0.01}}


class TestTuneRunner:
    """Tests for TuneRunner class."""

    def test_init_valid_config(
        self, trainer_spec, tune_config, search_space, effective_config
    ):
        """Test initialization with valid configuration."""
        runner = TuneRunner(trainer_spec, tune_config, search_space, effective_config)
        assert runner._trainer_spec == trainer_spec
        assert runner._tune_config == tune_config

    def test_legacy_runner_rejects_portable_registration(
        self, tune_config, search_space, effective_config
    ) -> None:
        runner = TuneRunner(
            AlgorithmSpec(
                name="portable",
                trainer_cls=None,
                operations=("fit",),
            ),
            tune_config,
            search_space,
            effective_config,
        )

        with pytest.raises(JobConfigurationError, match="portable execution path"):
            runner._build_trainable({}, "/tmp/test")

    def test_pu_trial_revalidates_explicit_class_prior(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
    ) -> None:
        """A sampled PU config cannot reach trainer construction without a prior."""
        from tributo.training.algorithm_spec import DataLoadingMode
        from tributo.training.pu_trainer import PUTrainingConfig

        pu_spec = AlgorithmSpec(
            name="pu-test",
            trainer_cls=MockTrainer,
            config_model=PUTrainingConfig,
            data_loading=DataLoadingMode.CANONICAL_TRAINER,
        )
        runner = TuneRunner(
            pu_spec,
            tune_config,
            search_space,
            {
                "data": {
                    "source": {
                        "type": "parquet",
                        "path": "/tmp/pu-train.parquet",
                    }
                },
                "pu": {"class_prior": 0.2},
            },
        )
        trainable = runner._build_trainable({}, "/tmp/test")

        with pytest.raises(JobConfigurationError, match="class_prior"):
            trainable({"pu.class_prior": None})

    def test_init_invalid_search_alg(
        self, trainer_spec, search_space, effective_config
    ):
        """Test initialization with invalid search algorithm."""
        with pytest.raises(ValueError):
            TuneSearchConfig(search_alg="invalid")

    def test_init_invalid_scheduler(self, trainer_spec, search_space, effective_config):
        """Test initialization with invalid scheduler."""
        with pytest.raises(ValueError):
            TuneSearchConfig(scheduler="invalid")

    @patch("tributo.training.tune_runner.Tuner")
    def test_build_trainable_reports_metrics(
        self, mock_tuner_cls, trainer_spec, tune_config, search_space, effective_config
    ):
        """Test that trainable function reports metrics correctly."""
        from ray import tune as ray_tune

        runner = TuneRunner(trainer_spec, tune_config, search_space, effective_config)
        trainable = runner._build_trainable(
            datasets={"train": MagicMock()},
            output_path="/tmp/test",
        )
        assert callable(trainable)

        with patch.object(ray_tune, "report") as mock_report:
            trainable({"training.learning_rate": 0.05})
            mock_report.assert_called_once()
            reported = mock_report.call_args[0][0]
            assert "loss" in reported
            assert reported["loss"] == 0.5
            assert "accuracy" in reported
            assert reported["accuracy"] == 0.9

    @patch("tributo.training.tune_runner.Tuner")
    def test_build_trainable_raises_when_no_metrics(
        self, mock_tuner_cls, trainer_spec, tune_config, search_space, effective_config
    ):
        """Test that trainable raises when trainer returns no numeric metrics."""

        class NoMetricTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return {"status": "done"}

            def export_model(self, checkpoint: Any, output_path: str) -> None:
                pass

        no_metric_spec = AlgorithmSpec(
            name="no_metric",
            trainer_cls=NoMetricTrainer,
            default_config={},
        )
        runner = TuneRunner(no_metric_spec, tune_config, search_space, effective_config)
        trainable = runner._build_trainable(
            datasets={"train": MagicMock()},
            output_path="/tmp/test",
        )

        with pytest.raises(JobExecutionError, match="no numeric metrics"):
            trainable({"learning_rate": 0.05})

    @patch("tributo.training.tune_runner.Tuner")
    def test_build_trainable_wraps_exception(
        self, mock_tuner_cls, trainer_spec, tune_config, search_space, effective_config
    ):
        """Test that trainable wraps trainer exceptions with context."""

        class FailingTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return None

            def export_model(self, checkpoint: Any, output_path: str) -> None:
                raise RuntimeError("export failed")

        failing_spec = AlgorithmSpec(
            name="failing",
            trainer_cls=FailingTrainer,
            default_config={},
        )
        runner = TuneRunner(failing_spec, tune_config, search_space, effective_config)
        trainable = runner._build_trainable(
            datasets={"train": MagicMock()},
            output_path="/tmp/test",
        )

        with pytest.raises(JobExecutionError):
            trainable({"learning_rate": 0.05})

    def test_bayesopt_requires_dependency(
        self, trainer_spec, search_space, effective_config
    ):
        """Test that BayesOpt raises ImportError when not installed."""
        tune_config = TuneSearchConfig(search_alg="bayesopt")

        with patch.dict(
            "tributo.training.tune_runner._SEARCH_ALG_MAP",
            {
                "bayesopt": lambda metric, mode: (_ for _ in ()).throw(
                    ImportError("bayesian-optimization is required")
                )
            },
        ):
            with pytest.raises(ImportError):
                TuneRunner(trainer_spec, tune_config, search_space, effective_config)


class TestExtractBestParams:
    """Tests for extract_best_params function."""

    def test_extract_best_params(self):
        """Test extracting best parameters from ResultGrid."""
        # Mock ResultGrid
        mock_result = MagicMock()
        mock_result.config = {"learning_rate": 0.01, "max_depth": 5}

        mock_result_grid = MagicMock()
        mock_result_grid.get_best_result.return_value = mock_result

        result = extract_best_params(mock_result_grid, metric="loss", mode="min")
        assert result == {"learning_rate": 0.01, "max_depth": 5}
        mock_result_grid.get_best_result.assert_called_once_with(
            metric="loss", mode="min"
        )

    def test_extract_best_params_missing_config(self):
        """Test error when best result has no config."""
        mock_result = MagicMock()
        mock_result.config = None

        mock_result_grid = MagicMock()
        mock_result_grid.get_best_result.return_value = mock_result

        with pytest.raises(JobExecutionError, match="No valid config found"):
            extract_best_params(mock_result_grid, metric="loss", mode="min")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
