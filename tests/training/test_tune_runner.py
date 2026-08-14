"""Tests for tributo.training.tune_runner module."""

from __future__ import annotations

import asyncio
import concurrent.futures
from array import array
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from tributo.algorithms.api import (
    DistributionSpec,
    DistributionStrategy,
    ExecutionProfile,
    FrameworkNativePolicy,
    InputDistribution,
    StateCoordination,
    WorkerRange,
    WorkerResources,
)
from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.training.algorithm_spec import AlgorithmSpec, Capability
from tributo.training.base import BaseTrainer
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_runner import (
    TuneRunner,
    _extract_target_metric,
    _safe_sampled_values,
    extract_best_params,
)


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
        capabilities=(Capability.TUNABLE,),
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


@pytest.fixture(autouse=True)
def tune_trial_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> MagicMock:
    """Provide the public Tune context required by every direct trainable call."""
    from ray import tune as ray_tune

    trial_dir = tmp_path / "trial-001"
    trial_dir.mkdir()
    context = MagicMock()
    context.get_trial_id.return_value = "trial-001"
    context.get_trial_name.return_value = "mock_trial"
    context.get_trial_dir.return_value = str(trial_dir)
    monkeypatch.setattr(ray_tune, "get_context", lambda: context)
    return context


class BrokenFloat(float):
    """Real scalar whose conversion fails."""

    def __float__(self) -> float:
        raise ValueError("conversion failed")


class TestTargetMetricExtraction:
    """Tests for the strict target metric boundary."""

    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            ({"loss": 1}, 1.0),
            ({"metrics": {"loss": 0.5}}, 0.5),
            ({"loss": 2, "metrics": {"loss": 2.0}}, 2.0),
            ({"metrics": {"loss": Fraction(1, 4)}}, 0.25),
            ({"metrics": {"loss": Fraction(3, 4)}}, 0.75),
        ],
    )
    def test_accepts_supported_finite_real_scalars(
        self,
        summary: object,
        expected: float,
    ) -> None:
        assert _extract_target_metric(summary, "loss") == expected

    def test_accepts_reserved_metric_name_from_nested_metrics(self) -> None:
        assert _extract_target_metric({"metrics": {"metrics": 0.3}}, "metrics") == 0.3

    def test_accepts_reserved_metric_name_from_direct_metrics(self) -> None:
        assert _extract_target_metric({"metrics": 0.3}, "metrics") == 0.3

    @pytest.mark.parametrize(
        ("summary", "message"),
        [
            ({"accuracy": 0.9}, "missing"),
            ({"loss": True}, "bool"),
            ({"loss": "0.5"}, "str"),
            ({"loss": [0.5]}, "list"),
            ({"loss": array("d", [0.5])}, "array"),
            ({"loss": float("nan")}, "finite"),
            ({"loss": float("inf")}, "finite"),
            ({"loss": float("-inf")}, "finite"),
            ({"loss": BrokenFloat(0.5)}, "could not be converted"),
            (None, "NoneType"),
        ],
    )
    def test_rejects_invalid_target_metric(
        self,
        summary: object,
        message: str,
    ) -> None:
        with pytest.raises(JobExecutionError, match=message) as caught:
            _extract_target_metric(summary, "loss")
        assert "loss" in str(caught.value)

    def test_rejects_conflicting_metric_locations(self) -> None:
        with pytest.raises(JobExecutionError, match="ambiguous"):
            _extract_target_metric(
                {"loss": 0.5, "metrics": {"loss": 0.6}},
                "loss",
            )

    def test_error_does_not_render_summary_values(self) -> None:
        secret = "should-not-appear-in-error"
        with pytest.raises(JobExecutionError) as caught:
            _extract_target_metric(
                {"metrics": {"accuracy": 0.9}, "credential": secret},
                "loss",
            )
        assert secret not in str(caught.value)


class TestSafeSampledValues:
    """Tests for bounded sampled-parameter diagnostics."""

    def test_redacts_credentials_and_non_scalar_values(self) -> None:
        safe = _safe_sampled_values(
            {
                "training.learning_rate": 0.05,
                "optimizer": "adam",
                "auth.access_key_id": "AKIA-SECRET",
                "nested": {"token": "secret"},
            }
        )

        assert safe == {
            "training.learning_rate": 0.05,
            "optimizer": "<str>",
            "auth.access_key_id": "<redacted>",
            "nested": "<dict>",
        }
        assert "AKIA-SECRET" not in repr(safe)


class TestTuneRunner:
    """Tests for TuneRunner class."""

    @staticmethod
    def _runner_for_trainer(
        trainer_cls: type[BaseTrainer],
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> TuneRunner:
        return TuneRunner(
            AlgorithmSpec(
                name="test-trainer",
                trainer_cls=trainer_cls,
                capabilities=(Capability.TUNABLE,),
            ),
            tune_config,
            search_space,
            effective_config,
        )

    def test_init_valid_config(
        self, trainer_spec, tune_config, search_space, effective_config
    ):
        """Test initialization with valid configuration."""
        runner = TuneRunner(trainer_spec, tune_config, search_space, effective_config)
        assert runner._trainer_spec == trainer_spec
        assert runner._tune_config == tune_config

    def test_random_search_delegates_concurrency_limit_to_ray(
        self, trainer_spec, search_space, effective_config
    ) -> None:
        """Ray must receive the concurrency limit with its default search generator."""
        tune_config = TuneSearchConfig(
            metric="loss",
            max_concurrent_trials=2,
            search_alg="random",
        )

        runner = TuneRunner(
            trainer_spec,
            tune_config,
            search_space,
            effective_config,
        )
        ray_tune_config = runner._build_tune_config()

        assert ray_tune_config.search_alg is None
        assert ray_tune_config.max_concurrent_trials == 2

    def test_distributed_trial_reserves_complete_worker_group(
        self,
        trainer_spec: AlgorithmSpec,
        tune_config: TuneSearchConfig,
        search_space: Any,
    ) -> None:
        distribution = DistributionSpec(
            strategy=DistributionStrategy.FRAMEWORK_NATIVE,
            supported_worker_range=WorkerRange(1, 8),
            supported_execution_profiles=(ExecutionProfile.LOCAL,),
            resources_per_worker=WorkerResources(
                num_cpus=2,
                custom={"accelerator_type_a": 0.25},
            ),
            input_distribution=InputDistribution.FRAMEWORK_OWNED,
            state_coordination=StateCoordination.FRAMEWORK_NATIVE,
            policy=FrameworkNativePolicy(
                framework="test-framework",
                evidence_collector_ref="tests.collector:evidence",
            ),
        )

        runner = TuneRunner(
            trainer_spec,
            tune_config,
            search_space,
            {"training": {"learning_rate": 0.01}, "ray": {"num_workers": 2}},
            distribution_spec=distribution,
        )

        assert runner._trial_resource_plan is not None
        placement = runner._trial_resource_plan.placement_group_factory
        assert placement.strategy == "SPREAD"
        assert placement.bundles == [
            {"CPU": 1.0},
            {"CPU": 2.0, "accelerator_type_a": 0.25},
            {"CPU": 2.0, "accelerator_type_a": 0.25},
        ]
        assert placement.required_resources == {
            "CPU": 5.0,
            "accelerator_type_a": 0.5,
        }

    def test_distributed_topology_cannot_be_tuned_as_hyperparameter(
        self,
        trainer_spec: AlgorithmSpec,
        tune_config: TuneSearchConfig,
    ) -> None:
        from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec

        distribution = DistributionSpec(
            strategy=DistributionStrategy.FRAMEWORK_NATIVE,
            supported_worker_range=WorkerRange(1, 8),
            supported_execution_profiles=(ExecutionProfile.LOCAL,),
            resources_per_worker=WorkerResources(),
            input_distribution=InputDistribution.FRAMEWORK_OWNED,
            state_coordination=StateCoordination.FRAMEWORK_NATIVE,
            policy=FrameworkNativePolicy(
                framework="test-framework",
                evidence_collector_ref="tests.collector:evidence",
            ),
        )
        topology_space = SearchSpaceSpec(
            parameters=(
                SearchParamSpec(
                    path="ray.num_workers",
                    kind="choice",
                    values=(1, 2),
                ),
            )
        )

        with pytest.raises(JobConfigurationError, match="not ordinary Tune"):
            TuneRunner(
                trainer_spec,
                tune_config,
                topology_space,
                {"ray": {"num_workers": 2}},
                distribution_spec=distribution,
            )

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
                "pu": {
                    "class_prior": 0.2,
                    "class_prior_method": "explicit",
                },
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

    def test_build_trainable_reports_only_target_metric(
        self, trainer_spec, tune_config, search_space, effective_config
    ):
        """The configured target is the only metric reported to Ray Tune."""
        from ray import tune as ray_tune

        runner = TuneRunner(trainer_spec, tune_config, search_space, effective_config)
        trainable = runner._build_trainable(
            datasets={"train": MagicMock()},
            output_path="/tmp/test",
        )
        assert callable(trainable)

        with patch.object(ray_tune, "report") as mock_report:
            trainable({"training.learning_rate": 0.05})
            mock_report.assert_called_once_with({"loss": 0.5})

    def test_build_trainable_raises_when_no_metrics(
        self, trainer_spec, tune_config, search_space, effective_config
    ):
        """An unrelated numeric field cannot satisfy the target metric."""

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

        with pytest.raises(JobExecutionError, match="target metric 'loss'.*missing"):
            trainable({"learning_rate": 0.05})

    def test_build_trainable_wraps_unknown_exception_with_cause(
        self, trainer_spec, tune_config, search_space, effective_config
    ):
        """Unknown fit failures use the Tune boundary error and retain cause."""

        class FailingTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                raise RuntimeError("fit failed")

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

        with pytest.raises(JobExecutionError, match="trial fit failed") as caught:
            trainable({"learning_rate": 0.05})
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == "fit failed"
        assert "FailingTrainer" not in str(caught.value)

    @pytest.mark.parametrize(
        "error",
        [
            JobConfigurationError("invalid trial config"),
            JobExecutionError("fit failed"),
        ],
    )
    def test_build_trainable_preserves_tributo_error(
        self,
        error: Exception,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        original_cause = RuntimeError("original framework cause")
        error.__cause__ = original_cause

        class FailingTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                raise error

        runner = self._runner_for_trainer(
            FailingTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        with pytest.raises(type(error)) as caught:
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})
        assert caught.value is error
        assert caught.value.__cause__ is original_cause

    def test_build_trainable_revalidates_before_construction(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        constructed = False

        class NeverConstructedTrainer(BaseTrainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                nonlocal constructed
                constructed = True
                super().__init__(*args, **kwargs)

            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return {"loss": 0.5}

        runner = self._runner_for_trainer(
            NeverConstructedTrainer,
            tune_config,
            search_space,
            effective_config,
        )
        trainable = runner._build_trainable({}, "/tmp/test")

        with pytest.raises(JobConfigurationError, match="non-mapping"):
            trainable({"training.learning_rate.value": 0.05})
        assert constructed is False

    def test_build_trainable_constructs_trainer_inside_each_call(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        instances: list[BaseTrainer] = []

        class TrackingTrainer(BaseTrainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                instances.append(self)

            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return {"loss": 0.5}

            def export_model(self, checkpoint: Any, output_path: str) -> None:
                if isinstance(checkpoint, dict):
                    self._summary.update(checkpoint)

        runner = self._runner_for_trainer(
            TrackingTrainer,
            tune_config,
            search_space,
            effective_config,
        )
        trainable = runner._build_trainable({}, "/tmp/test")
        assert instances == []

        from ray import tune as ray_tune

        with patch.object(ray_tune, "report"):
            trainable({"training.learning_rate": 0.05})
            trainable({"training.learning_rate": 0.06})
        assert len(instances) == 2
        assert instances[0] is not instances[1]

    def test_build_trainable_runs_fit_only_and_reports_checkpoint(
        self,
        tmp_path: Path,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        from ray.train import Checkpoint

        calls: list[str] = []
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        fit_result = MagicMock()
        fit_result.metrics = {"loss": 0.4}
        fit_result.checkpoint = Checkpoint.from_directory(checkpoint_dir)

        class OrderedTrainer(BaseTrainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                assert "callbacks" not in kwargs
                super().__init__(*args, **kwargs)

            def setup(self) -> None:
                calls.append("setup")

            def training_loop(self) -> Any:
                calls.append("fit")
                return fit_result

            def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                pytest.fail("Tune trials must not call BaseTrainer.run()")

            def export_artifacts(self, checkpoint: Any, output_path: str) -> None:
                pytest.fail("Tune trials must not export artifacts")

            def export_model(self, checkpoint: Any, output_path: str) -> None:
                pytest.fail("Tune trials must not publish legacy model artifacts")

        runner = self._runner_for_trainer(
            OrderedTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        from ray import tune as ray_tune

        with patch.object(ray_tune, "report") as report:
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})
        assert calls == ["setup", "fit"]
        report.assert_called_once()
        assert report.call_args.args == ({"loss": 0.4},)
        tune_checkpoint = report.call_args.kwargs["checkpoint"]
        assert tune_checkpoint.path == fit_result.checkpoint.path
        assert tune_checkpoint.filesystem.type_name == "local"

    def test_trial_run_config_is_unique_without_touching_ray_owned_paths(
        self,
        tmp_path: Path,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        from ray import tune as ray_tune

        run_configs: list[dict[str, Any]] = []

        class CapturingTrainer(BaseTrainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                run_configs.append(dict(self.run_config))

            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return {"loss": 0.5}

        trial_dirs = [tmp_path / "trial-one", tmp_path / "trial-two"]
        for trial_dir in trial_dirs:
            trial_dir.mkdir()
            (trial_dir / "ray-owned-state").write_text("keep")
        contexts = []
        for index, trial_dir in enumerate(trial_dirs, start=1):
            context = MagicMock()
            context.get_trial_id.return_value = f"trial-{index:03d}"
            context.get_trial_name.return_value = f"trial_name_{index}"
            context.get_trial_dir.return_value = str(trial_dir)
            contexts.append(context)

        runner = self._runner_for_trainer(
            CapturingTrainer,
            tune_config,
            search_space,
            effective_config,
        )
        trainable = runner._build_trainable({}, "/tmp/tune-root", "experiment")

        with (
            patch.object(ray_tune, "get_context", side_effect=contexts),
            patch.object(ray_tune, "report"),
        ):
            trainable({"training.learning_rate": 0.05})
            trainable({"training.learning_rate": 0.06})

        assert run_configs[0]["name"] != run_configs[1]["name"]
        assert run_configs[0]["storage_path"] == str(
            trial_dirs[0] / "_tributo_ray_train"
        )
        assert run_configs[1]["storage_path"] == str(
            trial_dirs[1] / "_tributo_ray_train"
        )
        assert all((path / "ray-owned-state").exists() for path in trial_dirs)

    def test_remote_trial_storage_uses_shared_output_root(
        self,
        tune_trial_context: MagicMock,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        from ray import tune as ray_tune

        run_configs: list[dict[str, Any]] = []
        tune_trial_context.get_trial_name.return_value = "secret=value"

        class CapturingTrainer(BaseTrainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                run_configs.append(dict(self.run_config))

            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return {"loss": 0.5}

        runner = self._runner_for_trainer(
            CapturingTrainer,
            tune_config,
            search_space,
            effective_config,
        )
        with patch.object(ray_tune, "report"):
            runner._build_trainable(
                {},
                "s3://bucket/tune-root",
                "experiment/name",
            )({"training.learning_rate": 0.05})

        assert run_configs[0]["storage_path"].startswith(
            "s3://bucket/tune-root/trials/experiment-name-"
        )
        assert run_configs[0]["storage_path"].endswith("/_tributo_ray_train/trial-001")
        assert "secret=value" not in run_configs[0]["name"]

    def test_missing_trial_identity_fails_before_trainer_construction(
        self,
        tune_trial_context: MagicMock,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        constructed = False
        tune_trial_context.get_trial_id.return_value = ""

        class NeverConstructedTrainer(BaseTrainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                nonlocal constructed
                constructed = True
                super().__init__(*args, **kwargs)

            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                return {"loss": 0.5}

        runner = self._runner_for_trainer(
            NeverConstructedTrainer,
            tune_config,
            search_space,
            effective_config,
        )
        with pytest.raises(JobExecutionError, match="invalid trial ID"):
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})
        assert constructed is False

    def test_setup_failure_prevents_training_loop(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        calls: list[str] = []

        class SetupFailingTrainer(BaseTrainer):
            def setup(self) -> None:
                calls.append("setup")
                raise JobExecutionError("setup failed")

            def training_loop(self) -> Any:
                calls.append("fit")
                return {"loss": 0.5}

        runner = self._runner_for_trainer(
            SetupFailingTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        with pytest.raises(JobExecutionError, match="setup failed"):
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})
        assert calls == ["setup"]

    def test_build_trainable_preserves_cancellation(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        cancellation = concurrent.futures.CancelledError()

        class CancelledTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                raise cancellation

        runner = self._runner_for_trainer(
            CancelledTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        with pytest.raises(concurrent.futures.CancelledError) as caught:
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})
        assert caught.value is cancellation

    def test_build_trainable_preserves_async_cancellation(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        class CancelledTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                raise asyncio.CancelledError()

        runner = self._runner_for_trainer(
            CancelledTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        with pytest.raises(asyncio.CancelledError):
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})

    def test_build_trainable_preserves_wrapped_cancellation(
        self,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        cancellation = concurrent.futures.CancelledError()
        wrapper = RuntimeError("cancelled task wrapper")
        wrapper.__cause__ = cancellation

        class CancelledTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                raise wrapper

        runner = self._runner_for_trainer(
            CancelledTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        with pytest.raises(RuntimeError) as caught:
            runner._build_trainable({}, "/tmp/test")({"training.learning_rate": 0.05})
        assert caught.value is wrapper
        assert caught.value.__cause__ is cancellation

    def test_build_trainable_wraps_report_failure_with_cause(
        self,
        trainer_spec: AlgorithmSpec,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        runner = TuneRunner(
            trainer_spec,
            tune_config,
            search_space,
            effective_config,
        )
        trainable = runner._build_trainable({}, "/tmp/test")

        from ray import tune as ray_tune

        report_error = RuntimeError("report failed")
        with (
            patch.object(ray_tune, "report", side_effect=report_error),
            pytest.raises(
                JobExecutionError,
                match="target metric reporting failed",
            ) as caught,
        ):
            trainable({"training.learning_rate": 0.05})
        assert caught.value.__cause__ is report_error

    def test_build_trainable_log_redacts_sampled_secrets(
        self,
        caplog: pytest.LogCaptureFixture,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        secret = "do-not-log-this-value"

        class FailingTrainer(BaseTrainer):
            def setup(self) -> None:
                pass

            def training_loop(self) -> Any:
                raise JobExecutionError("fit failed")

        runner = self._runner_for_trainer(
            FailingTrainer,
            tune_config,
            search_space,
            effective_config,
        )

        from ray import tune as ray_tune

        tune_context = MagicMock()
        tune_context.get_trial_id.return_value = "trial-123"
        tune_context.get_trial_name.return_value = "redaction-test"
        tune_context.get_trial_dir.return_value = "/tmp/redaction-test"
        with (
            patch.object(ray_tune, "get_context", return_value=tune_context),
            caplog.at_level("ERROR"),
            pytest.raises(JobExecutionError),
        ):
            runner._build_trainable({}, "/tmp/test")(
                {
                    "training.learning_rate": 0.05,
                    "auth.access_token": secret,
                }
            )
        assert "trial_id=trial-123" in caplog.text
        assert "training.learning_rate" in caplog.text
        assert "auth.access_token" in caplog.text
        assert secret not in caplog.text

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

    @patch("tributo.training.tune_runner.Tuner")
    def test_run_preserves_result_grid_and_summary_contract(
        self,
        tuner_cls: MagicMock,
        trainer_spec: AlgorithmSpec,
        tune_config: TuneSearchConfig,
        search_space: Any,
        effective_config: dict[str, Any],
    ) -> None:
        best_result = MagicMock()
        best_result.metrics = {"loss": 0.25}
        best_result.config = {"training.learning_rate": 0.02}
        result_grid = MagicMock()
        result_grid.get_best_result.return_value = best_result
        result_grid.__len__.return_value = 2
        result_grid.num_errors = 0
        tuner_cls.return_value.fit.return_value = result_grid

        runner = TuneRunner(
            trainer_spec,
            tune_config,
            search_space,
            effective_config,
        )

        with patch("tributo._common.storage.write_json") as write_json:
            returned = runner.run(
                datasets={},
                output_path="/tmp/tune-results",
                experiment_name="contract-test",
            )

        assert returned is result_grid
        tuner_cls.return_value.fit.assert_called_once_with()
        assert (
            tuner_cls.call_args.kwargs["run_config"].storage_path
            == "/tmp/tune-results/trials"
        )
        assert result_grid.get_best_result.call_args_list == [
            call(),
            call(metric="loss", mode="min"),
        ]
        write_json.assert_called_once_with(
            "/tmp/tune-results/tune_summary.json",
            {
                "experiment_name": "contract-test",
                "metric": "loss",
                "mode": "min",
                "num_samples": 2,
                "search_alg": "random",
                "scheduler": "fifo",
                "best_params": {"training.learning_rate": 0.02},
                "num_trials": 2,
                "num_errors": 0,
            },
        )


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
