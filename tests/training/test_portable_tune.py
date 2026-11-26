"""Tests for fit-only Tune execution of portable algorithm Wheels."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from ray import cloudpickle
from ray.tune.integration.ray_train import TuneReportCallback

from tests.algorithms.conftest import map_reduce_registration, request_for
from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRequest,
    DistributionStrategy,
    ExecutionProfile,
    ExecutionRequest,
    FrameworkNativePolicy,
)
from tributo.algorithms.core import AlgorithmPlanner, AlgorithmRegistrationRegistry
from tributo.algorithms.input import FakeInputResolver
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.training.algorithm_spec import Capability
from tributo.training.portable_tune import (
    PortableTuneRunner,
    _fit_only_plan,
    _IntermediateReportCallback,
    _trial_checkpoint_enabled,
    _trial_reporter,
    _trial_request,
)
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec


def _resolved_plan():
    registration = map_reduce_registration()
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})
    return planner.plan(
        ExecutionRequest(
            algorithm_request=request_for(
                "external_map_reduce", AlgorithmOperation.FIT
            ),
            profile=ExecutionProfile.CLUSTER,
            worker_count=2,
        )
    )


def test_fit_only_plan_rebinds_runtime_digest_and_disables_output_contract() -> None:
    plan = _fit_only_plan(_resolved_plan())
    assert plan.distribution_spec is not None
    assert plan.distribution_spec.result_policy.value == "fit_only"
    assert plan.runtime.distribution_digest == plan.distribution_spec.digest
    assert plan.contract_bindings is None
    plan.validate_integrity()


def test_trial_request_applies_only_sampled_config_and_isolates_checkpoint() -> None:
    base = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="external",
            operation=AlgorithmOperation.FIT,
            input_binding=request_for("external", AlgorithmOperation.FIT).input_binding,
            algorithm_config={
                "learning_rate": 0.1,
                "runtime": {"checkpoint_dir": "/old"},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
        resume_from="/resume",
    )
    trial = _trial_request(
        base,
        {"learning_rate": 0.2},
        checkpoint_dir="/trial/checkpoint",
    )
    assert trial.resume_from is None
    assert trial.algorithm_request.algorithm_config["learning_rate"] == 0.2
    assert trial.algorithm_request.algorithm_config["runtime"] == {
        "checkpoint_dir": "/trial/checkpoint"
    }


def test_torch_tune_trial_is_metric_only() -> None:
    torch_plan = SimpleNamespace(
        distribution_spec=SimpleNamespace(strategy=DistributionStrategy.RAY_TRAIN_TORCH)
    )
    assert not _trial_checkpoint_enabled(torch_plan)
    assert _trial_checkpoint_enabled(_resolved_plan())


def test_intermediate_report_uses_objective_and_actual_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded = []
    monkeypatch.setattr(TuneReportCallback, "__init__", lambda self: None)
    monkeypatch.setattr(
        TuneReportCallback,
        "after_report",
        lambda self, context, metrics, checkpoint: forwarded.append(metrics[0]),
    )
    callback = _IntermediateReportCallback("loss", "round")
    training = {"_tributo_xgboost_recovery": {"phase": "training"}}
    callback.after_report(None, [{"loss": 0.9, "round": 0, **training}], None)
    callback.after_report(None, [{"loss": 0.8, "round": 2, **training}], None)
    callback.after_report(None, [{"loss": 0.7, "round": 5, **training}], None)
    callback.after_report(None, [{"loss": 0.7, "round": 5, **training}], None)
    callback.after_report(None, [{"loss": 0.6, **training}], None)
    assert forwarded == [
        {"loss": 0.9, "round": 0},
        {"loss": 0.8, "round": 2},
        {"loss": 0.7, "round": 5},
    ]
    with pytest.raises(JobExecutionError, match="must be finite"):
        callback.after_report(
            None, [{"loss": 0.6, "round": float("nan"), **training}], None
        )


def test_xgboost_final_reports_are_not_forwarded_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded = []
    monkeypatch.setattr(TuneReportCallback, "__init__", lambda self: None)
    monkeypatch.setattr(
        TuneReportCallback,
        "after_report",
        lambda self, context, metrics, checkpoint: forwarded.append(metrics[0]),
    )
    callback = _IntermediateReportCallback("loss", "round")
    callback.after_report(None, [{"loss": 0.4, "round": 5}], None)
    callback.after_report(
        None,
        [{"loss": 0.3, "round": 6, "_tributo_xgboost_recovery": {"phase": "final"}}],
        None,
    )
    assert forwarded == []


def test_trainable_reports_final_metric_without_intermediate_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from ray import tune

    import tributo.algorithms

    class Dispatcher:
        def plan(self, *_args):
            return _resolved_plan()

        def execute_plan(self, *_args):
            assert _trial_reporter.get() is None
            return SimpleNamespace(execution=SimpleNamespace(metrics={"loss": 0.3}))

    monkeypatch.setattr(tributo.algorithms, "build_algorithm_dispatcher", Dispatcher)
    monkeypatch.setattr(
        tune,
        "get_context",
        lambda: SimpleNamespace(get_trial_dir=lambda: str(tmp_path)),
    )
    reports = []
    monkeypatch.setattr(
        tune, "report", lambda metrics, **kwargs: reports.append(metrics)
    )
    runner = object.__new__(PortableTuneRunner)
    runner._request = ExecutionRequest(
        algorithm_request=request_for("external_map_reduce", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    runner._input_context = InputExecutionContext({})
    runner._resolution_context = InputResolutionContext({})
    runner._tune_config = TuneSearchConfig(metric="loss", progress_attr="round")

    runner._trainable()({})
    assert reports == [{"loss": 0.3}]
    assert _trial_reporter.get() is None


def test_trainable_can_be_serialized_for_ray_tune() -> None:
    runner = object.__new__(PortableTuneRunner)
    runner._request = ExecutionRequest(
        algorithm_request=request_for("external_map_reduce", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    runner._input_context = InputExecutionContext({})
    runner._resolution_context = InputResolutionContext({})
    runner._tune_config = TuneSearchConfig(metric="loss", progress_attr="round")

    assert callable(cloudpickle.loads(cloudpickle.dumps(runner._trainable())))


@pytest.mark.parametrize(
    ("framework", "supports_reporting"),
    [("xgboost-rabit", True), ("other-framework", False)],
)
def test_trainable_only_attaches_reporter_for_supported_framework(
    monkeypatch: pytest.MonkeyPatch, tmp_path, framework: str, supports_reporting: bool
) -> None:
    from ray import tune

    import tributo.algorithms
    import tributo.training.portable_tune as portable_tune

    reporter = object()
    plan = SimpleNamespace(
        distribution_spec=SimpleNamespace(
            strategy=DistributionStrategy.FRAMEWORK_NATIVE,
            policy=FrameworkNativePolicy(
                framework=framework, evidence_collector_ref="tests.example:collect"
            ),
        )
    )

    class Dispatcher:
        def plan(self, *_args):
            return plan

        def execute_plan(self, *_args):
            assert _trial_reporter.get() is (reporter if supports_reporting else None)
            return SimpleNamespace(execution=SimpleNamespace(metrics={"loss": 0.3}))

    monkeypatch.setattr(tributo.algorithms, "build_algorithm_dispatcher", Dispatcher)
    monkeypatch.setattr(portable_tune, "_fit_only_plan", lambda value: value)
    monkeypatch.setattr(
        portable_tune, "_IntermediateReportCallback", lambda *_args: reporter
    )
    monkeypatch.setattr(
        tune,
        "get_context",
        lambda: SimpleNamespace(get_trial_dir=lambda: str(tmp_path)),
    )
    reports = []
    monkeypatch.setattr(
        tune, "report", lambda values, **_kwargs: reports.append(values)
    )
    runner = object.__new__(PortableTuneRunner)
    runner._request = ExecutionRequest(
        algorithm_request=request_for("external_map_reduce", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    runner._input_context = InputExecutionContext({})
    runner._resolution_context = InputResolutionContext({})
    runner._tune_config = TuneSearchConfig(metric="loss", progress_attr="round")

    runner._trainable()({})
    assert reports == [{"loss": 0.3}]
    assert _trial_reporter.get() is None


def test_portable_tune_requires_tunable_descriptor_and_cluster_profile() -> None:
    registration = map_reduce_registration()
    descriptor = registration.spec
    from tributo.algorithms.api import DistributedAlgorithmDescriptor

    distributed = DistributedAlgorithmDescriptor(
        package_name="tests-portable-tune",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        registration=replace(
            registration,
            spec=replace(
                descriptor,
                capabilities=(*descriptor.capabilities, Capability.TUNABLE),
            ),
        ),
    )
    request = ExecutionRequest(
        algorithm_request=request_for("external_map_reduce", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    runner = PortableTuneRunner(
        distributed,
        request,
        TuneSearchConfig(metric="loss", num_samples=1),
        SearchSpaceSpec(
            parameters=(
                SearchParamSpec(
                    path="learning_rate",
                    kind="uniform",
                    lower=0.01,
                    upper=0.1,
                ),
            )
        ),
        InputExecutionContext({}),
        InputResolutionContext({}),
    )
    assert runner is not None

    configured = PortableTuneRunner(
        distributed,
        request,
        TuneSearchConfig(
            metric="loss", scheduler="asha", progress_attr="round", scheduler_max_t=200
        ),
        runner._search_space,
        InputExecutionContext({}),
        InputResolutionContext({}),
    )
    assert configured._scheduler._time_attr == "round"
    assert configured._scheduler._max_t == 200
    assert configured._scheduler.set_search_properties("loss", "min")

    fifo = PortableTuneRunner(
        distributed,
        request,
        TuneSearchConfig(metric="loss", scheduler="fifo", progress_attr="round"),
        runner._search_space,
        InputExecutionContext({}),
        InputResolutionContext({}),
    )
    assert fifo._scheduler.set_search_properties("loss", "min")

    with pytest.raises(JobConfigurationError, match="do not support HyperBand"):
        PortableTuneRunner(
            distributed,
            request,
            TuneSearchConfig(
                metric="loss", scheduler="hyperband", progress_attr="round"
            ),
            runner._search_space,
            InputExecutionContext({}),
            InputResolutionContext({}),
        )

    legacy_hyperband = PortableTuneRunner(
        distributed,
        request,
        TuneSearchConfig(metric="loss", scheduler="hyperband"),
        runner._search_space,
        InputExecutionContext({}),
        InputResolutionContext({}),
    )
    assert legacy_hyperband._scheduler._time_attr == "training_iteration"
    assert legacy_hyperband._scheduler.set_search_properties("loss", "min")
