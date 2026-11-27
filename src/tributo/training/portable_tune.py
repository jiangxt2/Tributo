"""Ray Tune runner for portable distributed algorithm descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any

import ray
from ray.tune import ResultGrid, RunConfig, Tuner
from ray.tune.integration.ray_train import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from tributo._common.immutable import deep_thaw
from tributo.algorithms.api import (
    DistributedAlgorithmDescriptor,
    DistributionStrategy,
    ExecutionRequest,
    FrameworkNativePolicy,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    canonical_digest,
)
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.exceptions import JobConfigurationError, JobExecutionError
from tributo.training.algorithm_spec import Capability
from tributo.training.config import apply_dot_overrides
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_runner import (
    _DISTRIBUTION_SEARCH_PATHS,
    _SCHEDULER_MAP,
    _SEARCH_ALG_MAP,
    _extract_target_metric,
    _join_storage_path,
)
from tributo.training.tune_space import SearchSpaceSpec, to_ray_param_space
from tributo.util.annotations import PublicAPI

_trial_reporter: ContextVar[TuneReportCallback | None] = ContextVar(
    "portable_tune_reporter", default=None
)
_XGBOOST_REPORT_KEY = "_tributo_xgboost_recovery"


class _IntermediateReportCallback(TuneReportCallback):
    """Forward XGBoost training reports without forwarding its final report."""

    def __init__(self, metric: str, progress_attr: str) -> None:
        super().__init__()
        self._metric = metric
        self._progress_attr = progress_attr
        self._last_progress: float | None = None

    def after_report(
        self, run_context: Any, metrics: list[dict[str, Any]], checkpoint: Any
    ) -> None:
        values = metrics[0]
        phase = values.get(_XGBOOST_REPORT_KEY)
        if not isinstance(phase, Mapping) or phase.get("phase") != "training":
            return
        if self._progress_attr not in values:
            raise JobExecutionError(
                f"XGBoost training report is missing progress attribute "
                f"{self._progress_attr!r}"
            )
        if self._metric not in values:
            return
        progress = _extract_target_metric(values, self._progress_attr)
        if self._last_progress is not None and progress <= self._last_progress:
            return
        objective = _extract_target_metric(values, self._metric)
        self._last_progress = progress
        super().after_report(
            run_context,
            [{self._metric: objective, self._progress_attr: progress}],
            None,
        )


def _fit_only_plan(plan: ResolvedAlgorithmPlan) -> ResolvedAlgorithmPlan:
    """Derive an internal trial plan without publishing a formal Bundle."""
    distribution = plan.distribution_spec
    if distribution is None:
        raise JobConfigurationError("portable Tune requires a DistributionSpec")
    fit_only_distribution = replace(
        distribution,
        result_policy=ResultPolicy.FIT_ONLY,
    )
    runtime = replace(
        plan.runtime,
        distribution_digest=fit_only_distribution.digest,
        resume_from=None,
    )
    provisional = replace(
        plan,
        plan_id="0" * 64,
        runtime=runtime,
        distribution_spec=fit_only_distribution,
        contract_bindings=None,
    )
    return replace(
        provisional,
        plan_id=canonical_digest(provisional.to_dict(include_plan_id=False)),
    )


def _trial_request(
    base: ExecutionRequest,
    sampled_values: dict[str, Any],
    *,
    checkpoint_dir: str,
) -> ExecutionRequest:
    algorithm_request = base.algorithm_request
    config = apply_dot_overrides(
        deep_thaw(algorithm_request.algorithm_config),
        sampled_values,
    )
    runtime = config.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, dict):
            raise JobConfigurationError(
                "portable Tune runtime config must be a mapping"
            )
        runtime["checkpoint_dir"] = checkpoint_dir
    return replace(
        base,
        algorithm_request=replace(
            algorithm_request,
            algorithm_config=config,
        ),
        resume_from=None,
    )


def _trial_checkpoint_enabled(plan: ResolvedAlgorithmPlan) -> bool:
    """Return whether the existing portable checkpoint files apply to this plan."""
    return not (
        plan.distribution_spec is not None
        and plan.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )


@PublicAPI(stability="alpha")
class PortableTuneRunner:
    """Tune one Wheel algorithm through its normal distributed Core Runtime."""

    def __init__(
        self,
        descriptor: DistributedAlgorithmDescriptor,
        request: ExecutionRequest,
        tune_config: TuneSearchConfig,
        search_space: SearchSpaceSpec,
        input_context: InputExecutionContext,
        resolution_context: InputResolutionContext,
    ) -> None:
        if descriptor.name != request.algorithm_request.algorithm:
            raise JobConfigurationError(
                "portable Tune descriptor and request algorithm identities differ"
            )
        if Capability.TUNABLE not in descriptor.registration.spec.capabilities:
            raise JobConfigurationError(
                f"Algorithm {descriptor.name!r} does not declare Capability.TUNABLE"
            )
        if request.profile.value != "cluster":
            raise JobConfigurationError(
                "portable Tune trials must attach to an initialized Ray cluster"
            )
        if forbidden := sorted(
            parameter.path
            for parameter in search_space.parameters
            if parameter.path in _DISTRIBUTION_SEARCH_PATHS
        ):
            raise JobConfigurationError(
                f"distributed topology is not a Tune hyperparameter: {forbidden}"
            )
        distribution = descriptor.registration.distribution_spec
        if distribution is None:
            raise JobConfigurationError("portable Tune descriptor is not distributed")
        if not distribution.supported_worker_range.contains(request.worker_count):
            raise JobConfigurationError(
                "portable Tune worker count is outside the descriptor range"
            )
        self._descriptor = descriptor
        self._request = request
        self._tune_config = tune_config
        self._search_space = search_space
        self._input_context = input_context
        self._resolution_context = resolution_context
        try:
            search_factory = _SEARCH_ALG_MAP[tune_config.search_alg]
            scheduler_factory = _SCHEDULER_MAP[tune_config.scheduler]
        except KeyError as exc:
            raise JobConfigurationError("unsupported portable Tune strategy") from exc
        self._search_alg = (
            search_factory(tune_config.metric, tune_config.mode)
            if search_factory is not None
            else None
        )
        if tune_config.progress_attr and tune_config.scheduler == "hyperband":
            raise JobConfigurationError(
                "portable Tune intermediate metrics do not support HyperBand "
                "pause and resume; use ASHA or FIFO"
            )
        if (
            tune_config.scheduler == "asha"
            and tune_config.progress_attr
            and tune_config.scheduler_max_t is None
        ):
            raise JobConfigurationError(
                "portable Tune ASHA requires scheduler_max_t with progress_attr"
            )
        if tune_config.scheduler == "asha" and (
            tune_config.progress_attr or tune_config.scheduler_max_t is not None
        ):
            options: dict[str, Any] = {}
            if tune_config.progress_attr:
                options["time_attr"] = tune_config.progress_attr
            if tune_config.scheduler_max_t is not None:
                options["max_t"] = tune_config.scheduler_max_t
            self._scheduler = ASHAScheduler(**options)
        else:
            self._scheduler = scheduler_factory(None, None)

    def _trainable(self) -> Any:
        request = self._request
        input_context = self._input_context
        resolution_context = self._resolution_context
        metric_name = self._tune_config.metric
        progress_attr = self._tune_config.progress_attr

        def trainable(sampled_values: dict[str, Any]) -> None:
            from ray import tune as ray_tune
            from ray.train import Checkpoint

            from tributo.algorithms import build_algorithm_dispatcher
            from tributo.training.portable_tune import _trial_reporter

            trial_dir = Path(ray_tune.get_context().get_trial_dir())
            checkpoint_dir = trial_dir / "algorithm-checkpoint"
            trial_request = _trial_request(
                request,
                sampled_values,
                checkpoint_dir=str(checkpoint_dir),
            )
            dispatcher = build_algorithm_dispatcher()
            plan = dispatcher.plan(trial_request, resolution_context)
            reporter = (
                _IntermediateReportCallback(metric_name, progress_attr)
                if progress_attr
                and plan.distribution_spec is not None
                and plan.distribution_spec.strategy
                is DistributionStrategy.FRAMEWORK_NATIVE
                and isinstance(plan.distribution_spec.policy, FrameworkNativePolicy)
                and plan.distribution_spec.policy.framework == "xgboost-rabit"
                else None
            )
            token = _trial_reporter.set(reporter)
            try:
                result = dispatcher.execute_plan(_fit_only_plan(plan), input_context)
            finally:
                _trial_reporter.reset(token)
            metric = _extract_target_metric(result.execution.metrics, metric_name)
            metrics = {metric_name: metric}
            if progress_attr and progress_attr in result.execution.metrics:
                metrics[progress_attr] = _extract_target_metric(
                    result.execution.metrics, progress_attr
                )
            if not _trial_checkpoint_enabled(plan):
                ray_tune.report(metrics)
                return
            manifest = checkpoint_dir / "manifest.json"
            state = checkpoint_dir / "state.bin"
            if manifest.is_file() and state.is_file():
                ray_tune.report(
                    metrics,
                    checkpoint=Checkpoint.from_directory(checkpoint_dir),
                )
            else:
                ray_tune.report(metrics)

        return trainable

    def run(
        self,
        *,
        output_path: str,
        experiment_name: str = "tributo-portable-tune",
    ) -> ResultGrid:
        """Run fit-only trials and return Ray Tune's immutable ResultGrid."""
        if not ray.is_initialized():
            raise JobExecutionError(
                "portable Tune requires an initialized Ray cluster connection"
            )
        from ray.tune import FailureConfig, TuneConfig
        from ray.tune.execution.placement_groups import PlacementGroupFactory

        distribution = self._descriptor.registration.distribution_spec
        assert distribution is not None
        resources = distribution.resources_per_worker
        worker_bundle = {
            "CPU": resources.num_cpus,
            "GPU": resources.num_gpus,
            **dict(resources.custom),
        }
        placement = PlacementGroupFactory(
            [
                {"CPU": 1},
                *[worker_bundle for _ in range(self._request.worker_count)],
            ],
            strategy="SPREAD",
        )
        from ray import tune as ray_tune

        trainable = ray_tune.with_resources(self._trainable(), placement)
        tuner = Tuner(
            trainable=trainable,
            param_space=to_ray_param_space(self._search_space),
            tune_config=TuneConfig(
                metric=self._tune_config.metric,
                mode=self._tune_config.mode,
                num_samples=self._tune_config.num_samples,
                max_concurrent_trials=self._tune_config.max_concurrent_trials,
                time_budget_s=self._tune_config.time_budget_s,
                search_alg=self._search_alg,
                scheduler=self._scheduler,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=_join_storage_path(output_path, "trials"),
                failure_config=FailureConfig(
                    fail_fast=self._tune_config.fail_fast,
                ),
            ),
        )
        return tuner.fit()


__all__ = ["PortableTuneRunner"]
