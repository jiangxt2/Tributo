"""Ray Tune hyperparameter tuning runner.

Receives the pre-validated ``effective_config`` and a
``SearchSpaceSpec`` IR.  Per-trial, applies dot-path overrides from
sampled values and re-validates before constructing the trainer.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import math
import re
from collections.abc import Mapping
from numbers import Real
from typing import Any, Callable

import ray
import ray.data
from ray.exceptions import RayTaskError, TaskCancelledError
from ray.tune import Checkpoint as TuneCheckpoint
from ray.tune import ResultGrid, RunConfig, Tuner
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler, HyperBandScheduler

from tributo._common.immutable import deep_thaw
from tributo.exceptions import JobConfigurationError, JobExecutionError, TributoError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    require_legacy_trainer_cls,
)
from tributo.training.config import (
    apply_dot_overrides,
    validate_and_normalize_config,
    validate_execution_config,
)
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_space import SearchSpaceSpec, to_ray_param_space
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

_MISSING_METRIC = object()
_SENSITIVE_PARAM_TOKENS = (
    "password",
    "secret",
    "token",
    "credential",
    "apikey",
    "accesskey",
    "privatekey",
    "signature",
)
_SAFE_TRIAL_ID = re.compile(r"^[A-Za-z0-9_.:+-]{1,64}$")
_REMOTE_STORAGE_PATH = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_RAY_TRIAL_STORAGE_NAMESPACE = "trials"

# Scheduler mapping (lazy instantiation to avoid unnecessary imports)
_SCHEDULER_MAP: dict[str, Any] = {
    "fifo": lambda metric, mode: FIFOScheduler(),
    "asha": lambda metric, mode: ASHAScheduler(metric=metric, mode=mode),
    "hyperband": lambda metric, mode: HyperBandScheduler(metric=metric, mode=mode),
}

# Search algorithm mapping (lazy instantiation)
_SEARCH_ALG_MAP: dict[str, Any] = {
    # ``None`` lets Ray create its default BasicVariantGenerator with the
    # configured max_concurrent_trials value.
    "random": None,
}


def _get_bayesopt_search(metric: str, mode: str) -> Any:
    """Get a BayesOpt search algorithm instance (optional dependency)."""
    try:
        from ray.tune.search.bayesopt import BayesOptSearch

        return BayesOptSearch(metric=metric, mode=mode)
    except ImportError as err:
        raise ImportError(
            "bayesian-optimization is required for BayesOpt search algorithm. "
            "Install with: pip install bayesian-optimization"
        ) from err


_SEARCH_ALG_MAP["bayesopt"] = _get_bayesopt_search


def _metric_value_type(value: object) -> str:
    """Return a safe type description without rendering a metric value."""
    if value is _MISSING_METRIC:
        return "missing"
    return type(value).__name__


def _normalize_target_metric(
    metric: str,
    value: object,
    *,
    location: str,
) -> float:
    """Convert one target metric candidate to a finite Python float."""
    value_type = _metric_value_type(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise JobExecutionError(
            f"Ray Tune target metric {metric!r} at {location} must be a finite "
            f"real scalar; received value type {value_type!r}."
        )
    try:
        normalized = float(value)
    except Exception as exc:
        raise JobExecutionError(
            f"Ray Tune target metric {metric!r} at {location} could not be "
            f"converted to float; received value type {value_type!r}."
        ) from exc
    if not math.isfinite(normalized):
        raise JobExecutionError(
            f"Ray Tune target metric {metric!r} at {location} must be finite; "
            f"received value type {value_type!r}."
        )
    return normalized


def _extract_target_metric(result: object, metric: str) -> float:
    """Extract and validate the configured target metric from a result mapping.

    The Tune boundary supports direct metric mappings and the nested
    ``result["metrics"]`` compatibility shape. A metric present in both
    locations is accepted only when both values normalize to the same finite
    Python float.
    """
    if not isinstance(result, Mapping):
        raise JobExecutionError(
            f"Ray Tune target metric {metric!r} is missing because the trial "
            f"result has value type {type(result).__name__!r}, not a mapping."
        )

    top_value = result.get(metric, _MISSING_METRIC)
    nested_metrics = result.get("metrics")
    # A top-level ``metrics`` mapping is the compatibility container, not a
    # scalar candidate when the configured metric itself is named ``metrics``.
    if metric == "metrics" and isinstance(top_value, Mapping):
        top_value = _MISSING_METRIC
    nested_value = (
        nested_metrics.get(metric, _MISSING_METRIC)
        if isinstance(nested_metrics, Mapping)
        else _MISSING_METRIC
    )

    if top_value is _MISSING_METRIC and nested_value is _MISSING_METRIC:
        raise JobExecutionError(
            f"Ray Tune target metric {metric!r} is missing from the supported "
            "trial result locations (top level and nested 'metrics' mapping); "
            "received value type 'missing'."
        )

    top_metric = (
        _normalize_target_metric(metric, top_value, location="result top level")
        if top_value is not _MISSING_METRIC
        else None
    )
    nested_metric = (
        _normalize_target_metric(
            metric,
            nested_value,
            location="nested 'metrics' mapping",
        )
        if nested_value is not _MISSING_METRIC
        else None
    )

    if top_metric is not None and nested_metric is not None:
        if top_metric != nested_metric:
            raise JobExecutionError(
                f"Ray Tune target metric {metric!r} is ambiguous: the result "
                "top level and nested 'metrics' mapping contain different finite values "
                f"with types {_metric_value_type(top_value)!r} and "
                f"{_metric_value_type(nested_value)!r}."
            )
        return top_metric
    if top_metric is not None:
        return top_metric
    assert nested_metric is not None
    return nested_metric


def _is_sensitive_param_name(name: str) -> bool:
    compact = "".join(character for character in name.lower() if character.isalnum())
    return any(token in compact for token in _SENSITIVE_PARAM_TOKENS)


def _safe_sampled_values(sampled_values: Mapping[str, object]) -> dict[str, object]:
    """Build bounded, credential-safe trial context for diagnostic logs."""
    safe_values: dict[str, object] = {}
    for raw_key, value in sampled_values.items():
        key = raw_key[:128]
        if _is_sensitive_param_name(raw_key):
            safe_values[key] = "<redacted>"
        elif value is None or isinstance(value, bool):
            safe_values[key] = value
        elif isinstance(value, Real):
            try:
                safe_values[key] = float(value)
            except Exception:
                safe_values[key] = f"<{type(value).__name__}>"
        else:
            safe_values[key] = f"<{type(value).__name__}>"
    return safe_values


def _trial_id(ray_tune: Any) -> str:
    """Read a safe public Ray Tune trial ID without masking a trial failure."""
    try:
        trial_id = ray_tune.get_context().get_trial_id()
        if isinstance(trial_id, str) and _SAFE_TRIAL_ID.fullmatch(trial_id):
            return trial_id
    except Exception:
        pass
    return "unknown"


def _safe_storage_component(value: str, *, fallback: str) -> str:
    """Return a bounded path component without exposing the original value."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    prefix = normalized[:40] or fallback
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _join_storage_path(root: str, *components: str) -> str:
    """Join local or URI storage paths without changing their scheme."""
    suffix = "/".join(component.strip("/") for component in components)
    if not root:
        return suffix
    return f"{root.rstrip('/')}/{suffix}"


def _build_trial_run_config(
    ray_tune: Any,
    *,
    output_path: str,
    experiment_name: str,
    algorithm_name: str,
) -> dict[str, str]:
    """Derive stable, isolated inner Ray Train storage from Tune context."""
    try:
        context = ray_tune.get_context()
        trial_id = context.get_trial_id()
        trial_name = context.get_trial_name()
        trial_dir = context.get_trial_dir()
    except Exception as exc:
        raise JobExecutionError(
            "Ray Tune trial context is unavailable for inner Ray Train isolation."
        ) from exc

    if not isinstance(trial_id, str) or not _SAFE_TRIAL_ID.fullmatch(trial_id):
        raise JobExecutionError(
            "Ray Tune trial context returned an invalid trial ID for inner "
            "Ray Train isolation."
        )
    if not isinstance(trial_name, str) or not trial_name:
        raise JobExecutionError(
            "Ray Tune trial context returned an invalid trial name for inner "
            "Ray Train isolation."
        )
    if not isinstance(trial_dir, str) or not trial_dir:
        raise JobExecutionError(
            "Ray Tune trial context returned an invalid trial directory for inner "
            "Ray Train isolation."
        )

    algorithm_component = _safe_storage_component(
        algorithm_name,
        fallback="algorithm",
    )
    trial_name_digest = hashlib.sha256(trial_name.encode("utf-8")).hexdigest()[:10]
    run_name = f"{algorithm_component}-{trial_id}-{trial_name_digest}"

    if _REMOTE_STORAGE_PATH.match(output_path) and not output_path.startswith(
        "file://"
    ):
        experiment_component = _safe_storage_component(
            experiment_name,
            fallback="experiment",
        )
        storage_path = _join_storage_path(
            output_path,
            _RAY_TRIAL_STORAGE_NAMESPACE,
            experiment_component,
            "_tributo_ray_train",
            trial_id,
        )
    else:
        storage_path = _join_storage_path(trial_dir, "_tributo_ray_train")

    return {"name": run_name, "storage_path": storage_path}


def _fit_result_metric_mapping(fit_result: object) -> Mapping[str, object]:
    """Expose only fit metrics to the target-metric validation boundary."""
    if isinstance(fit_result, Mapping):
        return fit_result
    dynamic_result: Any = fit_result
    try:
        metrics = dynamic_result.metrics
    except Exception as exc:
        raise JobExecutionError(
            "Ray Tune trial fit result does not expose a readable metrics mapping."
        ) from exc
    if not isinstance(metrics, Mapping):
        raise JobExecutionError(
            "Ray Tune trial fit result metrics must be a mapping; received value "
            f"type {type(metrics).__name__!r}."
        )
    return metrics


def _fit_result_checkpoint(fit_result: object) -> TuneCheckpoint | None:
    """Adapt a Ray Train checkpoint for Tune without copying its contents."""
    if isinstance(fit_result, Mapping):
        return None
    dynamic_result: Any = fit_result
    try:
        checkpoint = dynamic_result.checkpoint
    except AttributeError:
        return None
    except Exception as exc:
        raise JobExecutionError(
            "Ray Tune trial fit result checkpoint could not be read."
        ) from exc
    if checkpoint is None:
        return None
    if isinstance(checkpoint, TuneCheckpoint):
        return checkpoint

    from ray.train import Checkpoint as TrainCheckpoint

    if not isinstance(checkpoint, TrainCheckpoint):
        raise JobExecutionError(
            "Ray Tune trial fit result checkpoint must be a Ray Checkpoint; "
            f"received value type {type(checkpoint).__name__!r}."
        )
    return TuneCheckpoint(
        path=checkpoint.path,
        filesystem=checkpoint.filesystem,
    )


def _is_trial_cancellation(error: BaseException) -> bool:
    """Recognize explicit local and Ray cancellation chains."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    cancellation_types = (
        asyncio.CancelledError,
        concurrent.futures.CancelledError,
        TaskCancelledError,
    )
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, cancellation_types):
            return True
        candidates: list[object] = [current.__cause__]
        if isinstance(current, RayTaskError):
            candidates.append(current.cause)
        pending.extend(
            candidate
            for candidate in candidates
            if isinstance(candidate, BaseException)
        )
    return False


def _log_trial_failure(
    ray_tune: Any,
    sampled_values: Mapping[str, object],
    *,
    stage: str,
    error: BaseException,
) -> None:
    """Log safe trial context while leaving the original exception untouched."""
    logger.error(
        "Ray Tune trial failed: trial_id=%s stage=%s error_type=%s sampled_values=%s",
        _trial_id(ray_tune),
        stage,
        type(error).__name__,
        _safe_sampled_values(sampled_values),
    )


@PublicAPI(stability="beta")
class TuneRunner:
    """Ray Tune hyperparameter tuning runner.

    Adapts Tributo's TrainerSpec to Ray Tune's trainable,
    supporting JSON search space configuration and multiple search algorithms/schedulers.

    Example:
        >>> from tributo.training.registry import get_trainer
        >>> from tributo.training.tune_config import TuneSearchConfig
        >>> from tributo.training.tune_space import parse_search_space
        >>> from tributo.training.tune_runner import TuneRunner
        >>>
        >>> trainer_spec = get_trainer("xgboost")
        >>> search_space = parse_search_space("space.json")
        >>> tune_config = TuneSearchConfig(num_samples=10, search_alg="random")
        >>> runner = TuneRunner(
        ...     trainer_spec, tune_config, search_space, effective_config
        ... )
        >>> result_grid = runner.run(datasets={"train": ds}, output_path="/tmp/out")
    """

    def __init__(
        self,
        trainer_spec: AlgorithmSpec,
        tune_config: TuneSearchConfig,
        search_space: SearchSpaceSpec,
        effective_config: dict[str, Any],
    ):
        """Initialize TuneRunner.

        Args:
            trainer_spec: Algorithm specification.
            tune_config: Search configuration.
            search_space: Search-space IR (from ``parse_search_space``).
            effective_config: Driver-merged & validated config dict.
        """
        self._trainer_spec = trainer_spec
        self._tune_config = tune_config
        self._search_space = search_space
        self._effective_config = deep_thaw(effective_config)
        self._ray_param_space = to_ray_param_space(search_space)

        # Instantiate search algorithm and scheduler early to detect missing optional dependencies
        # (e.g., uninstalled BayesOpt will throw ImportError immediately)
        try:
            search_alg_factory = _SEARCH_ALG_MAP[tune_config.search_alg]
            scheduler_factory = _SCHEDULER_MAP[tune_config.scheduler]
        except KeyError as exc:
            raise JobConfigurationError(
                f"Unsupported search_alg='{tune_config.search_alg}' "
                f"or scheduler='{tune_config.scheduler}'"
            ) from exc
        self._search_alg = (
            search_alg_factory(
                tune_config.metric,
                tune_config.mode,
            )
            if search_alg_factory is not None
            else None
        )
        self._scheduler = scheduler_factory(
            tune_config.metric,
            tune_config.mode,
        )

    def _build_trainable(
        self,
        datasets: dict[str, ray.data.Dataset],
        output_path: str,
        experiment_name: str = "tributo-tune",
    ) -> Callable[[dict[str, Any]], None]:
        """Convert TrainerSpec to a Ray Tune trainable function.

        Uses a closure to capture datasets and output_path, avoiding injection
        of non-hyperparameter data into param_space (Ray Tune tries to sample all keys in param_space).

        Args:
            datasets: Dataset dictionary (bound via closure, not entered into param_space).
            output_path: Output path (bound via closure, not entered into param_space).
            experiment_name: Tune experiment namespace used for remote trial storage.

        Returns:
            Ray Tune trainable function.
        """
        spec = self._trainer_spec
        effective_base = self._effective_config
        trainer_cls = require_legacy_trainer_cls(
            spec,
            consumer="TuneRunner",
        )

        def trainable(sampled_values: dict[str, Any]) -> None:
            from ray import tune as ray_tune

            stage = "configuration"
            try:
                trial_config = apply_dot_overrides(effective_base, sampled_values)
                trial_config = validate_and_normalize_config(spec, trial_config)
                validate_execution_config(
                    spec, trial_config, datasets_supplied="train" in datasets
                )
                stage = "trial isolation"
                trial_run_config = _build_trial_run_config(
                    ray_tune,
                    output_path=output_path,
                    experiment_name=experiment_name,
                    algorithm_name=spec.name,
                )
                stage = "trainer construction"
                trainer = trainer_cls(
                    datasets=datasets,
                    config=trial_config,
                    run_config=trial_run_config,
                )
                stage = "setup"
                trainer.setup()
                stage = "fit"
                fit_result = trainer.training_loop()
                stage = "fit result validation"
                metric_mapping = _fit_result_metric_mapping(fit_result)
                checkpoint = _fit_result_checkpoint(fit_result)
                stage = "target metric validation"
                target_metric = _extract_target_metric(
                    metric_mapping,
                    self._tune_config.metric,
                )
                stage = "target metric reporting"
                metrics = {self._tune_config.metric: target_metric}
                if checkpoint is None:
                    ray_tune.report(metrics)
                else:
                    ray_tune.report(metrics, checkpoint=checkpoint)
            except (
                asyncio.CancelledError,
                concurrent.futures.CancelledError,
                TaskCancelledError,
            ) as exc:
                _log_trial_failure(
                    ray_tune,
                    sampled_values,
                    stage=stage,
                    error=exc,
                )
                raise
            except TributoError as exc:
                _log_trial_failure(
                    ray_tune,
                    sampled_values,
                    stage=stage,
                    error=exc,
                )
                raise
            except Exception as exc:
                if _is_trial_cancellation(exc):
                    _log_trial_failure(
                        ray_tune,
                        sampled_values,
                        stage=stage,
                        error=exc,
                    )
                    raise
                _log_trial_failure(
                    ray_tune,
                    sampled_values,
                    stage=stage,
                    error=exc,
                )
                raise JobExecutionError(
                    f"Ray Tune trial {stage} failed with {type(exc).__name__}."
                ) from exc

        return trainable

    def _build_tune_config(self) -> Any:
        """Build the Ray Tune TuneConfig object.

        Returns:
            ray.tune.TuneConfig instance.
        """
        from ray.tune import TuneConfig

        return TuneConfig(
            metric=self._tune_config.metric,
            mode=self._tune_config.mode,
            num_samples=self._tune_config.num_samples,
            max_concurrent_trials=self._tune_config.max_concurrent_trials,
            time_budget_s=self._tune_config.time_budget_s,
            search_alg=self._search_alg,
            scheduler=self._scheduler,
        )

    def _build_failure_config(self) -> Any:  # type: ignore[no-untyped-def]
        """Build the failure configuration.

        Returns:
            ray.tune.FailureConfig instance.
        """
        from ray.tune import FailureConfig

        return FailureConfig(
            fail_fast=self._tune_config.fail_fast,
        )

    def run(
        self,
        datasets: dict[str, ray.data.Dataset],
        output_path: str,
        experiment_name: str = "tributo-tune",
        runtime_env: dict[str, Any] | None = None,
    ) -> ResultGrid:
        """Execute hyperparameter search.

        Args:
            datasets: Dataset dictionary (e.g. {"train": ds}).
            output_path: Output path (local or S3).
            experiment_name: Experiment name.
            runtime_env: Optional Ray runtime environment for Tune trial workers.
                In Ray Tune v1, this is set at the Ray session level and
                inherited by all trial actors. If Ray has not been initialized
                when this method is called, ``ray.init(runtime_env=runtime_env)``
                is called automatically. If Ray is already initialized, a
                warning is logged — set ``runtime_env`` at ``ray.init()`` before
                creating the ``TuneRunner`` instead.
                Example: ``{"pip": ["xgboost"]}`` to install optional extras.

        Returns:
            Ray Tune ResultGrid containing all trial results.
        """
        # Guard: reject algorithms that don't declare Capability.TUNABLE.
        if Capability.TUNABLE not in self._trainer_spec.capabilities:
            raise JobConfigurationError(
                f"Algorithm {self._trainer_spec.name!r} does not declare "
                f"Capability.TUNABLE and cannot be used with TuneRunner."
            )

        # Set runtime_env at Ray session level (only before ray.init)
        if runtime_env is not None:
            if not ray.is_initialized():
                ray.init(runtime_env=runtime_env)
            else:
                logger.warning(
                    "runtime_env=%s provided but Ray is already initialized. "
                    "Set runtime_env at ray.init() before creating TuneRunner.",
                    runtime_env,
                )

        # Capture datasets and output_path via closure, not injected into param_space
        trainable = self._build_trainable(
            datasets,
            output_path,
            experiment_name=experiment_name,
        )

        tuner = Tuner(
            trainable=trainable,
            param_space=self._ray_param_space,
            tune_config=self._build_tune_config(),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=_join_storage_path(
                    output_path,
                    _RAY_TRIAL_STORAGE_NAMESPACE,
                ),
                failure_config=self._build_failure_config(),
            ),
        )

        logger.info(
            "Starting Tune experiment '%s': %d samples, %s algorithm, %s scheduler",
            experiment_name,
            self._tune_config.num_samples,
            self._tune_config.search_alg,
            self._tune_config.scheduler,
        )

        result_grid = tuner.fit()

        # Output result summary
        try:
            best_result = result_grid.get_best_result()
        except RuntimeError as exc:
            raise JobExecutionError(
                f"Tune experiment failed to find best result for metric="
                f"'{self._tune_config.metric}', mode='{self._tune_config.mode}'. "
                "This usually means all trials failed or no trial reported the metric."
            ) from exc
        metric_value = (
            best_result.metrics.get(self._tune_config.metric, float("nan"))
            if best_result.metrics
            else float("nan")
        )
        logger.info(
            "Best trial: metric=%s=%.4f",
            self._tune_config.metric,
            metric_value,
        )

        # Persist experiment summary
        self._save_summary(result_grid, output_path, experiment_name)

        return result_grid

    def _save_summary(
        self,
        result_grid: ResultGrid,
        output_path: str,
        experiment_name: str,
    ) -> None:
        """Save experiment summary to output path.

        Args:
            result_grid: Tune experiment results.
            output_path: Output path (local directory or S3 URI).
            experiment_name: Experiment name.
        """
        try:
            best_params = extract_best_params(
                result_grid,
                metric=self._tune_config.metric,
                mode=self._tune_config.mode,
            )
            summary = {
                "experiment_name": experiment_name,
                "metric": self._tune_config.metric,
                "mode": self._tune_config.mode,
                "num_samples": self._tune_config.num_samples,
                "search_alg": self._tune_config.search_alg,
                "scheduler": self._tune_config.scheduler,
                "best_params": best_params,
                "num_trials": len(result_grid),
                "num_errors": result_grid.num_errors,
            }

            from tributo._common.storage import write_json

            summary_path = output_path.rstrip("/") + "/tune_summary.json"
            write_json(summary_path, summary)
            logger.info("Tune summary saved to %s", summary_path)
        except Exception:
            logger.exception("Failed to save tune summary, continuing")


@PublicAPI(stability="beta")
def extract_best_params(
    result_grid: ResultGrid,
    metric: str = "loss",
    mode: str = "min",
) -> dict[str, Any]:
    """Extract the best trial's hyperparameters from a ResultGrid.

    Args:
        result_grid: Return value of TuneRunner.run().
        metric: Optimization metric name.
        mode: Optimization direction (min/max).

    Returns:
        Hyperparameter dictionary of the best trial.

    Raises:
        JobExecutionError: When the best trial has no available configuration.
    """
    try:
        best_result = result_grid.get_best_result(metric=metric, mode=mode)
    except RuntimeError as exc:
        raise JobExecutionError(
            f"No best result found for metric='{metric}', mode='{mode}'. "
            "This usually means all trials failed or no trial reported the metric."
        ) from exc
    if best_result.config is None:
        raise JobExecutionError(
            f"No valid config found in best result for metric={metric}, mode={mode}"
        )
    return best_result.config
