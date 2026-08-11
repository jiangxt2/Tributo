"""Worker-only adapter for the execution part of a legacy BaseTrainer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
)
from tributo.training.base import BaseTrainer
from tributo.util.annotations import DeveloperAPI


def _portable_value(value: object, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise AlgorithmExecutionError(f"legacy metric {path!r} is not finite")
        return number
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise AlgorithmExecutionError(
                    f"legacy result {path!r} contains a non-string key"
                )
            normalized[key] = _portable_value(nested, path=f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _portable_value(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _portable_value(item(), path=path)
        except (TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                f"legacy result {path!r} cannot be converted to a portable value"
            ) from exc
    raise AlgorithmExecutionError(
        f"legacy result {path!r} has unsupported type {type(value).__name__!r}"
    )


def _extract_metrics(result: object) -> dict[str, Any]:
    if isinstance(result, Mapping):
        candidate = result.get("metrics", result)
    else:
        candidate = getattr(result, "metrics", {})
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise AlgorithmExecutionError("legacy Trainer returned non-mapping metrics")
    normalized = _portable_value(candidate, path="metrics")
    if not isinstance(normalized, dict):
        raise AlgorithmExecutionError("legacy Trainer metrics were not portable")
    return normalized


def _legacy_datasets(inputs: Mapping[str, object]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, MaterializedTabularInputView):
            try:
                import ray.data

                columns = value.columns()
                rows = [
                    dict(zip(columns, values, strict=True))
                    for values in zip(*columns.values(), strict=True)
                ]
                datasets[name] = ray.data.from_items(rows)
            except AlgorithmInputError:
                raise
            except Exception as exc:
                raise AlgorithmInputError(
                    "legacy Trainer input could not be converted to Ray Data"
                ) from exc
        else:
            datasets[name] = value
    return datasets


@DeveloperAPI
class LegacyTrainerExecutable:
    """Run setup and training_loop without invoking legacy artifact delivery."""

    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        trainer_cls: type[BaseTrainer],
        artifacts: tuple[ArtifactDraft, ...],
    ) -> None:
        if artifacts:
            raise AlgorithmConfigurationError(
                "the legacy Trainer adapter does not accept input artifacts"
            )
        self._plan = plan
        self._trainer_cls = trainer_cls

    def fit(self, context: AlgorithmExecutionContext) -> AlgorithmExecutionResult:
        """Execute only the bounded training portion of the legacy lifecycle."""
        if context.cancelled:
            raise AlgorithmExecutionError(
                "legacy Trainer execution was cancelled before construction"
            )
        try:
            trainer = self._trainer_cls(
                datasets=_legacy_datasets(context.inputs),
                config=dict(self._plan.algorithm_config),
            )
        except (
            AlgorithmConfigurationError,
            AlgorithmExecutionError,
            AlgorithmInputError,
        ):
            raise
        except Exception as exc:
            if exc.__class__.__module__.startswith("pydantic"):
                raise AlgorithmConfigurationError(
                    "legacy Trainer configuration validation failed"
                ) from exc
            raise AlgorithmExecutionError(
                f"legacy Trainer construction failed: {type(exc).__name__}"
            ) from exc
        try:
            trainer.setup()
            result = trainer.training_loop()
            metrics = _extract_metrics(result)
        except (
            AlgorithmConfigurationError,
            AlgorithmExecutionError,
            AlgorithmInputError,
        ):
            raise
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"legacy Trainer fit failed: {type(exc).__name__}"
            ) from exc
        checkpoint = getattr(result, "checkpoint", None)
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=metrics,
            outputs={
                "adapter": "legacy_trainer",
                "checkpoint_available": checkpoint is not None,
                "delivery_performed": False,
            },
        )


@DeveloperAPI
def create_executable(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[ArtifactDraft, ...],
) -> LegacyTrainerExecutable:
    """Validate a Worker-loaded Trainer class and create its adapter."""
    if not isinstance(implementation, type) or not issubclass(
        implementation, BaseTrainer
    ):
        raise AlgorithmExecutionError(
            "legacy Trainer reference must resolve to a BaseTrainer subclass"
        )
    return LegacyTrainerExecutable(
        plan=plan,
        trainer_cls=implementation,
        artifacts=artifacts,
    )


__all__ = ["LegacyTrainerExecutable", "create_executable"]
