"""Shared validation for portable metrics returned by FIT_ONLY executions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
)

# These fields are runtime evidence and are carried by ExecutionReceipt or
# WorkerExecutionResult metadata. They must not be exposed as user metrics.
FIT_ONLY_EVIDENCE_METRIC_NAMES = frozenset(
    {
        "execution_workers",
        "model_state_digest",
        "world_size",
        "state_coordination",
        "collective_backend",
        "checkpoint_owner_rank",
        "metric_reducers",
    }
)


def portable_fit_only_metrics(metrics: Mapping[Any, Any]) -> dict[str, Any]:
    """Return user metrics after one fail-closed portable-value validation.

    Runtime evidence fields are intentionally removed before validation. Any
    remaining invalid key or value is an execution error rather than a metric
    silently disappearing from a successful result.
    """
    if not isinstance(metrics, Mapping):
        raise AlgorithmExecutionError(
            "FIT_ONLY user metrics must be provided as a mapping"
        )
    user_metrics = {
        name: value
        for name, value in metrics.items()
        if not (isinstance(name, str) and name in FIT_ONLY_EVIDENCE_METRIC_NAMES)
    }
    if any(not isinstance(name, str) for name in user_metrics):
        raise AlgorithmExecutionError("FIT_ONLY user metric names must be strings")
    try:
        validated = AlgorithmExecutionResult(
            status="succeeded",
            metrics=user_metrics,
        )
    except AlgorithmConfigurationError as exc:
        raise AlgorithmExecutionError(
            "FIT_ONLY user metrics must contain only portable JSON values"
        ) from exc
    return dict(validated.metrics)
