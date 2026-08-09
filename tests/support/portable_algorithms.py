"""External-style factories and user functions for portable execution tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tributo.algorithms.api import (
        AlgorithmExecutionResult,
        ResolvedAlgorithmPlan,
        UserExecutionContext,
        WorkerExecutionResult,
    )
    from tributo.algorithms.spi import (
        MaterializedTabularInputView,
        PreparedInput,
        WorkerInputPayload,
    )

WORKER_CLEANUP_EVENTS: list[str] = []


def _joblib_worker_probe(task_index: int) -> dict[str, object]:
    """Return Ray identity from one joblib child task."""
    import ray

    runtime_context = ray.get_runtime_context()
    return {
        "task_index": task_index,
        "worker_id": str(runtime_context.get_worker_id()),
        "node_id": str(runtime_context.get_node_id()),
    }


class RayJoblibProbeClassifier:
    """Sklearn-compatible classifier that records Ray joblib task identities."""

    def __init__(self, n_jobs: int = 2, task_count: int = 4) -> None:
        self.n_jobs = n_jobs
        self.task_count = task_count

    def get_params(self, deep: bool = True) -> dict[str, object]:
        del deep
        return {"n_jobs": self.n_jobs, "task_count": self.task_count}

    def set_params(self, **params: object) -> RayJoblibProbeClassifier:
        for name, value in params.items():
            if name not in {"n_jobs", "task_count"}:
                raise ValueError(f"unknown probe parameter: {name}")
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"probe parameter {name} must be a positive integer")
            if name == "n_jobs":
                self.n_jobs = value
            else:
                self.task_count = value
        return self

    def fit(self, features: object, label: object) -> RayJoblibProbeClassifier:
        from joblib import Parallel, delayed

        del features, label
        self.worker_evidence_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_joblib_worker_probe)(index) for index in range(self.task_count)
        )
        return self

    def predict(self, features: object) -> list[int]:
        import numpy as np

        values = np.asarray(features)
        return [int(value >= 0) for value in values[:, 0]]


class UnboundedJoblibClassifier:
    """Sklearn-compatible estimator whose ``n_jobs=None`` must be rejected."""

    def __init__(self, n_jobs: int | None = None) -> None:
        self.n_jobs = n_jobs

    def get_params(self, deep: bool = True) -> dict[str, object]:
        del deep
        return {"n_jobs": self.n_jobs}

    def set_params(self, **params: object) -> UnboundedJoblibClassifier:
        if set(params) - {"n_jobs"}:
            raise ValueError("unknown unbounded probe parameter")
        self.n_jobs = cast(int | None, params.get("n_jobs", self.n_jobs))
        return self

    def fit(self, features: object, label: object) -> UnboundedJoblibClassifier:
        del features, label
        return self

    def predict(self, features: object) -> list[int]:
        import numpy as np

        return [0] * len(np.asarray(features))


class CallableFragment:
    """Callable object that the user-function channel must reject."""

    def __call__(self, context: UserExecutionContext) -> None:
        context.report({"invalid": True})


callable_fragment = CallableFragment()


def _make_lambda_fragment() -> object:
    return lambda context: None


lambda_fragment = _make_lambda_fragment()


def _make_closure_fragment() -> object:
    captured = "not-portable"

    def closure(context: UserExecutionContext) -> None:
        context.report({"captured": captured})

    return closure


closure_fragment = _make_closure_fragment()


def logistic_regression_factory() -> object:
    """Return an unfitted estimator through a module-qualified factory."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(random_state=7, solver="liblinear")


def logistic_pipeline_factory() -> object:
    """Return a cloneable preprocessing and estimator Pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(random_state=7, solver="liblinear")),
        ]
    )


def ray_joblib_probe_factory() -> object:
    """Return a cloneable estimator that uses the active joblib Backend."""
    return RayJoblibProbeClassifier()


def unbounded_joblib_factory() -> object:
    """Return an estimator with sklearn's unbounded ``n_jobs=None`` default."""
    return UnboundedJoblibClassifier()


def custom_training_fragment(context: UserExecutionContext) -> None:
    """Exercise the complete least-authority user reporting surface."""
    view = cast("MaterializedTabularInputView", context.get_input("train"))
    columns = view.columns()
    label_name = view.label_name
    if label_name is None:
        raise ValueError("custom training fragment requires a label")
    labels = columns[label_name]
    numeric_labels = [float(cast(str | int | float, value)) for value in labels]
    positive_rate = sum(numeric_labels, start=0.0) / len(numeric_labels)
    context.report(
        {
            "row_count": view.row_count,
            "positive_rate": positive_rate,
        }
    )
    context.report_outputs(
        {
            "threshold": context.configuration.get("threshold", 0.5),
            "worker_id": context.worker_metadata.get("worker_id", "direct"),
            "world_rank": context.worker_metadata.get("world_rank", 0),
            "world_size": context.worker_metadata.get("world_size", 1),
            "shard_values": list(columns[view.feature_names[0]]),
        }
    )
    rank = context.worker_metadata.get("world_rank", 0)
    context.stage_artifact(
        name=f"summary-rank-{rank}",
        kind="report",
        format="application/json",
        payload=json.dumps({"positive_rate": positive_rate}, sort_keys=True).encode(
            "utf-8"
        ),
    )
    context.report_checkpoint(
        payload=b"portable-checkpoint-v1",
        format="application/octet-stream",
        name=f"checkpoint-rank-{rank}",
    )


def reduce_custom_training_results(
    plan: ResolvedAlgorithmPlan,
    results: tuple[WorkerExecutionResult, ...],
) -> AlgorithmExecutionResult:
    """Reduce disjoint user-function shards in deterministic rank order."""
    from tributo.algorithms.api import AlgorithmExecutionResult

    del plan
    row_count = sum(int(result.execution.metrics["row_count"]) for result in results)
    positive_count = sum(
        float(result.execution.metrics["positive_rate"])
        * int(result.execution.metrics["row_count"])
        for result in results
    )
    ranks = [int(result.execution.outputs["world_rank"]) for result in results]
    worker_ids = [str(result.execution.outputs["worker_id"]) for result in results]
    shard_values = [
        value
        for result in results
        for value in result.execution.outputs["shard_values"]
    ]
    artifacts = tuple(
        artifact for result in results for artifact in result.execution.artifacts
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            "row_count": row_count,
            "positive_rate": positive_count / row_count,
        },
        outputs={
            "ranks": ranks,
            "worker_ids": worker_ids,
            "shard_values": shard_values,
        },
        artifacts=artifacts,
    )


def failing_training_fragment(context: UserExecutionContext) -> None:
    """Raise a stable user error after proving input access is available."""
    context.get_input("train")
    raise ValueError("user-visible failure")


def fail_rank_one_fragment(context: UserExecutionContext) -> None:
    """Fail only rank one to verify group-level failure semantics."""
    if context.worker_metadata.get("world_rank") == 1:
        raise ValueError("rank-one failure")
    custom_training_fragment(context)


def reducer_must_not_run(
    plan: ResolvedAlgorithmPlan,
    results: tuple[WorkerExecutionResult, ...],
) -> AlgorithmExecutionResult:
    """Fail if a Runtime incorrectly reduces a partially failed group."""
    del plan, results
    raise RuntimeError("reducer was called after a rank failed")


def sensitive_failure_fragment(context: UserExecutionContext) -> None:
    """Raise an error containing values that must not cross the Worker boundary."""
    context.get_input("train")
    raise ValueError(
        "password=hunter2 token=abc123 uri=https://alice:private@example.test/path"
    )


def cancellation_aware_fragment(context: UserExecutionContext) -> None:
    """Report the bounded cancellation snapshot exposed by the Runtime."""
    context.report_outputs({"cancelled": context.cancelled})


def prepare_tracked_input(payload: WorkerInputPayload) -> PreparedInput:
    """Wrap the fake Worker view with an observable exactly-once close."""
    from tributo.algorithms.input.fake import prepare_input
    from tributo.algorithms.spi import PreparedInput

    prepared = prepare_input(payload)
    return PreparedInput(
        prepared.views,
        close_callback=lambda: WORKER_CLEANUP_EVENTS.append("closed"),
    )


def invalid_returning_fragment(context: UserExecutionContext) -> str:
    """Return a value instead of using the explicit reporting protocol."""
    context.get_input("train")
    return "not-allowed"


def invalid_reporting_fragment(context: UserExecutionContext) -> None:
    """Report a non-finite metric to exercise user error normalization."""
    context.report({"loss": float("nan")})
