"""Worker-only managed scikit-learn estimator integration."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Protocol, cast

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
)
from tributo.util.annotations import DeveloperAPI

_MAX_MANAGED_MODEL_BYTES = 64 * 1024 * 1024


@contextmanager
def _execution_backend(plan: ResolvedAlgorithmPlan) -> Iterator[None]:
    """Enter Ray joblib only for an explicitly framework-managed plan."""
    if plan.runtime.topology is not RuntimeTopology.FRAMEWORK_MANAGED:
        with nullcontext():
            yield
        return
    try:
        from joblib import parallel_backend
        from ray.util.joblib import register_ray

        register_ray()
        with parallel_backend(
            "ray",
            n_jobs=plan.runtime.framework_parallelism,
        ):
            yield
    except Exception as exc:
        raise AlgorithmExecutionError(
            f"managed sklearn Ray joblib execution failed: {exc}"
        ) from exc


class _Estimator(Protocol):
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> _Estimator: ...

    def fit(self, features: object, label: object) -> _Estimator: ...

    def predict(self, features: object) -> object: ...


def _only_input(
    context: AlgorithmExecutionContext,
) -> MaterializedTabularInputView:
    if len(context.inputs) != 1:
        raise AlgorithmInputError(
            "managed sklearn execution requires exactly one named input"
        )
    view = next(iter(context.inputs.values()))
    if not isinstance(view, MaterializedTabularInputView):
        raise AlgorithmInputError(
            "managed sklearn requires a MaterializedTabularInputView"
        )
    if view.row_count == 0:
        raise AlgorithmInputError("managed sklearn input must not be empty")
    return view


def _arrays(
    view: MaterializedTabularInputView,
    *,
    require_label: bool,
) -> tuple[Any, Any | None]:
    import numpy as np

    columns = view.columns()
    missing = [name for name in view.feature_names if name not in columns]
    if missing:
        raise AlgorithmInputError(
            f"managed sklearn input is missing feature column(s): {missing}"
        )
    try:
        features = np.column_stack(
            [np.asarray(columns[name]) for name in view.feature_names]
        )
        label = None
        if view.label_name is not None:
            if view.label_name not in columns:
                raise AlgorithmInputError(
                    f"managed sklearn input is missing label {view.label_name!r}"
                )
            label = np.asarray(columns[view.label_name])
    except AlgorithmInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise AlgorithmInputError(
            f"managed sklearn could not materialize tabular input: {exc}"
        ) from exc
    if require_label and label is None:
        raise AlgorithmInputError("managed sklearn fit/evaluate requires a label_name")
    return features, label


def _model_artifact(artifacts: tuple[ArtifactDraft, ...]) -> ArtifactDraft:
    candidates = [artifact for artifact in artifacts if artifact.kind == "model"]
    if len(candidates) != 1:
        raise AlgorithmExecutionError(
            "managed sklearn predict/evaluate requires one model artifact"
        )
    artifact = candidates[0]
    if artifact.format != "application/x-python-pickle" or not artifact.trusted:
        raise AlgorithmExecutionError(
            "managed sklearn refuses an undeclared or untrusted pickle artifact"
        )
    if len(artifact.payload) > _MAX_MANAGED_MODEL_BYTES:
        raise AlgorithmExecutionError(
            "trusted sklearn model exceeds the 64 MiB execution limit"
        )
    return artifact


@DeveloperAPI
class ManagedSklearnExecutable:
    """Implement bounded estimator capabilities inside one Ray Worker."""

    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        estimator_factory: Callable[[], object],
        artifacts: tuple[ArtifactDraft, ...],
    ) -> None:
        self._plan = plan
        self._estimator_factory = estimator_factory
        self._artifacts = artifacts

    def _new_estimator(self) -> _Estimator:
        from sklearn.base import clone

        try:
            template = self._estimator_factory()
            if isinstance(template, type):
                raise TypeError("estimator factory returned a class")
            for method in ("get_params", "set_params", "fit", "predict"):
                if not callable(getattr(template, method, None)):
                    raise TypeError(f"estimator factory result lacks {method}()")
            estimator = clone(template)
            estimator.set_params(**dict(self._plan.algorithm_config))
            self._validate_framework_parallelism(estimator)
            return cast(_Estimator, estimator)
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"invalid managed sklearn estimator or configuration: {exc}"
            ) from exc

    def _validate_framework_parallelism(self, estimator: _Estimator) -> None:
        if self._plan.runtime.topology is not RuntimeTopology.FRAMEWORK_MANAGED:
            return
        parameters = estimator.get_params(deep=True)
        parallelism_parameters = {
            name: value
            for name, value in parameters.items()
            if name == "n_jobs" or name.endswith("__n_jobs")
        }
        if not parallelism_parameters:
            raise AlgorithmExecutionError(
                "framework_managed sklearn requires a declared n_jobs parameter"
            )
        invalid = {
            name: value
            for name, value in parallelism_parameters.items()
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > self._plan.runtime.framework_parallelism
            )
        }
        if invalid:
            raise AlgorithmExecutionError(
                "sklearn n_jobs must be positive and no greater than the declared "
                f"framework parallelism: {sorted(invalid)}"
            )

    def _load_estimator(self) -> _Estimator:
        artifact = _model_artifact(self._artifacts)
        try:
            estimator = pickle.loads(artifact.payload)
        except Exception as exc:
            raise AlgorithmExecutionError(
                "trusted sklearn model artifact could not be loaded"
            ) from exc
        if not callable(getattr(estimator, "predict", None)):
            raise AlgorithmExecutionError(
                "trusted model artifact does not contain a predictor"
            )
        return cast(_Estimator, estimator)

    @staticmethod
    def _predictions(estimator: _Estimator, features: object) -> list[object]:
        import numpy as np

        try:
            predicted = estimator.predict(features)
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"managed sklearn prediction failed: {exc}"
            ) from exc
        return cast(list[object], np.asarray(predicted).tolist())

    @staticmethod
    def _metrics(
        predictions: list[object],
        label: object | None,
        row_count: int,
    ) -> dict[str, int | float]:
        import numpy as np

        metrics: dict[str, int | float] = {"row_count": row_count}
        if label is not None:
            labels = np.asarray(label)
            predicted = np.asarray(predictions)
            metrics["accuracy"] = float(np.mean(predicted == labels))
        return metrics

    def fit(
        self,
        context: AlgorithmExecutionContext,
    ) -> AlgorithmExecutionResult:
        """Clone, configure, fit, evaluate, and stage the estimator."""
        view = _only_input(context)
        features, label = _arrays(view, require_label=True)
        estimator = self._new_estimator()
        try:
            with _execution_backend(self._plan):
                estimator.fit(features, label)
        except Exception as exc:
            raise AlgorithmExecutionError(f"managed sklearn fit failed: {exc}") from exc
        with _execution_backend(self._plan):
            predictions = self._predictions(estimator, features)
        artifacts: tuple[ArtifactDraft, ...] = ()
        if self._plan.implementation.artifact_format == "trusted_pickle":
            try:
                payload = pickle.dumps(estimator, protocol=5)
            except Exception as exc:
                raise AlgorithmExecutionError(
                    "managed sklearn estimator could not be serialized"
                ) from exc
            if len(payload) > _MAX_MANAGED_MODEL_BYTES:
                raise AlgorithmExecutionError(
                    "managed sklearn model exceeds the 64 MiB artifact limit"
                )
            artifacts = (
                ArtifactDraft.from_payload(
                    name="model",
                    kind="model",
                    format="application/x-python-pickle",
                    payload=payload,
                    trusted=True,
                ),
            )
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=self._metrics(predictions, label, view.row_count),
            outputs={"predictions": predictions},
            artifacts=artifacts,
        )

    def evaluate(
        self,
        context: AlgorithmExecutionContext,
    ) -> AlgorithmExecutionResult:
        """Evaluate one trusted model artifact against labeled input."""
        view = _only_input(context)
        features, label = _arrays(view, require_label=True)
        estimator = self._load_estimator()
        with _execution_backend(self._plan):
            predictions = self._predictions(estimator, features)
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=self._metrics(predictions, label, view.row_count),
            outputs={"predictions": predictions},
        )

    def predict(
        self,
        context: AlgorithmExecutionContext,
    ) -> AlgorithmExecutionResult:
        """Predict from one trusted model artifact without requiring a label."""
        view = _only_input(context)
        features, label = _arrays(view, require_label=False)
        estimator = self._load_estimator()
        with _execution_backend(self._plan):
            predictions = self._predictions(estimator, features)
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=self._metrics(predictions, label, view.row_count),
            outputs={"predictions": predictions},
        )


@DeveloperAPI
def create_executable(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[ArtifactDraft, ...],
) -> ManagedSklearnExecutable:
    """Create a managed estimator executable from a Worker-loaded factory."""
    if not callable(implementation):
        raise AlgorithmExecutionError("sklearn estimator factory is not callable")
    return ManagedSklearnExecutable(
        plan=plan,
        estimator_factory=implementation,
        artifacts=artifacts,
    )


__all__ = ["ManagedSklearnExecutable", "create_executable"]
