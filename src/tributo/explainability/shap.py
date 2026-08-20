"""First-party SHAP adapter with lazy optional imports."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tributo.explainability.contracts import (
    Exactness,
    ExplainabilityRequest,
    FeatureAttribution,
)
from tributo.explainability.protocols import (
    ExplainableModelContext,
    PreparedExplainer,
    SupportDecision,
)
from tributo.util.annotations import PublicAPI

_NATIVE_OUTPUT_TARGETS = frozenset({"model_output", "raw", "raw_margin"})


@dataclass(frozen=True)
class _NativeExplanation:
    values: np.ndarray
    data: np.ndarray
    base_values: np.ndarray
    model_outputs: np.ndarray


class _NativeTreeExplainer:
    """Strict XGBoost contribution wrapper with a SHAP-like result shape."""

    def __init__(
        self,
        booster: Any,
        *,
        feature_names: tuple[str, ...],
        objective: str | None,
    ) -> None:
        self._booster = booster
        self._feature_names = feature_names
        self._objective = objective

    def __call__(
        self,
        values: np.ndarray,
        *,
        check_additivity: bool = False,
    ) -> _NativeExplanation:
        del check_additivity
        import xgboost

        data = np.asarray(values, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"XGBoost native TreeSHAP input must be 2-D, got {data.shape}"
            )
        if self._feature_names and len(self._feature_names) != data.shape[1]:
            raise ValueError(
                "XGBoost feature names count does not match the explanation input"
            )
        feature_types = tuple(self._booster.feature_types or ())
        if feature_types and len(feature_types) != data.shape[1]:
            raise ValueError(
                "XGBoost feature types count does not match the explanation input"
            )
        matrix = xgboost.DMatrix(
            data,
            feature_names=list(self._feature_names) or None,
            feature_types=list(feature_types) or None,
        )
        contributions = np.asarray(
            self._booster.predict(
                matrix,
                pred_contribs=True,
                approx_contribs=False,
                strict_shape=True,
            )
        )
        model_outputs = np.asarray(
            self._booster.predict(
                matrix,
                output_margin=True,
                strict_shape=True,
            )
        )
        if model_outputs.ndim != 2 or model_outputs.shape[0] != data.shape[0]:
            raise ValueError(
                "XGBoost raw outputs violate the strict shape contract: "
                f"{model_outputs.shape}"
            )
        output_count = model_outputs.shape[1]
        if self._objective and self._objective.startswith("multi:"):
            if output_count < 2:
                raise ValueError(
                    "XGBoost multiclass raw outputs must contain multiple groups"
                )
        elif self._objective and _supported_objective(self._objective):
            if output_count != 1:
                raise ValueError(
                    "XGBoost regression/binary raw outputs must contain one group"
                )
        expected_shape = (data.shape[0], output_count, data.shape[1] + 1)
        if contributions.shape != expected_shape:
            raise ValueError(
                "XGBoost contributions violate the strict shape contract: "
                f"expected {expected_shape}, got {contributions.shape}"
            )
        return _NativeExplanation(
            values=np.transpose(contributions[:, :, :-1], (0, 2, 1)),
            data=data,
            base_values=contributions[:, :, -1],
            model_outputs=model_outputs,
        )


@PublicAPI(stability="alpha")
class ShapAdapter:
    """SHAP TreeExplainer and ONNX prediction-function adapter."""

    api_version: ClassVar[int] = 1
    adapter_id: ClassVar[str] = "shap-v1"
    adapter_version: ClassVar[str] = "1"

    @classmethod
    def supports(
        cls,
        context: ExplainableModelContext,
        request: ExplainabilityRequest,
    ) -> SupportDecision:
        if (
            request.backend in ("auto", "tree")
            and context.flavor_id == "xgboost-native-v1"
        ):
            native_strategy = _uses_native_tree_strategy(context, request)
            missing = _missing("xgboost")
            if not native_strategy:
                missing += _missing("shap")
            if context.artifact_format not in {"ubj", "xgboost-json"}:
                return SupportDecision(
                    supported=False,
                    reason="Tree SHAP requires an XGBoost UBJ or JSON artifact",
                    required_dependencies=missing,
                    backend="tree",
                    exactness="exact",
                )
            if context.objective and not _supported_objective(context.objective):
                return SupportDecision(
                    supported=False,
                    reason=f"Unsupported XGBoost objective {context.objective!r}",
                    backend="tree",
                    exactness="exact",
                )
            if request.output_target not in {
                "model_output",
                "raw",
                "raw_margin",
                "probability",
                "log_loss",
            }:
                return SupportDecision(
                    supported=False,
                    reason=(
                        "Tree SHAP output_target must be one of model_output, raw, "
                        "raw_margin, probability, or log_loss"
                    ),
                    backend="tree",
                    exactness="exact",
                )
            if request.output_selection == "predicted":
                if not native_strategy:
                    return SupportDecision(
                        supported=False,
                        reason=(
                            "output_selection='predicted' is only supported for "
                            "native XGBoost raw model outputs"
                        ),
                        backend="tree",
                        exactness="exact",
                    )
                if not context.objective or not context.objective.startswith(
                    ("binary:", "multi:")
                ):
                    return SupportDecision(
                        supported=False,
                        reason=(
                            "output_selection='predicted' requires an XGBoost "
                            "classification objective"
                        ),
                        backend="tree",
                        exactness="exact",
                    )
            if (
                request.output_target in {"probability", "log_loss"}
                and request.reference is None
            ):
                return SupportDecision(
                    supported=False,
                    reason=(
                        f"Tree SHAP output_target={request.output_target!r} requires "
                        "an explicit reference binding"
                    ),
                    backend="tree",
                    exactness="exact",
                )
            if request.output_target == "log_loss" and request.label_column is None:
                return SupportDecision(
                    supported=False,
                    reason="Tree SHAP log_loss output requires label_column",
                    backend="tree",
                    exactness="exact",
                )
            if missing:
                return SupportDecision(
                    supported=False,
                    reason="Tree SHAP dependencies are not installed",
                    required_dependencies=missing,
                    backend="tree",
                    exactness="exact",
                )
            return SupportDecision(
                supported=True,
                backend="tree",
                exactness="exact",
            )

        if request.backend in ("auto", "model_agnostic"):
            missing = _missing("shap")
            if request.output_selection == "predicted":
                return SupportDecision(
                    supported=False,
                    reason=(
                        "output_selection='predicted' is not implemented for "
                        "model-agnostic SHAP"
                    ),
                    backend="model_agnostic",
                    exactness="approximate",
                )
            if request.output_target == "log_loss":
                return SupportDecision(
                    supported=False,
                    reason="log_loss output is only supported by the tree backend",
                    backend="model_agnostic",
                    exactness="approximate",
                )
            if not request.allow_approximate:
                return SupportDecision(
                    supported=False,
                    reason=(
                        "ONNX model-agnostic SHAP is approximate and requires "
                        "allow_approximate=true"
                    ),
                    backend="model_agnostic",
                    exactness="approximate",
                )
            if context.flavor_id != "onnx-runtime-v1" or context.predict is None:
                return SupportDecision(
                    supported=False,
                    reason=(
                        "model_agnostic SHAP requires an onnx-runtime-v1 "
                        "prediction function"
                    ),
                    required_dependencies=missing,
                    backend="model_agnostic",
                    exactness="approximate",
                )
            if request.reference is None:
                return SupportDecision(
                    supported=False,
                    reason="model_agnostic SHAP requires an explicit reference binding",
                    required_dependencies=missing,
                    backend="model_agnostic",
                    exactness="approximate",
                )
            if missing:
                return SupportDecision(
                    supported=False,
                    reason="SHAP is not installed",
                    required_dependencies=missing,
                    backend="model_agnostic",
                    exactness="approximate",
                )
            return SupportDecision(
                supported=True,
                backend="model_agnostic",
                exactness="approximate",
            )

        return SupportDecision(
            supported=False,
            reason=f"SHAP backend {request.backend!r} is not implemented",
            backend=request.backend,
            exactness="conditional",
        )

    def prepare(
        self,
        context: ExplainableModelContext,
        request: ExplainabilityRequest,
    ) -> PreparedExplainer:
        decision = self.supports(context, request)
        if not decision.supported:
            raise ValueError(decision.reason)
        feature_names = context.feature_names
        if decision.backend == "tree":
            booster = context.model_object
            if booster is None:
                if context.artifact_path is None:
                    raise ValueError("Tree SHAP requires a verified artifact path")
                import xgboost

                booster = xgboost.Booster()
                booster.load_model(str(context.artifact_path))
            if not feature_names:
                feature_names = tuple(booster.feature_names or ())
            if _uses_native_tree_strategy(context, request):
                booster_kind = _xgboost_booster_kind(booster)
                if booster_kind not in {"gbtree", "dart"}:
                    raise ValueError(
                        "XGBoost native TreeSHAP requires a gbtree or dart "
                        f"booster, got {booster_kind!r}"
                    )
                return PreparedExplainer(
                    backend="tree",
                    exactness="exact",
                    explain=_NativeTreeExplainer(
                        booster,
                        feature_names=feature_names,
                        objective=context.objective,
                    ),
                    feature_names=feature_names,
                    preprocessor_digest=context.preprocessor_digest,
                    feature_map_digest=context.feature_map_digest,
                )
            shap = _require_shap()
            model_output = _tree_model_output(request.output_target)
            kwargs: dict[str, Any] = {}
            if model_output is not None:
                kwargs["model_output"] = model_output
            reference_data = (context.metadata or {}).get("reference_data")
            if reference_data is not None:
                kwargs["data"] = np.asarray(reference_data)
                kwargs["feature_perturbation"] = "interventional"
            explainer = shap.TreeExplainer(booster, **kwargs)
            predict = None
            if request.output_target != "log_loss":
                predict_type = "margin" if model_output in {None, "raw"} else "value"

                def predict(values: np.ndarray) -> np.ndarray:
                    return np.asarray(
                        booster.inplace_predict(
                            np.asarray(values, dtype=np.float32),
                            predict_type=predict_type,
                            validate_features=False,
                        )
                    )

            return PreparedExplainer(
                backend="tree",
                exactness="exact",
                explain=explainer,
                feature_names=feature_names,
                predict=predict,
                preprocessor_digest=context.preprocessor_digest,
                feature_map_digest=context.feature_map_digest,
            )

        shap = _require_shap()
        metadata = context.metadata or {}
        reference_data = metadata.get("reference_data")
        if reference_data is None:
            raise ValueError("model_agnostic SHAP requires materialized reference data")
        masker = np.asarray(reference_data)
        if masker.ndim != 2 or masker.shape[0] == 0:
            raise ValueError(
                "reference data must be a non-empty two-dimensional matrix"
            )
        explainer = shap.Explainer(context.predict, masker)
        return PreparedExplainer(
            backend="model_agnostic",
            exactness="approximate",
            explain=explainer,
            feature_names=feature_names,
            predict=context.predict,
            preprocessor_digest=context.preprocessor_digest,
            feature_map_digest=context.feature_map_digest,
        )

    def explain_batch(
        self,
        prepared: PreparedExplainer,
        batch: np.ndarray,
        *,
        input_ids: tuple[str, ...],
        model_digest: str,
        request: ExplainabilityRequest,
        labels: np.ndarray | None = None,
    ) -> tuple[FeatureAttribution, ...]:
        if batch.ndim != 2:
            raise ValueError(f"explain batch must be 2-D, got shape={batch.shape}")
        if prepared.backend == "tree":
            if request.output_target == "log_loss":
                if labels is None:
                    raise ValueError("Tree SHAP log_loss output requires labels")
                labels = np.asarray(labels)
                if labels.ndim != 1 or labels.shape[0] != batch.shape[0]:
                    raise ValueError("log_loss labels must be a one-dimensional batch")
                explanation = prepared.explain(batch, y=labels, check_additivity=False)
            else:
                explanation = prepared.explain(batch, check_additivity=False)
        else:
            explanation = prepared.explain(batch)
        values = np.asarray(getattr(explanation, "values", explanation))
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3 or values.shape[0] != len(input_ids):
            raise ValueError(f"SHAP values have unsupported shape {values.shape}")

        data = getattr(explanation, "data", batch)
        data_array = np.asarray(data)
        base_values = _normalise_base_values(
            getattr(explanation, "base_values", None), values, labels=labels
        )
        if request.output_target == "log_loss":
            model_outputs = values.sum(axis=1) + base_values
        elif getattr(explanation, "model_outputs", None) is not None:
            model_outputs = _normalise_model_outputs(
                explanation.model_outputs,
                values,
            )
        elif prepared.predict is not None:
            model_outputs = _normalise_model_outputs(prepared.predict(batch), values)
        else:
            model_outputs = None
        if prepared.backend == "tree" and model_outputs is not None:
            reconstructed = values.sum(axis=1) + base_values
            if not np.allclose(
                reconstructed,
                model_outputs,
                rtol=1e-3,
                atol=1e-4,
                equal_nan=False,
            ):
                raise ValueError(
                    "Tree SHAP additivity check failed for the declared "
                    f"output_target={request.output_target!r}"
                )
        feature_names = prepared.feature_names or tuple(
            f"feature_{index}" for index in range(values.shape[1])
        )
        if len(feature_names) != values.shape[1]:
            feature_names = tuple(
                f"feature_{index}" for index in range(values.shape[1])
            )
        output_indexes = _selected_output_indexes(
            request,
            model_outputs=model_outputs,
            row_count=values.shape[0],
            output_count=values.shape[2],
        )
        ranking_values = values
        if request.output_selection == "predicted":
            ranking_values = np.take_along_axis(
                values,
                output_indexes[:, None, :],
                axis=2,
            )
        max_features = (
            request.limits.top_k or request.limits.max_features or values.shape[1]
        )
        if max_features < values.shape[1]:
            feature_indexes = np.argsort(-np.abs(ranking_values).max(axis=2), axis=1)[
                :, :max_features
            ]
        else:
            feature_indexes = np.tile(np.arange(values.shape[1]), (values.shape[0], 1))

        rows: list[FeatureAttribution] = []
        for row_index, input_id in enumerate(input_ids):
            for feature_index in feature_indexes[row_index]:
                for output_index in output_indexes[row_index]:
                    feature_position = int(feature_index)
                    output_position = int(output_index)
                    rows.append(
                        FeatureAttribution(
                            input_id=input_id,
                            output_id=f"output_{output_position}",
                            feature_id=str(feature_names[feature_position]),
                            feature_name=str(feature_names[feature_position]),
                            feature_view=request.feature_view,
                            feature_value=(
                                _scalar(data_array[row_index, feature_position])
                                if request.result_policy.allow_sensitive_features
                                else None
                            ),
                            contribution=float(
                                values[
                                    row_index,
                                    feature_position,
                                    output_position,
                                ]
                            ),
                            base_value=float(base_values[row_index, output_position]),
                            model_output=(
                                _scalar(model_outputs[row_index, output_position])
                                if model_outputs is not None
                                else None
                            ),
                            output_target=request.output_target,
                            explainer=request.explainer,
                            backend=prepared.backend,
                            exactness=prepared.exactness,
                            model_digest=model_digest,
                            preprocessor_digest=prepared.preprocessor_digest,
                            feature_map_digest=prepared.feature_map_digest,
                        )
                    )
        return tuple(rows)

    def summarize(
        self,
        attribution_batch: tuple[FeatureAttribution, ...],
        *,
        exactness: Exactness | None = None,
    ) -> tuple[FeatureAttribution, ...]:
        """Filter v1 rows by exactness without changing their long-format shape.

        Aggregation is intentionally not part of the v1 summary contract;
        callers can request an exactness slice while retaining provenance.
        """
        if exactness is None:
            return attribution_batch
        if exactness not in {"exact", "approximate", "conditional"}:
            raise ValueError(f"unsupported exactness filter {exactness!r}")
        return tuple(row for row in attribution_batch if row.exactness == exactness)


def _missing(*names: str) -> tuple[str, ...]:
    return tuple(name for name in names if importlib.util.find_spec(name) is None)


def _require_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability. Install with: "
            "uv sync --extra explainability (SHAP 0.52.x is required for "
            "vector-valued XGBoost base scores)"
        ) from exc
    return shap


def _supported_objective(objective: str) -> bool:
    return objective.startswith(("binary:", "multi:")) or objective in {
        "reg:squarederror",
        "reg:logistic",
    }


def _xgboost_booster_kind(booster: Any) -> str:
    try:
        config = json.loads(booster.save_config())
        kind = config["learner"]["gradient_booster"]["name"]
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError(
            "XGBoost booster type metadata is missing or invalid"
        ) from None
    if not isinstance(kind, str) or not kind:
        raise ValueError("XGBoost booster type metadata is missing or invalid")
    return kind


def _uses_native_tree_strategy(
    context: ExplainableModelContext,
    request: ExplainabilityRequest,
) -> bool:
    return (
        context.flavor_id == "xgboost-native-v1"
        and request.output_target in _NATIVE_OUTPUT_TARGETS
        and request.reference is None
    )


def _selected_output_indexes(
    request: ExplainabilityRequest,
    *,
    model_outputs: np.ndarray | None,
    row_count: int,
    output_count: int,
) -> np.ndarray:
    if request.output_selection == "all":
        return np.tile(np.arange(output_count), (row_count, 1))
    if model_outputs is None:
        raise ValueError("output_selection='predicted' requires model outputs")
    if model_outputs.shape != (row_count, output_count):
        raise ValueError(
            "model outputs do not match the attribution output shape for "
            "output_selection='predicted'"
        )
    if output_count == 1:
        return np.zeros((row_count, 1), dtype=np.int64)
    return np.argmax(model_outputs, axis=1)[:, None]


def _tree_model_output(output_target: str) -> str | None:
    return {
        "model_output": None,
        "raw": "raw",
        "raw_margin": "raw",
        "probability": "probability",
        "log_loss": "log_loss",
    }.get(output_target, output_target)


def _normalise_base_values(
    raw: Any,
    values: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    if raw is None:
        return np.zeros((values.shape[0], values.shape[2]), dtype=np.float64)
    if callable(raw):
        if labels is None:
            raise ValueError("callable SHAP base_values require labels")
        raw = np.asarray([raw(label) for label in labels])
    array = np.asarray(raw)
    if array.dtype == object:
        dynamic_values = array.reshape(-1)
        if any(callable(value) for value in dynamic_values):
            if labels is None:
                raise ValueError("callable SHAP base_values require labels")
            if not all(callable(value) for value in dynamic_values):
                raise ValueError("SHAP base_values contain mixed callable values")
            if array.ndim != 1 or array.shape[0] != values.shape[0]:
                raise ValueError(
                    "callable SHAP base_values must contain one value per sample"
                )
            array = np.asarray(
                [value(label) for value, label in zip(array, labels, strict=True)]
            )
    if array.ndim == 0:
        return np.full((values.shape[0], values.shape[2]), float(array))
    if array.ndim == 1:
        if array.shape[0] == values.shape[0]:
            return np.repeat(array[:, None], values.shape[2], axis=1)
        return np.repeat(array[None, :], values.shape[0], axis=0)
    if array.ndim == 2:
        if array.shape[0] == values.shape[0]:
            if array.shape[1] == values.shape[2]:
                return array.astype(float)
            return np.repeat(array[:, :1], values.shape[2], axis=1).astype(float)
        return (
            np.repeat(array[:1, :1], values.shape[0], axis=0)
            .repeat(values.shape[2], axis=1)
            .astype(float)
        )
    return np.zeros((values.shape[0], values.shape[2]), dtype=np.float64)


def _normalise_model_outputs(raw: Any, values: np.ndarray) -> np.ndarray:
    array = np.asarray(raw)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != values.shape[0]:
        raise ValueError(f"model outputs have unsupported shape {array.shape}")
    if array.shape[1] != values.shape[2]:
        raise ValueError(
            f"model output columns {array.shape[1]} do not match "
            f"SHAP outputs {values.shape[2]}"
        )
    return array


def _scalar(value: Any) -> float | int | str | bool | None:
    value = np.asarray(value)
    if value.ndim == 0:
        item = value.item()
        return item if isinstance(item, (float, int, str, bool)) else str(item)
    return str(value.tolist())


__all__ = ["ShapAdapter"]
