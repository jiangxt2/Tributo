"""First-party SHAP adapter with lazy optional imports."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np

from tributo.explainability.contracts import (
    Exactness,
    ExplainabilityRequest,
    FeatureAttribution,
)
from tributo.explainability.protocols import (
    ExplainableModelContext,
    NativeAttributionModel,
    PreparedExplainer,
    SupportDecision,
)
from tributo.util.annotations import PublicAPI


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
        if request.backend in ("auto", "tree") and context.native_attribution_id:
            native_model = context.model_object
            if not isinstance(native_model, NativeAttributionModel):
                return SupportDecision(
                    supported=False,
                    reason="native attribution model contract is unavailable",
                    backend="tree",
                    exactness="conditional",
                )
            return native_model.native_attribution_support(request)

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
            if context.predict is None:
                return SupportDecision(
                    supported=False,
                    reason=("model_agnostic SHAP requires a prediction kernel"),
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
            native_model = context.model_object
            if not isinstance(native_model, NativeAttributionModel):
                raise ValueError("native attribution model contract is unavailable")
            reference_data = (context.metadata or {}).get("reference_data")
            prepared = native_model.prepare_native_attribution(
                request,
                feature_names=feature_names,
                reference_data=(
                    np.asarray(reference_data) if reference_data is not None else None
                ),
            )
            return replace(
                prepared,
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
            "uv sync --extra explainability"
        ) from exc
    return shap


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
    return np.asarray(np.argmax(model_outputs, axis=1), dtype=np.int64)[:, None]


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
