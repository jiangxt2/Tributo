"""Safe native XGBoost Bundle flavor for UBJ and JSON models."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np

from tributo.exceptions import ModelLoadError, UnsupportedArtifactFormat
from tributo.exporting.models import ResolvedArtifact
from tributo.exporting.runtime import SECURITY_MODE_SAFE, BundleModel
from tributo.integrations.xgboost_capabilities import supports_native_tree_shap
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class XGBoostNativeFlavor:
    """Load a native Booster and expose the canonical Tributo tensor contract."""

    api_version: ClassVar[int] = 1
    flavor_id: ClassVar[str] = "xgboost-native-v1"
    supported_formats: ClassVar[tuple[str, ...]] = ("ubj", "xgboost-json")
    batch_supported: ClassVar[bool] = True
    serveable: ClassVar[bool] = True
    security_mode: ClassVar[str] = SECURITY_MODE_SAFE
    signature_required: ClassVar[bool] = True
    required_dependencies: ClassVar[tuple[str, ...]] = ("xgboost",)
    operations: ClassVar[tuple[str, ...]] = (
        "prediction.batch",
        "prediction.online",
    )
    conditional_operations: ClassVar[tuple[str, ...]] = ("attribution.tree-shap",)

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        """Load a JSON/UBJ Booster without executing Python model code."""
        del role, unsafe
        if architecture_id not in (None, "xgboost"):
            raise UnsupportedArtifactFormat(
                f"xgboost-native-v1 cannot load architecture {architecture_id!r}"
            )
        try:
            import xgboost
        except ImportError as exc:
            raise ModelLoadError(
                "xgboost-native-v1 requires the 'training' extra"
            ) from exc
        booster = xgboost.Booster()
        try:
            booster.load_model(str(artifact.entrypoint_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load native XGBoost model ({type(exc).__name__})"
            ) from None
        return _XGBoostNativeModel(booster, xgboost)


class _XGBoostNativeModel:
    def __init__(self, booster: Any, xgboost: Any) -> None:
        self._booster = booster
        self._xgboost = xgboost
        self._objective, self._num_classes, self._booster_kind = _model_metadata(
            booster
        )
        self._num_features = int(booster.num_features())
        self._classification = self._objective.startswith(("binary:", "multi:"))
        if self._objective == "binary:hinge":
            raise UnsupportedArtifactFormat(
                "binary:hinge does not expose probabilities required by the "
                "canonical classification signature"
            )

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("float_input",)

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("label", "probabilities") if self._classification else ("prediction",)

    @property
    def input_dtypes(self) -> tuple[str, ...]:
        return ("float32",)

    @property
    def output_dtypes(self) -> tuple[str, ...]:
        return ("int64", "float32") if self._classification else ("float32",)

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return ((None, self._num_features),)

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        if self._classification:
            return ((None,), (None, self._num_classes))
        return ((None, 1),)

    @property
    def native_attribution_id(self) -> str | None:
        if supports_native_tree_shap(
            booster_kind=self._booster_kind,
            objective=self._objective,
        ):
            return "xgboost-tree-shap-v1"
        return None

    @property
    def native_model_object(self) -> Any:
        return self._booster

    @property
    def native_feature_names(self) -> tuple[str, ...]:
        return tuple(self._booster.feature_names or ())

    @property
    def native_objective(self) -> str | None:
        return self._objective

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tensor = np.asarray(inputs["float_input"], dtype=np.float32)
        matrix = self._xgboost.DMatrix(
            tensor,
            feature_names=self._booster.feature_names,
        )
        if not self._classification:
            prediction = np.asarray(self._booster.predict(matrix), dtype=np.float32)
            return {"prediction": prediction.reshape(-1, 1)}
        if self._objective.startswith("multi:"):
            margins = np.asarray(
                self._booster.predict(matrix, output_margin=True), dtype=np.float32
            )
            shifted = margins - margins.max(axis=1, keepdims=True)
            exponent = np.exp(shifted)
            probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        else:
            raw = np.asarray(self._booster.predict(matrix), dtype=np.float32).reshape(
                -1
            )
            if self._objective == "binary:logitraw":
                positive = 1.0 / (1.0 + np.exp(-raw))
            else:
                positive = raw
            probabilities = np.column_stack((1.0 - positive, positive)).astype(
                np.float32, copy=False
            )
        labels = probabilities.argmax(axis=1).astype(np.int64, copy=False)
        return {"label": labels, "probabilities": probabilities}


def _model_metadata(booster: Any) -> tuple[str, int, str]:
    try:
        config = json.loads(booster.save_config())
        learner = config["learner"]
        objective = str(learner["objective"]["name"])
        num_classes = max(2, int(learner["learner_model_param"]["num_class"]))
        booster_kind = str(learner["gradient_booster"]["name"])
        return objective, num_classes, booster_kind
    except Exception as exc:
        raise ModelLoadError(
            f"Native XGBoost metadata is invalid ({type(exc).__name__})"
        ) from None


__all__ = ["XGBoostNativeFlavor"]
