"""XGBoost native model round-trip and TreeSHAP capability validator."""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.errors import sanitize_error_message
from tributo.exporting.models import (
    ExportSource,
    FailureInfo,
    ResolvedArtifact,
    ValidationResult,
)
from tributo.integrations.xgboost_capabilities import supports_native_tree_shap
from tributo.util.annotations import PublicAPI


class _XGBoostNativeValidatorOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_samples: int = Field(default=2, ge=1)
    require_tree_shap: bool = False


@PublicAPI(stability="beta")
class XGBoostNativeRuntimeValidator:
    """Validate native prediction parity and optional exact TreeSHAP support."""

    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "xgboost-native-runtime-v1"
    options_model: ClassVar[type[BaseModel]] = _XGBoostNativeValidatorOptions

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        del upstream
        try:
            import numpy as np
            import xgboost

            source_booster = source.model_object
            if not isinstance(source_booster, xgboost.Booster):
                raise TypeError("XGBoost native validation requires a Booster source")

            started = time.perf_counter()
            loaded = xgboost.Booster()
            loaded.load_model(str(artifact.entrypoint_path))
            load_seconds = time.perf_counter() - started

            num_features = int(loaded.num_features())
            if num_features < 1 or num_features != int(source_booster.num_features()):
                raise ValueError("Reloaded Booster feature count does not match source")
            feature_names = tuple(source_booster.feature_names or ())
            _validate_feature_names(artifact, feature_names)

            num_samples = int(getattr(options, "num_samples", 2))
            values = np.zeros((num_samples, num_features), dtype=np.float32)
            source_matrix = xgboost.DMatrix(
                values,
                feature_names=list(feature_names) or None,
            )
            loaded_matrix = xgboost.DMatrix(
                values,
                feature_names=list(loaded.feature_names or ()) or None,
            )
            source_margin = np.asarray(
                source_booster.predict(
                    source_matrix,
                    output_margin=True,
                    strict_shape=True,
                )
            )
            loaded_margin = np.asarray(
                loaded.predict(
                    loaded_matrix,
                    output_margin=True,
                    strict_shape=True,
                )
            )
            if source_margin.shape != loaded_margin.shape or not np.allclose(
                source_margin,
                loaded_margin,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError("Reloaded Booster prediction does not match source")

            tree_shap_candidate = _tree_shap_candidate(loaded)
            if getattr(options, "require_tree_shap", False) and not tree_shap_candidate:
                raise ValueError(
                    "Exported Booster does not support the required TreeSHAP capability"
                )
            if tree_shap_candidate:
                contributions = np.asarray(
                    loaded.predict(
                        loaded_matrix,
                        pred_contribs=True,
                        approx_contribs=False,
                        strict_shape=True,
                    )
                )
                expected_shape = (
                    num_samples,
                    loaded_margin.shape[1],
                    num_features + 1,
                )
                if contributions.shape != expected_shape:
                    raise ValueError(
                        "TreeSHAP contributions violate the strict shape contract"
                    )
                if not np.allclose(
                    contributions.sum(axis=2),
                    loaded_margin,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError("TreeSHAP contributions fail margin additivity")

            return ValidationResult(
                validator_id=self.validator_id,
                status="passed",
                metrics={
                    "load_seconds": round(load_seconds, 6),
                    "feature_count": float(num_features),
                    "output_count": float(loaded_margin.shape[1]),
                    "tree_shap_candidate": float(tree_shap_candidate),
                },
            )
        except Exception as exc:
            return ValidationResult(
                validator_id=self.validator_id,
                status="failed",
                failure=FailureInfo(
                    code=type(exc).__name__,
                    category="validation",
                    message=sanitize_error_message(str(exc))[:4096],
                ),
            )


def _validate_feature_names(
    artifact: ResolvedArtifact,
    source_names: tuple[str, ...],
) -> None:
    sidecar = next(
        (
            artifact.path_for(file.relative_path)
            for file in artifact.descriptor.files
            if file.relative_path == "feature_names.json"
        ),
        None,
    )
    if sidecar is None:
        return
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(name, str) for name in raw):
        raise ValueError("feature_names.json must contain a string list")
    if source_names and tuple(raw) != source_names:
        raise ValueError("feature_names.json does not match the source Booster")


def _tree_shap_candidate(booster: Any) -> bool:
    config = json.loads(booster.save_config())
    learner = config["learner"]
    booster_kind = str(learner["gradient_booster"]["name"])
    objective = str(learner["objective"]["name"])
    return supports_native_tree_shap(
        booster_kind=booster_kind,
        objective=objective,
    )


__all__ = ["XGBoostNativeRuntimeValidator"]
