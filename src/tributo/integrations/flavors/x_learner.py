"""Safe fixed-composition X-Learner Bundle flavor."""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np

from tributo.exceptions import ModelLoadError, UnsupportedArtifactFormat
from tributo.exporting.models import ResolvedArtifact
from tributo.exporting.runtime import SECURITY_MODE_SAFE, BundleModel
from tributo.integrations.sources.ray_xgboost import _booster_objective
from tributo.training.x_learner import (
    X_LEARNER_FORMULA,
    X_LEARNER_QUADRANT_CODES,
    X_LEARNER_STAGE_OBJECTIVES,
    X_LEARNER_STAGES,
    XLearnerModel,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class XLearnerFlavor:
    api_version: ClassVar[int] = 1
    flavor_id: ClassVar[str] = "x-learner-v1"
    supported_formats: ClassVar[tuple[str, ...]] = ("x-learner",)
    batch_supported: ClassVar[bool] = True
    serveable: ClassVar[bool] = False
    security_mode: ClassVar[str] = SECURITY_MODE_SAFE
    signature_required: ClassVar[bool] = True
    required_dependencies: ClassVar[tuple[str, ...]] = ("xgboost",)

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        del role, unsafe
        if architecture_id not in (None, "x_learner"):
            raise UnsupportedArtifactFormat(
                "x-learner-v1 requires x_learner architecture"
            )
        try:
            import xgboost

            metadata = json.loads(artifact.entrypoint_path.read_text(encoding="utf-8"))
            if metadata.get("api_version") != 1:
                raise ValueError("api version")
            if metadata.get("formula") != X_LEARNER_FORMULA:
                raise ValueError("formula")
            if metadata.get("quadrant_codes") != X_LEARNER_QUADRANT_CODES:
                raise ValueError("quadrant codes")
            components = metadata["components"]
            if not isinstance(components, dict) or set(components) != set(
                X_LEARNER_STAGES
            ):
                raise ValueError("component set")
            component_paths = tuple(components[stage] for stage in X_LEARNER_STAGES)
            if any(
                not isinstance(path, str) or not path for path in component_paths
            ) or len(set(component_paths)) != len(X_LEARNER_STAGES):
                raise ValueError("component paths")
            boosters = {}
            for stage in X_LEARNER_STAGES:
                booster = xgboost.Booster()
                booster.load_model(str(artifact.path_for(components[stage])))
                if _booster_objective(booster) != X_LEARNER_STAGE_OBJECTIVES[stage]:
                    raise ValueError(f"component objective {stage}")
                boosters[stage] = booster
            model = XLearnerModel(
                boosters,
                feature_names=tuple(metadata["feature_names"]),
                response_threshold=float(metadata["response_threshold"]),
                propensity_clip=tuple(metadata["propensity_clip"]),
            )
        except Exception as exc:
            raise ModelLoadError(
                f"X-Learner artifact is invalid ({type(exc).__name__})"
            ) from None
        return _XLearnerBundleModel(model)


class _XLearnerBundleModel:
    def __init__(self, model: XLearnerModel) -> None:
        self._model = model

    input_names = ("float_input",)
    output_names = ("mu0", "mu1", "tau0", "tau1", "propensity", "cate", "quadrant")
    input_dtypes = ("float32",)
    output_dtypes = (
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "int64",
    )

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return ((None, len(self._model.feature_names)),)

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return tuple((None,) for _ in self.output_names)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = self._model.predict(inputs["float_input"])
        outputs = {
            name: np.asarray(getattr(result, name), dtype=np.float32)
            for name in self.output_names
            if name != "quadrant"
        }
        outputs["quadrant"] = np.asarray(
            [X_LEARNER_QUADRANT_CODES[str(value)] for value in result.quadrant],
            dtype=np.int64,
        )
        return outputs


__all__ = ["XLearnerFlavor"]
