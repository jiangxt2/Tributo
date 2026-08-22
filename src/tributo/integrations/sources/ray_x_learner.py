"""Staged Ray XGBoost checkpoints to one fixed X-Learner ExportSource."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar, Generator

from pydantic import BaseModel, ConfigDict

from tributo.exporting.models import CheckpointField, ExportCheckpointV1, ExportSource
from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider
from tributo.training.x_learner import (
    X_LEARNER_STAGE_OBJECTIVES,
    X_LEARNER_STAGES,
    XLearnerModel,
    XLearnerTrainingResult,
)
from tributo.util.annotations import PublicAPI


class _XLearnerSourceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


@PublicAPI(stability="alpha")
class RayXLearnerSourceProvider:
    """Open exactly five verified XGBoost stage checkpoints."""

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "ray-x-learner-v1"
    trainer_type: ClassVar[str] = "x_learner"
    priority: ClassVar[int] = 100

    def open_source(self, result: Any, config: BaseModel | None = None) -> Any:
        _XLearnerSourceOptions.model_validate(
            config.model_dump() if config is not None else {}
        )
        return _open_source(result)


@contextmanager
def _open_source(result: Any) -> Generator[ExportSource, None, None]:
    import xgboost

    if not isinstance(result, XLearnerTrainingResult):
        raise TypeError("X-Learner source requires XLearnerTrainingResult")
    if set(result.checkpoints) != set(X_LEARNER_STAGES):
        raise ValueError("X-Learner result is missing required stage checkpoints")
    provider = RayXGBoostSourceProvider()
    with ExitStack() as stack:
        sources = {
            stage: stack.enter_context(provider.open_source(result.checkpoints[stage]))
            for stage in X_LEARNER_STAGES
        }
        for stage, source in sources.items():
            if source.metadata.get("objective") != X_LEARNER_STAGE_OBJECTIVES[stage]:
                raise ValueError(
                    f"X-Learner stage {stage!r} has an incompatible XGBoost objective"
                )
        boosters = {stage: source.model_object for stage, source in sources.items()}
        fingerprint = hashlib.sha256(
            "|".join(
                sources[stage].source_fingerprint for stage in X_LEARNER_STAGES
            ).encode()
        ).hexdigest()[:16]
        fields = tuple(
            CheckpointField(name=name, dtype="float32", shape=("batch",))
            for name in ("mu0", "mu1", "tau0", "tau1", "propensity", "cate")
        ) + (CheckpointField(name="quadrant", dtype="int64", shape=("batch",)),)
        yield ExportSource(
            source_kind="x_learner_result",
            model_object=XLearnerModel(
                boosters,
                feature_names=result.feature_names,
                response_threshold=result.response_threshold,
                propensity_clip=result.propensity_clip,
            ),
            architecture_id="x_learner",
            feature_schema={"feature_names": list(result.feature_names)},
            metadata={
                "causal_study": dict(result.metrics),
                "framework": "xgboost",
                "framework_version": xgboost.__version__,
                "task_type": "causal_effect_estimation",
            },
            source_fingerprint=fingerprint,
            checkpoint_contract=ExportCheckpointV1(
                trainer_type="x_learner",
                architecture_id="x_learner",
                input_schema=(
                    CheckpointField(
                        name="float_input",
                        dtype="float32",
                        shape=("batch", len(result.feature_names)),
                    ),
                ),
                output_schema=fields,
                task_type="causal_effect_estimation",
                framework="xgboost",
                framework_version=xgboost.__version__,
                required_artifacts=("x_learner.json",),
            ),
        )


__all__ = ["RayXLearnerSourceProvider"]
