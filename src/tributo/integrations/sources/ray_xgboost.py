"""Ray XGBoost checkpoint → ExportSource provider.

Resolves a Ray Train ``XGBoostCheckpoint`` (or a local checkpoint directory)
into an ``ExportSource`` that the export pipeline can consume.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Generator, cast

from pydantic import BaseModel, ConfigDict

from tributo.exporting.models import CheckpointField, ExportCheckpointV1, ExportSource
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class _RayXGBoostSourceOptions(BaseModel):
    """Options for the Ray XGBoost source provider.

    Currently empty — reserved for future use (e.g. checkpoint filter).
    """

    model_config = ConfigDict(extra="forbid")


@PublicAPI(stability="beta")
class RayXGBoostSourceProvider:
    """Resolve a Ray XGBoost checkpoint to an ``ExportSource``.

    Accepts either a Ray Train ``Result`` object or a local path to a
    checkpoint directory.  Uses ``ray.train.xgboost.XGBoostCheckpoint``
    to load the booster.
    """

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "ray-xgboost-v1"
    trainer_type: ClassVar[str] = "xgboost"
    priority: ClassVar[int] = 100

    def open_source(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Any:  # ContextManager[ExportSource]
        """Open a Ray XGBoost checkpoint as an ``ExportSource``.

        Args:
            result: A Ray Train ``Result``, a path string, or a Path.
            config: Optional typed options (unused in v1).

        Yields:
            ``ExportSource`` with ``source_kind="xgboost_result"``.
        """
        return self._open(result)

    @contextmanager
    def _open(self, result: Any) -> Generator[ExportSource, None, None]:
        import xgboost
        from ray.train.xgboost import XGBoostCheckpoint

        # Resolve to a directory path.
        if hasattr(result, "checkpoint"):
            # Ray Train Result object.
            checkpoint = result.checkpoint
        elif hasattr(result, "to_directory"):
            # Direct checkpoint object.
            checkpoint = result
        elif isinstance(result, (str, Path)):
            checkpoint_dir = Path(result)
            # For local paths, use the path directly.
            checkpoint = _path_to_checkpoint(checkpoint_dir)
        else:
            raise TypeError(
                f"Expected Ray Result, Checkpoint, or path string, got {type(result)}"
            )

        if isinstance(checkpoint, Path):
            checkpoint_dir = checkpoint
            booster = xgboost.Booster()
            model_path = checkpoint_dir / "model.json"
            if model_path.exists():
                booster.load_model(str(model_path))
            else:
                # Try ubj format.
                ubj_path = checkpoint_dir / "model.ubj"
                if ubj_path.exists():
                    booster.load_model(str(ubj_path))
                else:
                    raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
        else:
            # Ray XGBoostCheckpoint.
            xgb_checkpoint = XGBoostCheckpoint.from_directory(checkpoint.to_directory())
            # get_model() is a runtime method not present in the ray.train
            # type stubs — cast to Any to access it.
            booster = cast(Any, xgb_checkpoint).get_model()

        # Compute source fingerprint.
        model_dump = booster.save_raw()
        fingerprint = hashlib.sha256(
            model_dump if isinstance(model_dump, bytes) else str(model_dump).encode()
        ).hexdigest()[:16]

        # Extract feature names and build the framework-neutral checkpoint
        # contract used by the manifest signature.
        feature_names = booster.feature_names
        n_features = (
            len(feature_names) if feature_names else int(booster.num_features())
        )
        effective_feature_names = feature_names or [f"f{i}" for i in range(n_features)]
        feature_schema: dict[str, Any] = {
            "feature_names": list(effective_feature_names)
        }
        objective = _booster_objective(booster)
        is_classification = objective.startswith(("binary:", "multi:"))
        task_type = (
            "classification"
            if is_classification
            else "regression"
            if objective.startswith("reg:")
            else "unknown"
        )
        if is_classification:
            n_classes = _booster_num_classes(booster, objective)
            output_schema = (
                CheckpointField(name="label", dtype="int64", shape=("batch",)),
                CheckpointField(
                    name="probabilities",
                    dtype="float32",
                    shape=("batch", n_classes),
                ),
            )
        else:
            output_schema = (
                # onnxmltools XGBRegressor emits a two-dimensional [batch, 1] tensor.
                CheckpointField(name="prediction", dtype="float32", shape=("batch", 1)),
            )
        contract = ExportCheckpointV1(
            trainer_type="xgboost",
            architecture_id="xgboost",
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", n_features),
                ),
            ),
            output_schema=output_schema,
            preprocessing={"type": "none"},
            task_type=task_type,
            framework="xgboost",
            framework_version=xgboost.__version__,
            checkpoint_format_version=1,
        )

        source = ExportSource(
            source_kind="xgboost_result",
            model_object=booster,
            architecture_id="xgboost",
            feature_schema=feature_schema,
            metadata={
                "framework": "xgboost",
                "framework_version": xgboost.__version__,
                "n_features": n_features,
                "objective": objective,
                "task_type": task_type,
                "has_categorical_features": any(
                    ft and ft.startswith("c") for ft in (booster.feature_types or [])
                ),
            },
            source_fingerprint=fingerprint,
            checkpoint_contract=contract,
        )
        yield source


def _booster_objective(booster: Any) -> str:
    """Extract the objective name from the booster's learner config.

    The objective lives in the learner config (``booster.save_config()``),
    not in the string attributes — ``booster.attr("objective")`` is None
    unless the user explicitly set it with ``set_attr``.
    """
    try:
        config = json.loads(booster.save_config())
        return str(config.get("learner", {}).get("objective", {}).get("name", ""))
    except Exception:
        return ""


def _booster_num_classes(booster: Any, objective: str) -> int:
    """Return the classification width declared by an XGBoost booster."""
    if objective.startswith("binary:"):
        return 2
    try:
        config = json.loads(booster.save_config())
        raw = config["learner"]["learner_model_param"]["num_class"]
        return max(2, int(raw))
    except Exception:
        return 2


def _path_to_checkpoint(path: Path) -> Path:
    """Resolve a local path to a checkpoint directory."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {p}")
    return p
