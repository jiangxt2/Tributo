"""Ray XGBoost checkpoint → ExportSource provider.

Resolves a Ray Train ``XGBoostCheckpoint`` (or a local checkpoint directory)
into an ``ExportSource`` that the export pipeline can consume.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Generator

from pydantic import BaseModel, ConfigDict

from tributo.exporting.models import ExportSource
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
                    raise FileNotFoundError(
                        f"No model file found in {checkpoint_dir}"
                    )
        else:
            # Ray XGBoostCheckpoint.
            xgb_checkpoint = XGBoostCheckpoint.from_directory(
                checkpoint.to_directory()
            )
            booster = xgb_checkpoint.get_model()

        # Compute source fingerprint.
        model_dump = booster.save_raw()
        fingerprint = hashlib.sha256(
            model_dump if isinstance(model_dump, bytes) else str(model_dump).encode()
        ).hexdigest()[:16]

        # Extract feature names.
        feature_names = booster.feature_names
        n_features = len(feature_names) if feature_names else 0
        feature_schema: dict[str, Any] = (
            {"feature_names": list(feature_names)} if feature_names else {}
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
            },
            source_fingerprint=fingerprint,
        )
        yield source


def _path_to_checkpoint(path: Path) -> Path:
    """Resolve a local path to a checkpoint directory."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {p}")
    return p
