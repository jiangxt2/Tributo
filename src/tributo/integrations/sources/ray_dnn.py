"""Ray DNN/PyTorch checkpoint → ExportSource provider.

Resolves a Ray Train TorchCheckpoint (or local checkpoint directory)
into an ExportSource that the export pipeline can consume.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Generator

from pydantic import BaseModel, ConfigDict

from tributo.exporting.models import ExportSource
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class _RayDnnSourceOptions(BaseModel):
    """Options for the Ray DNN source provider."""

    model_config = ConfigDict(extra="forbid")

    architecture_id: str | None = None
    model_kwargs: dict[str, Any] = {}


@PublicAPI(stability="beta")
class RayDnnSourceProvider:
    """Resolve a Ray DNN checkpoint to an ``ExportSource``.

    Loads ``state_dict`` and optionally reconstructs the model skeleton
    when ``architecture_id`` is provided via the model config.
    """

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "ray-dnn-v1"
    trainer_type: ClassVar[str] = "dnn"
    priority: ClassVar[int] = 100

    def open_source(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Any:
        """Open a Ray DNN checkpoint as an ``ExportSource``.

        Args:
            result: A Ray Train Result, checkpoint, or path string.
            config: Optional typed options.

        Yields:
            ExportSource with source_kind="dnn_result".
        """
        return self._open(result, config)

    @contextmanager
    def _open(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Generator[ExportSource, None, None]:
        import torch

        # Resolve checkpoint directory.
        if hasattr(result, "checkpoint"):
            checkpoint = result.checkpoint
        elif hasattr(result, "to_directory"):
            checkpoint = result
        elif isinstance(result, (str, Path)):
            checkpoint_dir = Path(result)
            checkpoint = checkpoint_dir
        else:
            raise TypeError(
                f"Expected Ray Result, Checkpoint, or path, got {type(result)}"
            )

        # Get checkpoint directory.
        if isinstance(checkpoint, Path):
            ckpt_dir = checkpoint
        else:
            ckpt_dir = Path(checkpoint.to_directory())

        # Try loading model.
        model = None
        architecture_id: str | None = None
        model_config_data: dict[str, Any] = {}

        # Check for full model checkpoint.
        model_pt_path = ckpt_dir / "model.pt"
        if model_pt_path.exists():
            model = torch.load(model_pt_path, map_location="cpu", weights_only=True)
        else:
            # Check for state_dict only.
            state_dict_path = ckpt_dir / "state_dict.pt"
            if state_dict_path.exists():
                state_dict = torch.load(
                    state_dict_path, map_location="cpu", weights_only=True
                )
                # Attempt model reconstruction from config.
                config_path = ckpt_dir / "model_config.json"
                if config_path.exists():
                    model_config_data = json.loads(config_path.read_text())
                    architecture_id = model_config_data.get("architecture_id")
                    if architecture_id is not None:
                        model = _reconstruct_model(
                            architecture_id, model_config_data, state_dict
                        )
                if model is None:
                    # Return state_dict as model_object.
                    model = state_dict

        if model is None:
            raise FileNotFoundError(
                f"No model checkpoint found in {ckpt_dir}"
            )

        # Extract metadata.
        metrics_path = ckpt_dir / "metrics.json"
        metadata: dict[str, Any] = {
            "framework": "pytorch",
            "framework_version": torch.__version__,
        }
        if metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text())
            metadata.update(metrics_data)

        # Compute fingerprint.
        state_bytes = b""
        if isinstance(model, dict):
            for k in sorted(model):
                state_bytes += str(k).encode()
        else:
            state_bytes = str(model).encode()
        fingerprint = hashlib.sha256(state_bytes).hexdigest()[:16]

        source = ExportSource(
            source_kind="dnn_result",
            model_object=model,
            architecture_id=architecture_id,
            model_config_data=model_config_data,
            feature_schema=model_config_data.get("feature_schema", {}),
            metadata=metadata,
            source_fingerprint=fingerprint,
        )
        yield source


def _reconstruct_model(
    architecture_id: str,
    config: dict[str, Any],
    state_dict: dict[str, Any],
) -> Any:
    """Reconstruct a model from architecture_id + config, load state_dict.

    For known built-in architectures (DNN), construct directly.
    For third-party architectures, use ModelFactoryRegistry.
    """
    import torch

    if architecture_id == "dnn":
        from tributo.training.models.dnn import DNN

        model = DNN(
            input_dim=config.get("input_dim", 1),
            hidden_dims=config.get("hidden_dims", [64, 32]),
            output_dim=config.get("output_dim", 1),
            dropout=config.get("dropout", 0.0),
        )
        model.load_state_dict(state_dict)
        return model

    # Third-party architectures: use ModelFactoryRegistry.
    from tributo.exporting.registries import ModelFactoryRegistry

    factory_registry = ModelFactoryRegistry()
    try:
        factory_cls = factory_registry.get(architecture_id)
        factory = factory_cls()
        model = factory.build(config)
        model.load_state_dict(state_dict)
        return model
    except Exception:
        logger.debug(
            "ModelFactory for %r not available — returning state_dict as-is",
            architecture_id,
        )
        return state_dict
