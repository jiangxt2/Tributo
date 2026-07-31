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
        model_config_data, architecture_id = _read_model_config(ckpt_dir)

        # Check for full model checkpoint.
        model_pt_path = ckpt_dir / "model.pt"
        if model_pt_path.exists():
            loaded = torch.load(model_pt_path, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                # DNN trainers save ``model.state_dict()`` into model.pt —
                # a bare state_dict cannot be exported directly.  Reconstruct
                # the model skeleton when an architecture is registered,
                # otherwise reject explicitly instead of passing a raw dict
                # down to exporters.
                if architecture_id is None:
                    raise ValueError(
                        "model.pt contains a state_dict but model_config.json "
                        "is missing or has no architecture_id — cannot "
                        "reconstruct.  Provide a model_config.json with "
                        "architecture_id, or save a full nn.Module checkpoint."
                    )
                model = _reconstruct_model(architecture_id, model_config_data, loaded)
            else:
                model = loaded
        else:
            # Check for state_dict only.
            state_dict_path = ckpt_dir / "state_dict.pt"
            if state_dict_path.exists():
                state_dict = torch.load(
                    state_dict_path, map_location="cpu", weights_only=True
                )
                if architecture_id is None:
                    raise ValueError(
                        "Checkpoint contains state_dict but model_config.json is "
                        "missing or has no architecture_id.  Provide a "
                        "model_config.json with architecture_id to enable model "
                        "reconstruction, or use a full model checkpoint (model.pt)."
                    )
                model = _reconstruct_model(
                    architecture_id, model_config_data, state_dict
                )

        if model is None:
            raise FileNotFoundError(f"No model checkpoint found in {ckpt_dir}")

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


def _features_from_config(config: dict[str, Any]) -> list[Any]:
    """Build DNN feature columns from a ``model_config.json`` ``features`` list.

    Each entry is either a sparse column (``vocab_size``) or a dense
    column (``dimension``)::

        {"features": [
            {"name": "user_id", "vocab_size": 1000, "embedding_dim": 8},
            {"name": "age", "dimension": 1, "norm": "standard"},
        ], "dnn_hidden_units": [256, 128, 64], "dnn_dropout": 0.1}

    Raises ValueError when the feature list is missing or malformed.
    """
    from tributo.training.features.column_types import DenseFeat, SparseFeat

    raw = config.get("features")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "DNN reconstruction requires 'features' in model_config.json "
            "(list of sparse/dense column definitions)"
        )
    columns: list[Any] = []
    for entry in raw:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"Invalid feature column definition: {entry!r}")
        if "vocab_size" in entry:
            fields = SparseFeat.__dataclass_fields__
            columns.append(
                SparseFeat(**{k: v for k, v in entry.items() if k in fields})
            )
        else:
            fields = DenseFeat.__dataclass_fields__
            columns.append(DenseFeat(**{k: v for k, v in entry.items() if k in fields}))
    return columns


def _read_model_config(ckpt_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Read ``model_config.json`` and return ``(config, architecture_id)``."""
    config_path = ckpt_dir / "model_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return config, config.get("architecture_id")
    return {}, None


def _reconstruct_model(
    architecture_id: str,
    config: dict[str, Any],
    state_dict: dict[str, Any],
) -> Any:
    """Reconstruct a model from architecture_id + config, load state_dict.

    For known built-in architectures (DNN), construct directly.
    For third-party architectures, use ModelFactoryRegistry.

    Raises ValueError with a clear message when reconstruction fails —
    never returns the bare state_dict as a model.
    """

    if architecture_id == "dnn":
        from tributo.training.models.dnn import DNNModel

        model = DNNModel(
            features=_features_from_config(config),
            dnn_hidden_units=config.get("dnn_hidden_units"),
            dnn_dropout=config.get("dnn_dropout", 0.0),
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
    except Exception as exc:
        raise ValueError(
            f"Cannot reconstruct model for architecture_id {architecture_id!r}: "
            f"{exc}.  Register the architecture in ModelFactoryRegistry, or "
            "provide a full nn.Module checkpoint (model.pt)."
        ) from exc
