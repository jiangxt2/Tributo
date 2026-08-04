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

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.models import ExportCheckpointV1, ExportSource
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class _RayDnnSourceOptions(BaseModel):
    """Options for the Ray DNN source provider."""

    model_config = ConfigDict(extra="forbid")

    architecture_id: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


@PublicAPI(stability="beta")
class RayDnnSourceProvider:
    """Resolve a Ray DNN checkpoint to an ``ExportSource``.

    Requires an ``ExportCheckpointV1`` model config and reconstructs the
    model skeleton declared by its ``architecture_id``.
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
        return _open_torch_source(
            result,
            config,
            source_kind="dnn_result",
            trainer_type="dnn",
        )


@contextmanager
def _open_torch_source(
    result: Any,
    config: BaseModel | None,
    *,
    source_kind: str,
    trainer_type: str,
) -> Generator[ExportSource, None, None]:
    """Open a DNN-family checkpoint after validating its export contract."""
    import torch

    opts = _RayDnnSourceOptions.model_validate(
        config.model_dump() if config is not None else {}
    )
    ckpt_dir = _resolve_checkpoint_dir(result)
    model_config_data, contract = _read_model_config(ckpt_dir, trainer_type)
    if (
        opts.architecture_id is not None
        and opts.architecture_id != contract.architecture_id
    ):
        raise ValueError(
            f"architecture_id override {opts.architecture_id!r} does not match "
            f"checkpoint contract {contract.architecture_id!r}"
        )

    model_pt_path = ckpt_dir / "model.pt"
    loaded = torch.load(model_pt_path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        model = _reconstruct_model(contract.architecture_id, model_config_data, loaded)
    else:
        model = loaded
    if contract.architecture_id == "dnn":
        model = _wrap_named_inputs(model, [f.name for f in contract.input_schema])

    preprocessing_artifact = contract.preprocessing.get("artifact")
    if not isinstance(preprocessing_artifact, str) or not preprocessing_artifact:
        raise ValueError(
            "ExportCheckpointV1.preprocessing must name a required artifact"
        )
    if preprocessing_artifact not in contract.required_artifacts:
        raise ValueError(
            "ExportCheckpointV1.preprocessing artifact must be listed in "
            "required_artifacts"
        )
    preprocessing_state = json.loads((ckpt_dir / preprocessing_artifact).read_text())

    metrics_path = ckpt_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metadata: dict[str, Any] = {
        "framework": contract.framework,
        "framework_version": contract.framework_version,
        "task_type": contract.task_type,
        "trainer_type": contract.trainer_type,
        "checkpoint_format_version": contract.checkpoint_format_version,
        "metrics": metrics,
    }

    feature_names = [field.name for field in contract.input_schema]
    feature_schema = {
        "feature_names": feature_names,
        "input_schema": [
            field.model_dump(mode="json") for field in contract.input_schema
        ],
    }
    sample_inputs = {
        field.name: torch.zeros(
            _sample_shape(field.shape), dtype=_torch_dtype(field.dtype)
        )
        for field in contract.input_schema
    }
    state_bytes = model_pt_path.read_bytes()
    fingerprint = hashlib.sha256(state_bytes).hexdigest()[:16]

    yield ExportSource(
        source_kind=source_kind,
        model_object=model,
        architecture_id=contract.architecture_id,
        model_config_data=model_config_data,
        feature_schema=feature_schema,
        preprocessing_state=preprocessing_state,
        sample_inputs=sample_inputs,
        metadata=metadata,
        source_fingerprint=fingerprint,
        checkpoint_contract=contract,
    )


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


def _resolve_checkpoint_dir(result: Any) -> Path:
    """Resolve a Ray result/checkpoint or local path to a directory."""
    if hasattr(result, "checkpoint"):
        checkpoint = result.checkpoint
    elif hasattr(result, "to_directory"):
        checkpoint = result
    elif isinstance(result, (str, Path)):
        checkpoint = Path(result)
    else:
        raise TypeError(f"Expected Ray Result, Checkpoint, or path, got {type(result)}")

    if checkpoint is None:
        raise ValueError("Training result has no checkpoint")
    if isinstance(checkpoint, (str, Path)):
        checkpoint_dir = Path(checkpoint)
    else:
        checkpoint_dir = Path(checkpoint.to_directory())
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Checkpoint path is not a directory: {checkpoint_dir}"
        )
    return checkpoint_dir


def _read_model_config(
    ckpt_dir: Path, trainer_type: str
) -> tuple[dict[str, Any], ExportCheckpointV1]:
    """Read and validate the required ``ExportCheckpointV1`` envelope."""
    config_path = ckpt_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Required checkpoint artifact 'model_config.json' is missing"
        )
    config = json.loads(config_path.read_text())
    try:
        contract = ExportCheckpointV1.model_validate(config)
    except Exception as exc:
        raise ValueError(
            f"Invalid ExportCheckpointV1 metadata in {config_path}: {exc}"
        ) from exc
    if contract.trainer_type != trainer_type:
        raise ValueError(
            f"Checkpoint trainer_type {contract.trainer_type!r} does not match "
            f"provider trainer_type {trainer_type!r}"
        )

    required_artifacts = set(contract.required_artifacts)
    required_artifacts.add("model_config.json")
    required_artifacts.add("model.pt")
    root = ckpt_dir.resolve()
    for artifact in required_artifacts:
        artifact_path = (ckpt_dir / artifact).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError(
                f"Required checkpoint artifact escapes checkpoint root: {artifact!r}"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Required checkpoint artifact {artifact!r} is missing"
            )
    return config, contract


def _sample_shape(shape: tuple[int | str, ...]) -> tuple[int, ...]:
    """Replace dynamic batch axes with two rows for exporter sample inputs."""
    return tuple(2 if isinstance(dim, str) else dim for dim in shape) or (2,)


def _torch_dtype(dtype: str) -> Any:
    """Map a checkpoint dtype to a Torch dtype used for sample inputs."""
    import torch

    try:
        return {
            "int64": torch.int64,
            "int32": torch.int32,
            "float64": torch.float64,
            "float32": torch.float32,
        }[dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported checkpoint dtype {dtype!r}; expected one of "
            "int64, int32, float64, float32"
        ) from exc


def _wrap_named_inputs(model: Any, input_names: list[str]) -> Any:
    """Adapt DNN dict inputs to the positional ONNX exporter protocol."""
    import torch

    if not isinstance(model, torch.nn.Module):
        return model

    class _NamedInputModule(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *inputs: Any) -> Any:
            return self.wrapped(dict(zip(input_names, inputs, strict=True)))

    return _NamedInputModule(model)


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

    # Third-party architectures: use the entry-point-loaded
    # ModelFactoryRegistry (never an empty one).
    from tributo.exporting.registries import build_factory_registry

    factory_registry = build_factory_registry()
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
