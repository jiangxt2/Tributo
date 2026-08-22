"""Ray Torch recipe checkpoint to the existing ExportSource contract."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Generator

from pydantic import BaseModel, ConfigDict, Field

from tributo.algorithms.api import QualifiedReference
from tributo.algorithms.core.worker import _load_reference, _validate_module_digest
from tributo.algorithms.spi import TorchTrainingRecipe
from tributo.exporting.models import ExportCheckpointV1, ExportSource
from tributo.training.checkpoint import checkpoint_directory
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class TorchRecipeSourceOptions(BaseModel):
    """Bind a checkpoint to the exact trusted recipe implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_ref: str = Field(min_length=1)
    recipe_code_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    implementation_id: str = Field(min_length=1)


@PublicAPI(stability="alpha")
class RayTorchRecipeSourceProvider:
    """Reconstruct a recipe model for the existing Torch ONNX exporter."""

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "ray-torch-recipe-v1"
    trainer_type: ClassVar[str] = "torch_recipe"
    priority: ClassVar[int] = 100

    def open_source(
        self,
        result: Any,
        config: BaseModel | None = None,
    ) -> Any:
        """Open one verified Ray Train recipe checkpoint."""
        options = TorchRecipeSourceOptions.model_validate(
            config.model_dump() if config is not None else {}
        )
        return _open_recipe_source(result, options)


@contextmanager
def _open_recipe_source(
    result: Any,
    options: TorchRecipeSourceOptions,
) -> Generator[ExportSource, None, None]:
    checkpoint = getattr(result, "checkpoint", result)
    if checkpoint is None:
        raise ValueError("Torch recipe training result has no checkpoint")
    with checkpoint_directory(checkpoint) as checkpoint_dir:
        yield _build_source(checkpoint_dir, options)


def _build_source(
    checkpoint_dir: Path,
    options: TorchRecipeSourceOptions,
) -> ExportSource:
    import torch

    root = checkpoint_dir.resolve()
    model_path = checkpoint_dir / "model.pt"
    config_path = checkpoint_dir / "model_config.json"
    for path in (model_path, config_path):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Torch recipe checkpoint artifact escapes its root")
        if not path.is_file():
            raise FileNotFoundError(f"Torch recipe checkpoint is missing {path.name!r}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    contract = ExportCheckpointV1.model_validate(payload)
    if contract.trainer_type != "torch_recipe":
        raise ValueError("checkpoint is not a Torch recipe export source")
    if payload.get("recipe_ref") != options.recipe_ref:
        raise ValueError("checkpoint recipe reference does not match the plan")
    if payload.get("recipe_code_digest") != options.recipe_code_digest:
        raise ValueError("checkpoint recipe code digest does not match the plan")
    if contract.architecture_id != options.implementation_id:
        raise ValueError("checkpoint implementation identity does not match the plan")

    reference = QualifiedReference.parse(options.recipe_ref)
    _validate_module_digest(reference, options.recipe_code_digest)
    recipe_cls = _load_reference(reference)
    if not isinstance(recipe_cls, type) or not issubclass(
        recipe_cls, TorchTrainingRecipe
    ):
        raise ValueError("recipe reference does not resolve to TorchTrainingRecipe")
    try:
        recipe = recipe_cls()
    except TypeError as exc:
        raise ValueError("Torch recipe must have a no-argument constructor") from exc
    model_config = payload.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("Torch recipe model config must be a JSON object")
    model = recipe.model_factory(model_config)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch recipe model_factory did not return nn.Module")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Torch recipe model.pt must contain one state_dict")
    model.load_state_dict(state)
    input_names = [field.name for field in contract.input_schema]
    model = _wrap_dense_columns(model, input_names)
    sample_inputs = {
        field.name: torch.zeros(
            tuple(2 if value == "batch" else int(value) for value in field.shape),
            dtype=torch.float32,
        )
        for field in contract.input_schema
    }
    metrics_path = checkpoint_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    return ExportSource(
        source_kind="torch_module",
        model_object=model,
        architecture_id=contract.architecture_id,
        model_config_data=model_config,
        feature_schema={
            "feature_names": input_names,
            "input_schema": [
                field.model_dump(mode="json") for field in contract.input_schema
            ],
        },
        preprocessing_state={},
        sample_inputs=sample_inputs,
        metadata={
            "framework": contract.framework,
            "framework_version": contract.framework_version,
            "task_type": contract.task_type,
            "trainer_type": contract.trainer_type,
            "metrics": metrics,
        },
        source_fingerprint=_sha256_file(model_path)[:16],
        checkpoint_contract=contract,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_dense_columns(model: Any, input_names: list[str]) -> Any:
    import torch

    class _DenseColumnModule(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *columns: Any) -> Any:
            if not columns:
                raise ValueError("Torch recipe export requires input columns")
            rows = columns[0].shape[0]
            features = torch.cat(
                [column.reshape(rows, -1).float() for column in columns],
                dim=1,
            )
            return self.wrapped(features)

    if not isinstance(model, torch.nn.Module) or not input_names:
        raise ValueError("Torch recipe export requires a model and named inputs")
    return _DenseColumnModule(model)


__all__ = ["RayTorchRecipeSourceProvider", "TorchRecipeSourceOptions"]
