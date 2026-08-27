"""Lower narrow PyTorch recipes onto Ray Train's collective runtime."""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tributo._common.config import StrictConfigModel
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionResult,
    MetricReduction,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.core.worker import _load_reference, _validate_module_digest
from tributo.algorithms.spi import (
    CollectiveAlgorithm,
    MetricPlan,
    OptimizationPlan,
    TorchTrainingRecipe,
    TrainingRecipeV2,
    TrainingStepResult,
)
from tributo.util.annotations import DeveloperAPI

_TRAINER_TYPE = "torch_recipe"


@runtime_checkable
class _FirstPartyTorchRecipeAdapter(Protocol):
    """Internal typed contract for first-party domain-specific Recipe hooks."""

    _trainer_type: str

    def _bind_datasets(
        self,
        datasets: Mapping[str, object],
        *,
        config: Mapping[str, Any],
        worker_count: int,
        resume_from: str | None,
    ) -> Mapping[str, object]: ...

    def _lower_worker_config(
        self,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def _prepare_batch(
        self,
        batch: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str,
        weight_name: str | None,
        config: Mapping[str, Any],
    ) -> tuple[object, object, object | None, int]: ...

    def _checkpoint_contract(
        self,
        *,
        config: Mapping[str, Any],
        feature_count: int,
        output_shape: tuple[int, ...],
        framework_version: str,
        model_digest: str,
        world_size: int,
    ) -> dict[str, Any]: ...

    def _write_checkpoint_artifacts(self, checkpoint_dir: Path) -> tuple[str, ...]: ...

    def _validate_checkpoint_artifacts(self, checkpoint_dir: Path) -> None: ...


class _LoopConfig(StrictConfigModel):
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=256, ge=1)
    prefetch_batches: int = Field(default=1, ge=0)
    local_shuffle_buffer_size: int | None = Field(default=None, ge=1)
    seed: int = 42
    amp: bool = False
    early_stopping_patience: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_shuffle_buffer(self) -> _LoopConfig:
        """Keep Ray's local shuffle buffer at least one full batch."""
        if (
            self.local_shuffle_buffer_size is not None
            and self.local_shuffle_buffer_size < self.batch_size
        ):
            raise ValueError(
                "local_shuffle_buffer_size must be at least training.batch_size"
            )
        return self


class _OutputConfig(StrictConfigModel):
    bundle_uri: str = Field(min_length=1)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmConfigurationError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise AlgorithmConfigurationError(f"{name} keys must be strings")
    return value


def _recipe_type(
    reference: object, code_digest: str | None
) -> type[TorchTrainingRecipe] | type[TrainingRecipeV2]:
    from tributo.algorithms.api import QualifiedReference

    if not isinstance(reference, QualifiedReference):
        raise AlgorithmConfigurationError("recipe reference is invalid")
    _validate_module_digest(reference, code_digest)
    implementation = _load_reference(reference)
    if not isinstance(implementation, type) or not issubclass(
        implementation, (TorchTrainingRecipe, TrainingRecipeV2)
    ):
        raise AlgorithmConfigurationError(
            "Torch recipe implementation must subclass TorchTrainingRecipe or "
            "TrainingRecipeV2"
        )
    expected_version = 2 if issubclass(implementation, TrainingRecipeV2) else 1
    if getattr(implementation, "api_version", None) != expected_version:
        raise AlgorithmConfigurationError(
            f"Torch recipe api_version must be {expected_version}"
        )
    return implementation


def _new_recipe(
    reference: object,
    code_digest: str | None,
) -> TorchTrainingRecipe | TrainingRecipeV2:
    recipe_cls = _recipe_type(reference, code_digest)
    try:
        return recipe_cls()
    except TypeError as exc:
        raise AlgorithmConfigurationError(
            "Torch recipe classes must have a no-argument constructor"
        ) from exc


class _TorchRecipeCollectiveAlgorithm(CollectiveAlgorithm):
    """Internal adapter that owns infrastructure around one user recipe."""

    def __init__(
        self,
        plan: ResolvedAlgorithmPlan,
        recipe: TorchTrainingRecipe | TrainingRecipeV2,
    ) -> None:
        self._plan = plan
        self._recipe = recipe
        ray_config = _mapping(plan.algorithm_config.get("ray", {}), "ray config")
        max_failures = ray_config.get("max_failures", 0)
        if (
            not isinstance(max_failures, int)
            or isinstance(max_failures, bool)
            or max_failures != 0
        ):
            raise AlgorithmConfigurationError(
                "Torch recipes require ray.max_failures=0 until the independent "
                "failure-recovery gate passes"
            )
        resume = _mapping(ray_config.get("resume", {}), "ray.resume config")
        configured_checkpoint = resume.get("checkpoint_path")
        if configured_checkpoint is not None and (
            configured_checkpoint != plan.runtime.resume_from
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.resume.checkpoint_path must match "
                "ExecutionRequest.resume_from"
            )

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        """Accept pre-bound train/val/test datasets without performing a split."""
        if isinstance(self._recipe, _FirstPartyTorchRecipeAdapter):
            bound = self._recipe._bind_datasets(
                datasets,
                config=self._plan.algorithm_config,
                worker_count=self._plan.runtime.worker_count,
                resume_from=self._plan.runtime.resume_from,
            )
            if not isinstance(bound, Mapping) or not bound:
                raise AlgorithmConfigurationError(
                    "Torch recipe dataset adapter must return named Datasets"
                )
            return dict(bound)
        if "train" in datasets:
            unknown = sorted(set(datasets) - {"train", "val", "test"})
            if unknown:
                raise AlgorithmConfigurationError(
                    f"Torch recipe received unknown dataset role(s): {unknown}"
                )
            return dict(datasets)
        if len(datasets) != 1:
            raise AlgorithmConfigurationError(
                "Torch recipe input must bind one train Dataset or explicit "
                "train/val/test Dataset roles"
            )
        return {"train": next(iter(datasets.values()))}

    def build_model(self, config: Mapping[str, Any]) -> object:
        """Build a model through the user recipe."""
        if isinstance(self._recipe, TrainingRecipeV2):
            modules = self._recipe.build_modules(config)
            if not isinstance(modules, Mapping) or "model" not in modules:
                raise AlgorithmConfigurationError(
                    "TrainingRecipeV2 build_modules must provide model"
                )
            return modules["model"]
        return self._recipe.model_factory(_mapping(config.get("model", {}), "model"))

    def build_optimizer(self, model: object, config: Mapping[str, Any]) -> object:
        """Build an optimizer through the user recipe."""
        if isinstance(self._recipe, TrainingRecipeV2):
            plan = self._recipe.optimization_plan(
                model,
                _mapping(config.get("optimizer", {}), "optimizer"),
            )
            if not isinstance(plan, OptimizationPlan):
                raise AlgorithmConfigurationError(
                    "TrainingRecipeV2 optimization_plan must return OptimizationPlan"
                )
            return plan.optimizer
        return self._recipe.optimizer_factory(
            model,
            _mapping(config.get("optimizer", {}), "optimizer"),
        )

    def build_loss(self, config: Mapping[str, Any]) -> object:
        """Build a loss through the user recipe."""
        if isinstance(self._recipe, TrainingRecipeV2):
            modules = self._recipe.build_modules(config)
            if not isinstance(modules, Mapping) or "loss" not in modules:
                raise AlgorithmConfigurationError(
                    "TrainingRecipeV2 build_modules must provide loss"
                )
            return modules["loss"]
        return self._recipe.loss_factory(_mapping(config.get("loss", {}), "loss"))

    def checkpoint_state(self, model: object, optimizer: object) -> Mapping[str, Any]:
        """Return bounded replicated state for conformance tooling."""
        from tributo.training.distributed_torch import unwrapped_model

        model_state = getattr(unwrapped_model(model), "state_dict", None)
        optimizer_state = getattr(optimizer, "state_dict", None)
        if not callable(model_state) or not callable(optimizer_state):
            raise AlgorithmConfigurationError(
                "Torch recipe model and optimizer must expose state_dict"
            )
        return {"model": model_state(), "optimizer": optimizer_state()}

    def train_loop_per_worker(self, config: Mapping[str, Any]) -> None:
        """Run the framework-owned loop with the recipe's four factories."""
        worker_config = dict(config)
        if isinstance(self._recipe, _FirstPartyTorchRecipeAdapter):
            lowered = self._recipe._lower_worker_config(worker_config)
            if not isinstance(lowered, Mapping):
                raise AlgorithmConfigurationError(
                    "Torch recipe config adapter must return a mapping"
                )
            worker_config = dict(lowered)
        worker_config["_tributo_recipe_ref"] = str(
            self._plan.implementation.implementation_ref
        )
        worker_config["_tributo_recipe_code_digest"] = (
            self._plan.implementation.code_digest
        )
        worker_config["_tributo_implementation_id"] = (
            self._plan.implementation.implementation_id
        )
        worker_config["_tributo_algorithm"] = self._plan.resolution.algorithm
        worker_config["_tributo_input_binding_digest"] = (
            self._plan.primary_input_descriptor.binding_digest
        )
        worker_config["_tributo_distribution_spec_digest"] = (
            self._plan.runtime.distribution_digest
        )
        worker_config["_tributo_resume_from"] = self._plan.runtime.resume_from
        worker_config["_tributo_feature_names"] = list(
            self._plan.primary_input_binding.feature_names
        )
        worker_config["_tributo_label_name"] = (
            self._plan.primary_input_binding.label_name
        )
        worker_config["_tributo_weight_name"] = (
            self._plan.primary_input_binding.sample_weight_name
        )
        torch_recipe_train_loop_per_worker(worker_config, self._recipe)


def _batch_rows(value: object) -> int:
    length = getattr(value, "__len__", None)
    if not callable(length):
        raise AlgorithmConfigurationError(
            "Torch recipe batch values must expose a batch dimension"
        )
    try:
        return int(length())
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(
            "Torch recipe batch dimension must be an integer"
        ) from exc


def _dense_batch(
    batch: object,
    *,
    feature_names: tuple[str, ...],
    label_name: str,
    weight_name: str | None,
) -> tuple[Any, Any, Any | None, int]:
    import torch

    if not isinstance(batch, Mapping):
        raise AlgorithmConfigurationError("Ray Data Torch batches must be mappings")
    required = (*feature_names, label_name, *((weight_name,) if weight_name else ()))
    missing = [name for name in required if name not in batch]
    if missing:
        raise AlgorithmConfigurationError(
            f"Torch recipe batch is missing required column(s): {missing}"
        )
    labels = batch[label_name]
    if not isinstance(labels, torch.Tensor):
        raise AlgorithmConfigurationError("Ray Data label batches must be tensors")
    rows = _batch_rows(labels)
    columns: list[Any] = []
    for name in feature_names:
        value = batch[name]
        if not isinstance(value, torch.Tensor) or _batch_rows(value) != rows:
            raise AlgorithmConfigurationError(
                f"feature {name!r} must be a Tensor with the shared batch dimension"
            )
        column = value.reshape(rows, -1).to(dtype=torch.float32)
        if column.shape[1] != 1:
            raise AlgorithmConfigurationError(
                "the first Torch recipe input profile requires scalar feature columns"
            )
        columns.append(column)
    features = torch.cat(columns, dim=1)
    targets = labels.to(dtype=torch.float32)
    weights = None
    if weight_name is not None:
        weights = batch[weight_name]
        if not isinstance(weights, torch.Tensor) or _batch_rows(weights) != rows:
            raise AlgorithmConfigurationError(
                "sample weights must be a Tensor with the shared batch dimension"
            )
        weights = weights.to(dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise AlgorithmConfigurationError(
                "sample weights must contain only finite non-negative values"
            )
    return features, targets, weights, rows


def _prepare_batch(
    recipe: TorchTrainingRecipe | TrainingRecipeV2,
    batch: object,
    *,
    feature_names: tuple[str, ...],
    label_name: str | None,
    weight_name: str | None,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any | None, int]:
    if isinstance(recipe, TrainingRecipeV2):
        prepared = recipe.batch_adapter(
            batch,
            feature_names=feature_names,
            label_name=label_name,
            weight_name=weight_name,
            config=config,
        )
        if not isinstance(prepared, tuple) or len(prepared) != 4:
            raise AlgorithmConfigurationError(
                "TrainingRecipeV2 batch_adapter must return four values"
            )
        features, targets, weights, rows = prepared
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise AlgorithmConfigurationError(
                "TrainingRecipeV2 batch_adapter returned an invalid row count"
            )
        return features, targets, weights, rows
    if label_name is None:
        raise AlgorithmConfigurationError(
            "TorchTrainingRecipe requires one label column"
        )
    if not isinstance(recipe, _FirstPartyTorchRecipeAdapter):
        return _dense_batch(
            batch,
            feature_names=feature_names,
            label_name=label_name,
            weight_name=weight_name,
        )
    prepared = recipe._prepare_batch(
        batch,
        feature_names=feature_names,
        label_name=label_name,
        weight_name=weight_name,
        config=config,
    )
    if not isinstance(prepared, tuple) or len(prepared) != 4:
        raise AlgorithmConfigurationError(
            "Torch recipe batch adapter must return features, targets, weights, rows"
        )
    features, targets, weights, rows = prepared
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise AlgorithmConfigurationError(
            "Torch recipe batch adapter returned an invalid row count"
        )
    return features, targets, weights, rows


def _empty_batch(
    template: tuple[Any, Any, Any | None, int],
) -> tuple[Any, Any, Any | None, int]:
    features, targets, weights, _ = template
    empty_features = (
        {name: value[:0] for name, value in features.items()}
        if isinstance(features, Mapping)
        else features[:0]
    )
    return (
        empty_features,
        targets[:0],
        weights[:0] if weights is not None else None,
        0,
    )


def _evaluation_metric_name(split: str, name: str) -> str:
    if name.startswith("train_"):
        return f"{split}_{name.removeprefix('train_')}"
    return f"{split}_{name}"


def _metric_update(
    state: dict[str, float],
    value: object,
    *,
    reduction: MetricReduction,
    rows: int,
    weights: Any | None,
) -> None:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().to(dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise AlgorithmConfigurationError(
            "Torch recipe metric produced empty or non-finite values"
        )
    if reduction is MetricReduction.SUM_COUNT:
        if tensor.numel() == 1:
            state["value"] += float(tensor.item()) * rows
            state["weight"] += rows
        else:
            state["value"] += float(tensor.sum().item())
            state["weight"] += int(tensor.numel())
        return
    if reduction is MetricReduction.WEIGHTED_MEAN:
        if weights is None:
            raise AlgorithmConfigurationError(
                "weighted_mean metric requires an InputBinding sample_weight_name"
            )
        metric_values = tensor
        if metric_values.numel() == 1:
            metric_values = metric_values.expand_as(weights)
        if metric_values.numel() != weights.numel():
            raise AlgorithmConfigurationError(
                "weighted metric values must match the sample-weight tensor"
            )
        state["value"] += float((metric_values * weights).sum().item())
        state["weight"] += float(weights.sum().item())
        return
    candidate = float(
        (tensor.min() if reduction is MetricReduction.MIN else tensor.max()).item()
    )
    if reduction is MetricReduction.MIN:
        state["value"] = min(state["value"], candidate)
    else:
        state["value"] = max(state["value"], candidate)


def _reduce_metrics(
    states: Mapping[str, dict[str, float]],
    reducers: Mapping[str, MetricReduction],
) -> dict[str, float]:
    from tributo.training.distributed_torch import (
        all_gather_objects,
        all_reduce_values,
    )

    reduced: dict[str, float] = {}
    for name in sorted(reducers):
        reduction = reducers[name]
        state = states[name]
        if reduction in {MetricReduction.SUM_COUNT, MetricReduction.WEIGHTED_MEAN}:
            value, weight = all_reduce_values((state["value"], state["weight"]))
            if weight <= 0:
                raise AlgorithmConfigurationError(
                    f"metric {name!r} has a non-positive global weight"
                )
            reduced[name] = value / weight
        else:
            candidates = tuple(
                float(value) for value in all_gather_objects(state["value"])
            )
            finite = [value for value in candidates if math.isfinite(value)]
            if not finite:
                raise AlgorithmConfigurationError(
                    f"metric {name!r} has no finite global value"
                )
            reduced[name] = (
                min(finite) if reduction is MetricReduction.MIN else max(finite)
            )
    return reduced


def _checkpoint_contract(
    *,
    config: Mapping[str, Any],
    feature_count: int,
    output_shape: tuple[int, ...],
    framework_version: str,
) -> dict[str, Any]:
    from tributo.exporting.models import CheckpointField, ExportCheckpointV1

    feature_names = tuple(config.get("_tributo_feature_names") or ())
    if len(feature_names) != feature_count or any(
        not isinstance(name, str) or not name for name in feature_names
    ):
        raise AlgorithmConfigurationError(
            "Torch recipe checkpoint feature declaration is invalid"
        )
    contract = ExportCheckpointV1(
        trainer_type=_TRAINER_TYPE,
        architecture_id=str(config["_tributo_implementation_id"]),
        input_schema=tuple(
            CheckpointField(
                name=name,
                dtype="float32",
                shape=("batch",),
            )
            for name in feature_names
        ),
        output_schema=(
            CheckpointField(
                name="output",
                dtype="float32",
                shape=("batch", *output_shape),
            ),
        ),
        task_type=str(config["_tributo_algorithm"]),
        framework="pytorch",
        framework_version=framework_version,
        required_artifacts=("model.pt",),
    ).model_dump(mode="json")
    contract.update(
        {
            "model": dict(_mapping(config.get("model", {}), "model")),
            "recipe_ref": str(config["_tributo_recipe_ref"]),
            "recipe_code_digest": config.get("_tributo_recipe_code_digest"),
        }
    )
    return contract


def _recipe_checkpoint_contract(
    recipe: TorchTrainingRecipe | TrainingRecipeV2,
    *,
    config: Mapping[str, Any],
    feature_count: int,
    output_shape: tuple[int, ...],
    framework_version: str,
    model_digest: str,
    world_size: int,
) -> dict[str, Any]:
    if not isinstance(recipe, _FirstPartyTorchRecipeAdapter):
        return _checkpoint_contract(
            config=config,
            feature_count=feature_count,
            output_shape=output_shape,
            framework_version=framework_version,
        )
    contract = recipe._checkpoint_contract(
        config=config,
        feature_count=feature_count,
        output_shape=output_shape,
        framework_version=framework_version,
        model_digest=model_digest,
        world_size=world_size,
    )
    if not isinstance(contract, dict):
        raise AlgorithmConfigurationError(
            "Torch recipe checkpoint adapter must return a dictionary"
        )
    return contract


def _write_recipe_checkpoint_artifacts(
    recipe: TorchTrainingRecipe | TrainingRecipeV2,
    checkpoint_dir: Path,
) -> tuple[str, ...]:
    if not isinstance(recipe, _FirstPartyTorchRecipeAdapter):
        return ()
    files = recipe._write_checkpoint_artifacts(checkpoint_dir)
    if not isinstance(files, tuple) or any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not (checkpoint_dir / name).is_file()
        for name in files
    ):
        raise AlgorithmConfigurationError(
            "Torch recipe checkpoint adapter returned invalid artifact files"
        )
    return files


def _validate_resume(
    metadata: Mapping[str, Any],
    *,
    world_size: int,
    distribution_digest: str | None,
) -> None:
    if metadata.get("world_size") != world_size:
        raise AlgorithmConfigurationError(
            "Torch recipe resume requires the original world size"
        )
    if metadata.get("distribution_spec_digest") != distribution_digest:
        raise AlgorithmConfigurationError(
            "Torch recipe resume DistributionSpec digest does not match"
        )


def _evaluate_dataset(
    data: object,
    *,
    split: str,
    recipe: TorchTrainingRecipe | TrainingRecipeV2,
    model: object,
    loss_fn: object,
    metrics: Mapping[str, Any],
    reducers: Mapping[str, MetricReduction],
    loop: _LoopConfig,
    feature_names: tuple[str, ...],
    label_name: str | None,
    weight_name: str | None,
    modules: Mapping[str, object],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], int]:
    import torch

    from tributo.training.distributed_torch import all_reduce_values

    iterate = getattr(data, "iter_torch_batches", None)
    if not callable(iterate):
        raise AlgorithmConfigurationError(
            f"Torch recipe {split} input is not a Ray Data iterator"
        )
    iterator = iter(
        iterate(
            batch_size=loop.batch_size,
            prefetch_batches=loop.prefetch_batches,
            dtypes=torch.float32,
            drop_last=False,
        )
    )
    states: dict[str, dict[str, float]] = {
        name: {
            "value": (
                float("inf")
                if reduction is MetricReduction.MIN
                else float("-inf")
                if reduction is MetricReduction.MAX
                else 0.0
            ),
            "weight": 0.0,
        }
        for name, reduction in reducers.items()
    }
    local_rows = 0
    unwrapped = getattr(model, "module", model)
    train = getattr(unwrapped, "train", None)
    if not callable(train):
        raise AlgorithmConfigurationError("Torch recipe model must expose train()")
    train(False)
    current = next(iterator, None)
    while True:
        rows = 0
        prepared = None
        if current is not None:
            prepared = _prepare_batch(
                recipe,
                current,
                feature_names=feature_names,
                label_name=label_name,
                weight_name=weight_name,
                config=config,
            )
            rows = prepared[3]
        (global_rows,) = all_reduce_values((float(rows),))
        if global_rows <= 0:
            break
        if prepared is not None:
            features, targets, weights, rows = prepared
            with torch.no_grad():
                if isinstance(recipe, TrainingRecipeV2):
                    step = recipe.validation_step(
                        {**modules, "model": unwrapped},
                        features,
                        targets,
                        weights,
                        config,
                    )
                    if not isinstance(step, TrainingStepResult):
                        raise AlgorithmConfigurationError(
                            "TrainingRecipeV2 validation_step must return "
                            "TrainingStepResult"
                        )
                    predictions = step.predictions
                    loss = step.loss
                else:
                    predictions = recipe.forward(unwrapped, features)
                if not isinstance(predictions, torch.Tensor):
                    raise AlgorithmConfigurationError(
                        "Torch recipe forward must return one Tensor"
                    )
                aligned_targets = targets
                if predictions.numel() == targets.numel():
                    aligned_targets = targets.reshape_as(predictions)
                if not isinstance(recipe, TrainingRecipeV2):
                    loss = recipe.compute_loss(loss_fn, predictions, aligned_targets)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise AlgorithmConfigurationError(
                    "Torch recipe evaluation loss must return one scalar batch mean"
                )
            _metric_update(
                states["train_loss"],
                loss,
                reduction=MetricReduction.SUM_COUNT,
                rows=rows,
                weights=None,
            )
            for name, metric in metrics.items():
                _metric_update(
                    states[name],
                    metric(predictions, aligned_targets),
                    reduction=reducers[name],
                    rows=rows,
                    weights=weights,
                )
            local_rows += rows
        current = next(iterator, None)
    reduced = _reduce_metrics(states, reducers)
    return {
        f"{split}_loss": reduced["train_loss"],
        **{
            _evaluation_metric_name(split, name): value
            for name, value in reduced.items()
            if name != "train_loss"
        },
    }, local_rows


@DeveloperAPI
def torch_recipe_train_loop_per_worker(
    config: Mapping[str, Any],
    recipe: TorchTrainingRecipe | TrainingRecipeV2,
) -> None:
    """Execute one dense-tabular recipe with Ray Data and replicated DDP."""
    import numpy as np
    import ray.train
    import torch
    import torch.distributed as dist
    from ray.train.torch import enable_reproducibility

    from tributo.training.checkpoint import (
        ResumeConfig,
        capture_rng_state,
        checkpoint_directory,
        read_resume_manifest,
        restore_rng_state,
        write_resume_manifest,
    )
    from tributo.training.distributed_torch import (
        all_gather_objects,
        all_reduce_values,
        broadcast_bool,
        collective_execution_evidence,
        prepare_model,
        unwrapped_model,
    )

    loop = _LoopConfig.model_validate(config.get("training") or {})
    ray_config = _mapping(config.get("ray", {}), "ray config")
    resume = ResumeConfig.model_validate(ray_config.get("resume") or {})
    feature_names = tuple(config.get("_tributo_feature_names") or ())
    label_name = config.get("_tributo_label_name")
    weight_name = config.get("_tributo_weight_name")
    if not feature_names:
        raise AlgorithmConfigurationError("Torch recipe requires feature columns")
    if not isinstance(recipe, TrainingRecipeV2) and (
        not isinstance(label_name, str) or not label_name
    ):
        raise AlgorithmConfigurationError(
            "TorchTrainingRecipe requires one label column"
        )
    if label_name is not None and (not isinstance(label_name, str) or not label_name):
        raise AlgorithmConfigurationError(
            "TrainingRecipeV2 label column must be non-empty when provided"
        )
    if weight_name is not None and not isinstance(weight_name, str):
        raise AlgorithmConfigurationError("sample weight column must be a string")

    context = ray.train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    trainer_type = (
        recipe._trainer_type
        if isinstance(recipe, _FirstPartyTorchRecipeAdapter)
        else _TRAINER_TYPE
    )
    enable_reproducibility(loop.seed)
    random.seed(loop.seed + rank)
    np.random.seed(loop.seed + rank)

    modules: dict[str, object]
    optimization_plan: OptimizationPlan | None = None
    if isinstance(recipe, TrainingRecipeV2):
        built_modules = recipe.build_modules(config)
        if (
            not isinstance(built_modules, Mapping)
            or "model" not in built_modules
            or "loss" not in built_modules
        ):
            raise AlgorithmConfigurationError(
                "TrainingRecipeV2 build_modules must provide model and loss"
            )
        modules = dict(built_modules)
        model: Any = modules["model"]
        loss_fn = modules["loss"]
        optimization_plan = recipe.optimization_plan(
            model,
            _mapping(config.get("optimizer", {}), "optimizer"),
        )
        if not isinstance(optimization_plan, OptimizationPlan):
            raise AlgorithmConfigurationError(
                "TrainingRecipeV2 optimization_plan must return OptimizationPlan"
            )
        optimizer: Any = optimization_plan.optimizer
        metric_plan = recipe.metric_plan(_mapping(config.get("metrics", {}), "metrics"))
        if not isinstance(metric_plan, MetricPlan):
            raise AlgorithmConfigurationError(
                "TrainingRecipeV2 metric_plan must return MetricPlan"
            )
        metrics = metric_plan.factories
    else:
        model = recipe.model_factory(_mapping(config.get("model", {}), "model"))
        loss_fn = recipe.loss_factory(_mapping(config.get("loss", {}), "loss"))
        optimizer = recipe.optimizer_factory(
            model,
            _mapping(config.get("optimizer", {}), "optimizer"),
        )
        metrics = recipe.metric_factories(
            _mapping(config.get("metrics", {}), "metrics")
        )
        modules = {"model": model, "loss": loss_fn}
    if not isinstance(model, torch.nn.Module):
        raise AlgorithmConfigurationError(
            "Torch recipe model_factory must return torch.nn.Module"
        )
    if world_size > 1 and any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    ):
        raise AlgorithmConfigurationError(
            "replicated Torch recipes reject BatchNorm until synchronized buffers "
            "have a separate gate"
        )
    if not callable(loss_fn):
        raise AlgorithmConfigurationError(
            "Torch recipe loss_factory must return a callable"
        )
    for method in ("zero_grad", "step", "state_dict", "load_state_dict"):
        if not callable(getattr(optimizer, method, None)):
            raise AlgorithmConfigurationError(
                "Torch recipe optimizer does not implement the optimizer contract"
            )
    if not isinstance(metrics, Mapping) or any(
        not isinstance(name, str) or not name or not callable(metric)
        for name, metric in metrics.items()
    ):
        raise AlgorithmConfigurationError(
            "Torch recipe metric_factories must return named callables"
        )
    reducer_payload = _mapping(
        config.get("_tributo_metric_reducers", {}),
        "metric reducers",
    )
    try:
        reducers = {
            name: MetricReduction(value) for name, value in reducer_payload.items()
        }
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError("Torch metric reducer is invalid") from exc
    expected_metric_names = set(reducers) - {"train_loss"}
    if set(metrics) != expected_metric_names:
        raise AlgorithmConfigurationError(
            "Torch recipe metric names must exactly match CollectivePolicy reducers"
        )
    scheduler: Any = (
        optimization_plan.scheduler if optimization_plan is not None else None
    )
    if scheduler is not None and any(
        not callable(getattr(scheduler, method, None))
        for method in ("step", "state_dict", "load_state_dict")
    ):
        raise AlgorithmConfigurationError(
            "TrainingRecipeV2 scheduler does not implement the scheduler contract"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=loop.amp)

    checkpoint = ray.train.get_checkpoint()
    if checkpoint is None:
        explicit_checkpoint = config.get("_tributo_resume_from")
        if explicit_checkpoint is not None and not isinstance(explicit_checkpoint, str):
            raise AlgorithmConfigurationError(
                "Torch recipe explicit checkpoint path must be a string"
            )
        checkpoint = explicit_checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if checkpoint is not None:
        with checkpoint_directory(checkpoint) as checkpoint_dir:
            envelope = read_resume_manifest(
                checkpoint_dir,
                expected_trainer_type=trainer_type,
                expected_resume_id=resume.resume_id,
            )
            _validate_resume(
                envelope.payload_metadata,
                world_size=world_size,
                distribution_digest=config.get("_tributo_distribution_spec_digest"),
            )
            model.load_state_dict(
                torch.load(
                    checkpoint_dir / "model.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            optimizer.load_state_dict(
                torch.load(
                    checkpoint_dir / "optimizer.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            scheduler_path = checkpoint_dir / "scheduler.pt"
            if scheduler is not None:
                if not scheduler_path.is_file():
                    raise AlgorithmConfigurationError(
                        "TrainingRecipeV2 checkpoint is missing scheduler state"
                    )
                scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location="cpu", weights_only=True)
                )
            scaler_path = checkpoint_dir / "scaler.pt"
            if scaler_path.is_file():
                scaler.load_state_dict(
                    torch.load(scaler_path, map_location="cpu", weights_only=True)
                )
            rng = json.loads((checkpoint_dir / "rng_state.json").read_text())
            training_state_path = checkpoint_dir / "training_state.json"
            training_state = (
                json.loads(training_state_path.read_text())
                if training_state_path.is_file()
                else {}
            )
            if isinstance(recipe, _FirstPartyTorchRecipeAdapter):
                recipe._validate_checkpoint_artifacts(checkpoint_dir)
        rank_states = rng.get("rank_states") if isinstance(rng, dict) else None
        if not isinstance(rank_states, list) or len(rank_states) != world_size:
            raise AlgorithmConfigurationError(
                "Torch recipe checkpoint RNG state does not match world size"
            )
        restore_rng_state(rank_states[rank])
        start_epoch = envelope.completed_step
        best_val_loss = float(training_state.get("best_val_loss", float("inf")))
        patience_counter = int(training_state.get("patience_counter", 0))

    model, device = prepare_model(model)
    modules["model"] = model
    if loop.amp and getattr(device, "type", None) != "cuda":
        raise AlgorithmConfigurationError("Torch recipe AMP requires a CUDA worker")
    train_data = ray.train.get_dataset_shard("train")
    if train_data is None:
        raise AlgorithmConfigurationError("Torch recipe did not receive train data")
    evaluation_data: dict[str, object] = {}
    for split in ("val", "test"):
        try:
            shard = ray.train.get_dataset_shard(split)
        except KeyError:
            shard = None
        if shard is not None:
            evaluation_data[split] = shard

    output_shape: tuple[int, ...] | None = None
    for epoch in range(start_epoch, loop.epochs):
        iterator = iter(
            train_data.iter_torch_batches(
                batch_size=loop.batch_size,
                prefetch_batches=loop.prefetch_batches,
                dtypes=torch.float32,
                drop_last=False,
                local_shuffle_buffer_size=loop.local_shuffle_buffer_size,
                local_shuffle_seed=loop.seed + epoch,
            )
        )
        first_raw = next(iterator, None)
        local_non_empty = 1.0 if first_raw is not None else 0.0
        (non_empty_ranks,) = all_reduce_values((local_non_empty,))
        if int(non_empty_ranks) != world_size:
            raise AlgorithmConfigurationError(
                "exact-coverage Torch recipes require at least one batch per rank"
            )
        assert first_raw is not None
        template = _prepare_batch(
            recipe,
            first_raw,
            feature_names=feature_names,
            label_name=label_name,
            weight_name=weight_name,
            config=config,
        )
        current: tuple[Any, Any, Any | None, int] | None = template
        states: dict[str, dict[str, float]] = {
            name: {
                "value": (
                    float("inf")
                    if reduction is MetricReduction.MIN
                    else float("-inf")
                    if reduction is MetricReduction.MAX
                    else 0.0
                ),
                "weight": 0.0,
            }
            for name, reduction in reducers.items()
        }
        local_rows = 0
        local_batches = 0
        collective_steps = 0
        local_coverage_counts: dict[str, int] = {}
        model.train()
        optimizer.zero_grad()
        while True:
            active = current is not None
            prepared = _empty_batch(template) if current is None else current
            features, targets, weights, rows = prepared
            (global_rows,) = all_reduce_values((float(rows),))
            if global_rows <= 0:
                break
            with torch.autocast(
                device_type=device.type,
                enabled=loop.amp,
            ):
                if isinstance(recipe, TrainingRecipeV2):
                    step = recipe.training_step(
                        modules,
                        features,
                        targets,
                        weights,
                        config,
                    )
                    if not isinstance(step, TrainingStepResult):
                        raise AlgorithmConfigurationError(
                            "TrainingRecipeV2 training_step must return "
                            "TrainingStepResult"
                        )
                    predictions = step.predictions
                    raw_loss = step.loss
                    if active:
                        for name, count in step.coverage_counts.items():
                            local_coverage_counts[name] = (
                                local_coverage_counts.get(name, 0) + count
                            )
                else:
                    predictions = recipe.forward(model, features)
                if not isinstance(predictions, torch.Tensor):
                    raise AlgorithmConfigurationError(
                        "Torch recipe forward must return one Tensor"
                    )
                if output_shape is None and active:
                    output_shape = tuple(int(size) for size in predictions.shape[1:])
                aligned_targets = targets
                if predictions.numel() == targets.numel():
                    aligned_targets = targets.reshape_as(predictions)
                if active and not isinstance(recipe, TrainingRecipeV2):
                    raw_loss = recipe.compute_loss(
                        loss_fn,
                        predictions,
                        aligned_targets,
                    )
                    if not isinstance(raw_loss, torch.Tensor) or raw_loss.ndim != 0:
                        raise AlgorithmConfigurationError(
                            "Torch recipe loss must return one scalar batch mean"
                        )
                    if not bool(torch.isfinite(raw_loss)):
                        raise AlgorithmConfigurationError(
                            "Torch recipe loss produced a non-finite value"
                        )
                    local_loss_sum = raw_loss * rows
                elif not active:
                    raw_loss = predictions.sum() * 0.0
                    local_loss_sum = raw_loss
                else:
                    if not isinstance(raw_loss, torch.Tensor) or raw_loss.ndim != 0:
                        raise AlgorithmConfigurationError(
                            "TrainingRecipeV2 loss must return one scalar batch mean"
                        )
                    if not bool(torch.isfinite(raw_loss)):
                        raise AlgorithmConfigurationError(
                            "TrainingRecipeV2 loss produced a non-finite value"
                        )
                    local_loss_sum = raw_loss * rows
                backward_loss = local_loss_sum * world_size / global_rows
            scaler.scale(backward_loss).backward()
            accumulation = (
                optimization_plan.gradient_accumulation_steps
                if optimization_plan is not None
                else 1
            )
            if (collective_steps + 1) % accumulation == 0:
                if (
                    optimization_plan is not None
                    and optimization_plan.max_gradient_norm is not None
                ):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        optimization_plan.max_gradient_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if active:
                _metric_update(
                    states["train_loss"],
                    raw_loss,
                    reduction=MetricReduction.SUM_COUNT,
                    rows=rows,
                    weights=None,
                )
                for name, metric in metrics.items():
                    value = metric(predictions.detach(), aligned_targets.detach())
                    _metric_update(
                        states[name],
                        value,
                        reduction=reducers[name],
                        rows=rows,
                        weights=weights,
                    )
                local_rows += rows
                local_batches += 1
            collective_steps += 1
            raw = next(iterator, None)
            current = (
                _prepare_batch(
                    recipe,
                    raw,
                    feature_names=feature_names,
                    label_name=label_name,
                    weight_name=weight_name,
                    config=config,
                )
                if raw is not None
                else None
            )

        accumulation = (
            optimization_plan.gradient_accumulation_steps
            if optimization_plan is not None
            else 1
        )
        remainder = collective_steps % accumulation
        if remainder:
            gradient_scale = accumulation / remainder
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
            if (
                optimization_plan is not None
                and optimization_plan.max_gradient_norm is not None
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    optimization_plan.max_gradient_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_metrics = {"epoch": epoch + 1, **_reduce_metrics(states, reducers)}
        if scheduler is not None:
            scheduler.step()
        input_rows = {"train": local_rows}
        input_rows.update(
            {
                f"coverage.{name}": count
                for name, count in sorted(local_coverage_counts.items())
            }
        )
        for split, data in evaluation_data.items():
            if split == "test" and epoch + 1 != loop.epochs:
                continue
            split_metrics, split_rows = _evaluate_dataset(
                data,
                split=split,
                recipe=recipe,
                model=model,
                loss_fn=loss_fn,
                metrics=metrics,
                reducers=reducers,
                loop=loop,
                feature_names=feature_names,
                label_name=label_name,
                weight_name=weight_name,
                modules=modules,
                config=config,
            )
            epoch_metrics.update(split_metrics)
            input_rows[split] = split_rows
        if any(not math.isfinite(float(value)) for value in epoch_metrics.values()):
            raise AlgorithmConfigurationError(
                "Torch recipe produced non-finite global metrics"
            )
        stop_after_report = False
        if loop.early_stopping_patience is not None and "val_loss" in epoch_metrics:
            if rank == 0:
                val_loss = float(epoch_metrics["val_loss"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    stop_after_report = patience_counter >= loop.early_stopping_patience
            stop_after_report = broadcast_bool(stop_after_report, source_rank=0)
        execution_workers, model_digest = collective_execution_evidence(
            model,
            shard_rows=local_rows,
            input_binding_digest=config.get("_tributo_input_binding_digest"),
            input_rows=input_rows,
            batch_count=local_batches,
            collective_steps=collective_steps,
        )
        report: dict[str, Any] = {
            **epoch_metrics,
            "execution_workers": list(execution_workers),
            "model_state_digest": model_digest,
            "world_size": world_size,
            "state_coordination": "all_reduce",
            "collective_backend": (
                str(dist.get_backend()) if dist.is_initialized() else "none"
            ),
            "checkpoint_owner_rank": 0,
            "metric_reducers": {
                name: reducer.value for name, reducer in reducers.items()
            },
        }
        should_checkpoint = (
            (epoch + 1) % resume.checkpoint_interval == 0
            or epoch + 1 == loop.epochs
            or stop_after_report
        )
        rank_rng_states = (
            all_gather_objects(capture_rng_state()) if should_checkpoint else ()
        )
        if rank == 0 and should_checkpoint:
            from ray.train import Checkpoint

            checkpoint_dir = Path(tempfile.mkdtemp(prefix="torch_recipe_ckpt_"))
            try:
                torch.save(
                    unwrapped_model(model).state_dict(),
                    checkpoint_dir / "model.pt",
                )
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                torch.save(scaler.state_dict(), checkpoint_dir / "scaler.pt")
                scheduler_files: tuple[str, ...] = ()
                if scheduler is not None:
                    torch.save(
                        scheduler.state_dict(),
                        checkpoint_dir / "scheduler.pt",
                    )
                    scheduler_files = ("scheduler.pt",)
                if output_shape is None:
                    raise AlgorithmConfigurationError(
                        "Torch recipe did not observe a model output shape"
                    )
                model_config = _recipe_checkpoint_contract(
                    recipe,
                    config=config,
                    feature_count=len(feature_names),
                    output_shape=output_shape,
                    framework_version=torch.__version__,
                    model_digest=model_digest,
                    world_size=world_size,
                )
                (checkpoint_dir / "model_config.json").write_text(
                    json.dumps(model_config, ensure_ascii=False),
                    encoding="utf-8",
                )
                (checkpoint_dir / "metrics.json").write_text(
                    json.dumps(epoch_metrics, ensure_ascii=False),
                    encoding="utf-8",
                )
                (checkpoint_dir / "rng_state.json").write_text(
                    json.dumps({"rank_states": list(rank_rng_states)}),
                    encoding="utf-8",
                )
                (checkpoint_dir / "training_state.json").write_text(
                    json.dumps(
                        {
                            "best_val_loss": best_val_loss,
                            "patience_counter": patience_counter,
                        }
                    ),
                    encoding="utf-8",
                )
                extra_files = _write_recipe_checkpoint_artifacts(
                    recipe,
                    checkpoint_dir,
                )
                envelope = write_resume_manifest(
                    checkpoint_dir,
                    resume_id=resume.resume_id,
                    trainer_type=trainer_type,
                    completed_step=epoch + 1,
                    framework="pytorch",
                    framework_version=torch.__version__,
                    payload_files=(
                        "metrics.json",
                        "model.pt",
                        "model_config.json",
                        "optimizer.pt",
                        "scaler.pt",
                        *scheduler_files,
                        "rng_state.json",
                        "training_state.json",
                        *extra_files,
                    ),
                    payload_metadata={
                        "world_size": world_size,
                        "distribution_spec_digest": config.get(
                            "_tributo_distribution_spec_digest"
                        ),
                        **(
                            {"preprocessing": "preprocessor.json"}
                            if "preprocessor.json" in extra_files
                            else {}
                        ),
                    },
                )
                report["resume_id"] = envelope.resume_id
                ray.train.report(
                    report,
                    checkpoint=Checkpoint.from_directory(str(checkpoint_dir)),
                )
            finally:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        else:
            ray.train.report(report)
        if stop_after_report:
            break


@DeveloperAPI
def create_torch_recipe_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> CollectiveAlgorithm:
    """Construct the internal collective adapter for one trusted recipe class."""
    del artifacts
    recipe_cls = _recipe_type(
        plan.implementation.implementation_ref,
        plan.implementation.code_digest,
    )
    if implementation is not recipe_cls:
        raise AlgorithmConfigurationError(
            "Torch recipe implementation drifted after descriptor resolution"
        )
    return _TorchRecipeCollectiveAlgorithm(
        plan,
        _new_recipe(
            plan.implementation.implementation_ref,
            plan.implementation.code_digest,
        ),
    )


@DeveloperAPI
def export_torch_recipe_result(
    *,
    result: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    """Publish a recipe checkpoint through the existing ONNX Bundle pipeline."""
    import importlib.metadata

    from tributo.exporting.models import BundleOutputConfig, ExportTarget
    from tributo.exporting.service import BundleExportService
    from tributo.integrations.algorithm_runtimes.portable_metrics import (
        portable_fit_only_metrics,
    )
    from tributo.integrations.sources.ray_torch_recipe import (
        RayTorchRecipeSourceProvider,
        TorchRecipeSourceOptions,
    )

    output = _OutputConfig.model_validate(plan.algorithm_config.get("output") or {})
    provider = RayTorchRecipeSourceProvider()
    options = TorchRecipeSourceOptions(
        recipe_ref=str(plan.implementation.implementation_ref),
        recipe_code_digest=plan.implementation.code_digest,
        implementation_id=plan.implementation.implementation_id,
    )
    bundle_config = BundleOutputConfig(
        bundle_uri=output.bundle_uri,
        request_id=run_id,
        run_id=run_id,
        targets=[
            ExportTarget(
                name="onnx-model",
                format="onnx",
                exporter_id="torch-onnx-v1",
                options={"opset": 18},
            )
        ],
        roles={"inference": "onnx-model"},
    )
    with provider.open_source(result, options) as source:
        published = BundleExportService().export_bundle(
            source,
            bundle_config,
            tributo_version=importlib.metadata.version("tributo"),
        )
    raw_metrics = getattr(result, "metrics", None) or {}
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics=portable_fit_only_metrics(raw_metrics),
        outputs={
            "bundle_id": published.bundle_id,
            "bundle_uri": published.canonical_uri,
            "execution_id": published.execution_id,
            "manifest_sha256": published.manifest_sha256,
        },
    )


__all__ = [
    "create_torch_recipe_algorithm",
    "export_torch_recipe_result",
    "torch_recipe_train_loop_per_worker",
]
