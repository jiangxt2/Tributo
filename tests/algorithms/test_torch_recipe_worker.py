"""Worker-level tests for the Ray-native default Torch recipe loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import ray
import ray.train

from tributo.algorithms import (
    TorchBatch,
    TorchLossContribution,
    TorchMetricContribution,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStageRunIdentity,
    TorchStepResult,
)
from tributo.algorithms.api import AlgorithmExecutionError
from tributo.integrations.algorithm_runtimes.ray_train_torch import (
    torch_recipe_train_loop_per_worker,
)

torch = pytest.importorskip("torch")
ray_train_torch = pytest.importorskip("ray.train.torch")


class BinaryLinearRecipe(TorchRecipe):
    def build_modules(self, context: object) -> TorchModuleSet:
        return TorchModuleSet(
            {"model": torch.nn.Linear(2, 1), "loss": torch.nn.MSELoss()}
        )

    def adapt_batch(self, batch: object, context: object) -> TorchBatch:
        del context
        features = torch.stack((batch["x1"], batch["x2"]), dim=1)
        return TorchBatch(
            positional=(features,),
            targets=batch["label"].reshape(-1, 1),
            local_rows=len(batch["label"]),
        )

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        del context
        predictions = modules["model"](batch.positional[0])
        numerator = torch.nn.functional.mse_loss(
            predictions, batch.targets, reduction="sum"
        )
        return TorchStepResult(
            outputs={"prediction": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def configure_optimizers(
        self, modules: TorchModuleSet, context: object
    ) -> TorchOptimizationPlan:
        del context
        return TorchOptimizationPlan(
            torch.optim.SGD(modules["model"].parameters(), lr=0.1)
        )

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"train_loss": "sum_count"})

    def artifact_plan(self, context: object) -> dict[str, object]:
        del context
        return {"source_kind": "torch_module"}


class UnexpectedMetricRecipe(BinaryLinearRecipe):
    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: object
    ) -> TorchStepResult:
        step = super().training_step(modules, batch, context)
        return TorchStepResult(
            outputs=step.outputs,
            loss=step.loss,
            metrics={"unexpected": TorchMetricContribution(1.0, 1.0)},
        )


class _Iterator:
    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self._batches = batches
        self.calls: list[dict[str, Any]] = []

    def iter_torch_batches(self, **kwargs: Any):
        self.calls.append(kwargs)
        yield from self._batches


class _TrainContext:
    def get_world_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1


class _RuntimeContext:
    def get_worker_id(self) -> str:
        return "worker-0"

    def get_node_id(self) -> str:
        return "node-0"

    def get_assigned_resources(self) -> dict[str, float]:
        return {"CPU": 1.0}


def test_recipe_worker_reports_typed_checkpoint_and_exact_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = _Iterator(
        [
            {
                "x1": torch.tensor([0.0, 1.0]),
                "x2": torch.tensor([0.0, 1.0]),
                "label": torch.tensor([0.0, 1.0]),
            },
            {
                "x1": torch.tensor([1.0]),
                "x2": torch.tensor([0.0]),
                "label": torch.tensor([1.0]),
            },
        ]
    )
    val_data = _Iterator(
        [
            {
                "x1": torch.tensor([0.5]),
                "x2": torch.tensor([0.5]),
                "label": torch.tensor([1.0]),
            }
        ]
    )
    reports: list[tuple[dict[str, Any], set[str]]] = []

    def report(metrics: dict[str, Any], checkpoint: Any | None = None) -> None:
        assert checkpoint is not None
        with checkpoint.as_directory() as directory:
            reports.append(
                (dict(metrics), {path.name for path in Path(directory).iterdir()})
            )

    monkeypatch.setattr(ray_train_torch, "prepare_model", lambda model: model)
    monkeypatch.setattr(ray.train, "get_context", lambda: _TrainContext())
    monkeypatch.setattr(
        ray.train,
        "get_dataset_shard",
        lambda name: (
            data
            if name == "train"
            else val_data
            if name == "val"
            else (_ for _ in ()).throw(KeyError(name))
        ),
    )
    monkeypatch.setattr(
        ray.train,
        "get_checkpoint",
        lambda: (_ for _ in ()).throw(AssertionError("retry state must not be read")),
    )
    monkeypatch.setattr(ray.train, "report", report)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _RuntimeContext())
    monkeypatch.setattr(
        "tributo.integrations.algorithm_runtimes.ray_train_torch._validate_module_digest",
        lambda reference, digest: None,
    )
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "example",
        "example.binary",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    runtime = TorchRuntimeContext(
        {},
        "example.binary",
        1,
        "1" * 64,
        "2" * 64,
        identity,
        input_binding_digest="3" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train", "val"))
    torch_recipe_train_loop_per_worker(
        {
            "training": {"epochs": 1, "batch_size": 2},
            "_core_implementation_ref": "tests.algorithms.test_torch_recipe_worker:BinaryLinearRecipe",
            "_core_implementation_code_digest": "0" * 64,
            "_core_input_binding_digest": "3" * 64,
            "_core_stage_context": stage.to_dict(),
        }
    )
    assert reports and reports[0][0]["checkpoint_descriptor"]["completed_step"] == 2
    assert reports[0][0]["execution_workers"][0]["input_rows"] == {
        "train": 3,
        "val": 1,
    }
    assert {
        "model.pt",
        "torch_execution_evidence.json",
        "torch_checkpoint_descriptor.json",
    } <= reports[0][1]


def test_recipe_worker_replays_ray_retry_from_stage_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _Iterator(
        [
            {
                "x1": torch.tensor([0.0, 1.0]),
                "x2": torch.tensor([0.0, 1.0]),
                "label": torch.tensor([0.0, 1.0]),
            }
        ]
    )
    reports: list[dict[str, Any]] = []

    def get_dataset_shard(name: str) -> _Iterator:
        if name == "train":
            return data
        raise KeyError(name)

    monkeypatch.setattr(ray.train, "get_context", lambda: _TrainContext())
    monkeypatch.setattr(ray.train, "get_dataset_shard", get_dataset_shard)
    monkeypatch.setattr(ray_train_torch, "prepare_model", lambda model: model)

    monkeypatch.setattr(
        ray.train,
        "get_checkpoint",
        lambda: (_ for _ in ()).throw(AssertionError("retry state must not be read")),
    )
    monkeypatch.setattr(
        ray.train,
        "report",
        lambda metrics, checkpoint=None: reports.append(dict(metrics)),
    )
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _RuntimeContext())
    monkeypatch.setattr(
        "tributo.integrations.algorithm_runtimes.ray_train_torch._validate_module_digest",
        lambda reference, digest: None,
    )

    config = {
        "model": {"input_features": 2},
        "optimizer": {"learning_rate": 0.1},
        "training": {"epochs": 2, "batch_size": 2, "seed": 7},
        "_core_implementation_ref": "tests.support.torch_recipe:BinaryLinearRecipe",
        "_core_implementation_code_digest": "0" * 64,
        "_core_input_binding_digest": "a" * 64,
        "_core_feature_names": ["x1", "x2"],
        "_core_label_name": "label",
        "_core_stage_context": TorchStageContext(
            TorchRuntimeContext(
                {},
                "example.binary_linear",
                1,
                "1" * 64,
                "2" * 64,
                TorchStageRunIdentity(
                    "aabbccdd",
                    "11223344",
                    "train",
                    1,
                    "binary",
                    "example.binary_linear",
                    "0" * 64,
                    "1" * 64,
                    "2" * 64,
                ),
                input_binding_digest="a" * 64,
            ),
            "train",
            0,
            True,
            ("train",),
        ).to_dict(),
    }
    torch.manual_seed(101)
    torch_recipe_train_loop_per_worker(config)
    torch.manual_seed(202)
    torch_recipe_train_loop_per_worker(config)

    assert len(reports) == 2
    assert all(
        report["checkpoint_descriptor"]["completed_step"] == 2 for report in reports
    )
    assert reports[0]["model_state_digest"] == reports[1]["model_state_digest"]


def test_recipe_worker_rejects_metric_plan_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _Iterator(
        [
            {
                "x1": torch.tensor([0.0]),
                "x2": torch.tensor([0.0]),
                "label": torch.tensor([0.0]),
            }
        ]
    )
    monkeypatch.setattr(ray.train, "get_context", lambda: _TrainContext())
    monkeypatch.setattr(
        ray.train,
        "get_dataset_shard",
        lambda name: data if name == "train" else (_ for _ in ()).throw(KeyError(name)),
    )
    monkeypatch.setattr(ray_train_torch, "prepare_model", lambda model: model)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _RuntimeContext())
    monkeypatch.setattr(
        "tributo.integrations.algorithm_runtimes.ray_train_torch._validate_module_digest",
        lambda reference, digest: None,
    )
    stage = TorchStageContext(
        TorchRuntimeContext(
            {},
            "example.unexpected_metric",
            1,
            "1" * 64,
            "2" * 64,
            TorchStageRunIdentity(
                "aabbccdd",
                "11223344",
                "train",
                1,
                "example",
                "example.unexpected_metric",
                "0" * 64,
                "1" * 64,
                "2" * 64,
            ),
            input_binding_digest="3" * 64,
        ),
        "train",
        0,
        True,
        ("train",),
    )
    with pytest.raises(
        AlgorithmExecutionError,
        match="metrics do not match TorchMetricPlan",
    ):
        torch_recipe_train_loop_per_worker(
            {
                "training": {"epochs": 1, "batch_size": 1},
                "_core_implementation_ref": (
                    "tests.algorithms.test_torch_recipe_worker:UnexpectedMetricRecipe"
                ),
                "_core_implementation_code_digest": "0" * 64,
                "_core_input_binding_digest": "3" * 64,
                "_core_stage_context": stage.to_dict(),
            }
        )
