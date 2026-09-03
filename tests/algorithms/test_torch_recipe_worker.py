"""Worker-level tests for the Ray-native default Torch recipe loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import ray
import ray.train

from tributo.algorithms import (
    TorchBatch,
    TorchLossContribution,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStageRunIdentity,
    TorchStepResult,
)
from tributo.integrations.algorithm_runtimes.ray_train_torch import (
    torch_recipe_train_loop_per_worker,
)

torch = pytest.importorskip("torch")


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
    reports: list[tuple[dict[str, Any], set[str]]] = []

    def report(metrics: dict[str, Any], checkpoint: Any | None = None) -> None:
        assert checkpoint is not None
        with checkpoint.as_directory() as directory:
            reports.append(
                (dict(metrics), {path.name for path in Path(directory).iterdir()})
            )

    monkeypatch.setattr(ray.train.torch, "prepare_model", lambda model: model)
    monkeypatch.setattr(ray.train, "get_context", lambda: _TrainContext())
    monkeypatch.setattr(
        ray.train,
        "get_dataset_shard",
        lambda name: data if name == "train" else (_ for _ in ()).throw(KeyError(name)),
    )
    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: None)
    monkeypatch.setattr(ray.train, "report", report)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _RuntimeContext())
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
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
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
    assert {
        "model.pt",
        "optimizer.pt",
        "scaler.pt",
        "rng_state.pt",
        "torch_checkpoint_descriptor.json",
    } <= reports[0][1]


def test_recipe_worker_rejects_stale_retry_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tributo.algorithms import TorchCheckpointDescriptor

    recipe = BinaryLinearRecipe()
    modules = recipe.build_modules(None)
    model = modules["model"]
    optimizer = recipe.configure_optimizers(modules, None).optimizer
    torch.save(model.state_dict(), tmp_path / "model.pt")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save({}, tmp_path / "scaler.pt")
    (tmp_path / "rng_state.pt").write_bytes(torch.get_rng_state().numpy().tobytes())
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
    payload_files = {
        name: __import__("hashlib").sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("model.pt", "optimizer.pt", "scaler.pt", "rng_state.pt")
    }
    descriptor = TorchCheckpointDescriptor(
        1,
        identity,
        identity.run_config_name,
        "replicated",
        1,
        1,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "0" * 64,
        payload_files,
    )
    (tmp_path / "torch_checkpoint_descriptor.json").write_text(
        json.dumps(descriptor.to_dict()), encoding="utf-8"
    )
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
    monkeypatch.setattr(ray.train.torch, "prepare_model", lambda model: model)

    class _Checkpoint:
        def as_directory(self):
            from contextlib import contextmanager

            @contextmanager
            def opened():
                yield tmp_path

            return opened()

    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: _Checkpoint())
    monkeypatch.setattr(
        ray.train,
        "report",
        lambda metrics, checkpoint=None: reports.append(dict(metrics)),
    )
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _RuntimeContext())

    torch_recipe_train_loop_per_worker(
        {
            "model": {"input_features": 2},
            "optimizer": {"learning_rate": 0.1},
            "training": {"epochs": 2, "batch_size": 2, "seed": 7},
            "ray": {"resume": {"checkpoint_interval": 1}},
            "_tributo_recipe_ref": ("tests.support.torch_recipe:BinaryLinearRecipe"),
            "_tributo_recipe_code_digest": None,
            "_tributo_implementation_id": "example.binary_linear",
            "_tributo_algorithm": "binary_linear",
            "_tributo_feature_names": ["x1", "x2"],
            "_tributo_label_name": "label",
            "_tributo_weight_name": None,
            "_tributo_input_binding_digest": "a" * 64,
            "_tributo_distribution_spec_digest": "b" * 64,
            "_tributo_resume_from": str(tmp_path),
            "_tributo_metric_reducers": {
                "accuracy": "sum_count",
                "train_loss": "sum_count",
            },
            "_core_implementation_ref": "tests.support.torch_recipe:BinaryLinearRecipe",
            "_core_implementation_code_digest": "0" * 64,
            "_core_input_binding_digest": "a" * 64,
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
        },
    )

    assert reports and reports[0]["checkpoint_descriptor"]["completed_step"] == 3
