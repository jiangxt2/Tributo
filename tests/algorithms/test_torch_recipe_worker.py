"""Worker-level tests for the Ray-native default Torch recipe loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import ray
import ray.train

from tests.support.torch_recipe import BinaryLinearRecipe
from tributo.integrations.algorithm_runtimes.torch_recipe import (
    torch_recipe_train_loop_per_worker,
)

torch = pytest.importorskip("torch")


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


def test_default_recipe_streams_torch_batches_and_reports_resumable_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        {
            "x1": torch.tensor([0.0, 1.0, 0.0]),
            "x2": torch.tensor([0.0, 0.0, 1.0]),
            "label": torch.tensor([0.0, 1.0, 1.0]),
        },
        {
            "x1": torch.tensor([1.0, 1.0]),
            "x2": torch.tensor([0.0, 1.0]),
            "label": torch.tensor([1.0, 1.0]),
        },
    ]
    data = _Iterator(batches)
    validation = _Iterator(
        [
            {
                "x1": torch.tensor([0.0, 1.0]),
                "x2": torch.tensor([0.0, 1.0]),
                "label": torch.tensor([0.0, 1.0]),
            }
        ]
    )
    reports: list[tuple[dict[str, Any], set[str]]] = []

    def report(metrics: dict[str, Any], checkpoint: Any | None = None) -> None:
        files: set[str] = set()
        if checkpoint is not None:
            with checkpoint.as_directory() as raw_directory:
                files = {
                    path.name
                    for path in Path(raw_directory).iterdir()
                    if path.is_file()
                }
        reports.append((dict(metrics), files))

    def get_dataset_shard(name: str) -> _Iterator:
        if name == "train":
            return data
        if name == "val":
            return validation
        raise KeyError(name)

    monkeypatch.setattr(ray.train, "get_context", lambda: _TrainContext())
    monkeypatch.setattr(ray.train, "get_dataset_shard", get_dataset_shard)
    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: None)
    monkeypatch.setattr(ray.train, "report", report)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _RuntimeContext())

    torch_recipe_train_loop_per_worker(
        {
            "model": {"input_features": 2},
            "optimizer": {"learning_rate": 0.1},
            "training": {
                "epochs": 1,
                "batch_size": 3,
                "prefetch_batches": 2,
                "seed": 7,
            },
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
            "_tributo_metric_reducers": {
                "accuracy": "sum_count",
                "train_loss": "sum_count",
            },
        },
        BinaryLinearRecipe(),
    )

    assert data.calls == [
        {
            "batch_size": 3,
            "prefetch_batches": 2,
            "dtypes": torch.float32,
            "drop_last": False,
            "local_shuffle_buffer_size": None,
            "local_shuffle_seed": 7,
        }
    ]
    assert len(reports) == 1
    metrics, files = reports[0]
    assert metrics["epoch"] == 1
    assert 0.0 <= metrics["val_accuracy"] <= 1.0
    assert metrics["val_loss"] >= 0.0
    assert metrics["metric_reducers"] == {
        "accuracy": "sum_count",
        "train_loss": "sum_count",
    }
    assert metrics["execution_workers"][0]["input_rows"] == {
        "train": 5,
        "val": 2,
    }
    assert metrics["execution_workers"][0]["batch_count"] == 2
    assert files == {
        "metrics.json",
        "model.pt",
        "model_config.json",
        "optimizer.pt",
        "resume.json",
        "rng_state.json",
        "scaler.pt",
        "training_state.json",
    }
    assert validation.calls == [
        {
            "batch_size": 3,
            "prefetch_batches": 2,
            "dtypes": torch.float32,
            "drop_last": False,
        }
    ]


def test_default_recipe_resumes_from_explicit_worker_visible_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tributo.training.checkpoint import (
        capture_rng_state,
        write_resume_manifest,
    )

    recipe = BinaryLinearRecipe()
    model = recipe.model_factory({"input_features": 2})
    assert isinstance(model, torch.nn.Module)
    optimizer = recipe.optimizer_factory(model, {"learning_rate": 0.1})
    torch.save(model.state_dict(), tmp_path / "model.pt")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    (tmp_path / "rng_state.json").write_text(
        json.dumps({"rank_states": [capture_rng_state()]}),
        encoding="utf-8",
    )
    write_resume_manifest(
        tmp_path,
        trainer_type="torch_recipe",
        completed_step=1,
        framework="pytorch",
        framework_version=torch.__version__,
        payload_files=("model.pt", "optimizer.pt", "rng_state.json"),
        payload_metadata={
            "world_size": 1,
            "distribution_spec_digest": "b" * 64,
        },
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
    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: None)
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
        },
        recipe,
    )

    assert [report["epoch"] for report in reports] == [2]
