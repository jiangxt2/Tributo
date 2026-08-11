"""Run the DNN or PU correctness baseline inside a Ray cluster."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
def _stage_parquet(path: str, records: dict[str, list[Any]]) -> None:
    """Create the same local fixture on every Ray node."""
    import pandas as pd

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(destination, index=False)


@ray.remote(num_cpus=0)
def _remove_fixture(root: str) -> None:
    """Remove only the per-job directory created by this script."""
    path = Path(root)
    if path.name.startswith("tributo-correctness-") and path.exists():
        shutil.rmtree(path)


def _on_every_node(remote_function: Any, *args: Any) -> list[Any]:
    refs = []
    for node in ray.nodes():
        if not node["Alive"] or node["Resources"].get("CPU", 0) <= 0:
            continue
        refs.append(
            remote_function.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote(*args)
        )
    return ray.get(refs)


def _worker_node_id() -> str:
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("CPU", 0) > 0:
            return str(node["NodeID"])
    raise RuntimeError("No live Ray worker node with CPU resources")


def _training_config(algorithm: str, data_path: str, output_dir: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "features": [
            {"name": "f0", "type": "dense"},
            {"name": "f1", "type": "dense"},
        ],
        "label_col": "label",
        "model": {"dnn_hidden_units": [8], "dnn_dropout": 0.0},
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "val_size": 0.25,
            "seed": 42,
        },
        "ray": {"num_workers": 1, "use_gpu": False, "max_failures": 0},
        "output": {"bundle_uri": output_dir},
    }
    if algorithm in {"dnn", "dnn_nnpu"}:
        config["data"] = {"type": "parquet", "path": data_path}
        if algorithm == "dnn":
            config["loss"] = {"type": "bce"}
        else:
            config["loss"] = {"type": "nnpu"}
            config["pu_learning"] = {"enabled": True, "class_prior": 0.35}
    elif algorithm == "pu":
        config["data"] = {"source": {"provider": "tributo.parquet", "uri": data_path}}
        config["pu"] = {"loss_type": "nnpu", "class_prior": 0.35}
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return config


@ray.remote(num_cpus=0)
def _run_training(algorithm: str, data_path: str, output_dir: str) -> dict[str, Any]:
    """Keep framework imports and model export off the constrained Jobs driver."""
    import math

    config = _training_config(algorithm, data_path, output_dir)
    if algorithm in {"dnn", "dnn_nnpu"}:
        from tributo.training.dnn_trainer import run_dnn_training_with_config

        summary = run_dnn_training_with_config(config)
    else:
        from tributo.training.pu_trainer import run_pu_training_with_config

        summary = run_pu_training_with_config(config)

    from tributo.exporting.bundle_reader import BundleReader

    metrics = summary["metrics"]
    with BundleReader().open_artifact(
        summary["bundle_uri"], role="inference"
    ) as artifact:
        onnx_path = artifact.path_for("model.onnx")
        onnx_exists = onnx_path.is_file()
        onnx_size = onnx_path.stat().st_size if onnx_exists else 0
    result = {
        "algorithm": algorithm,
        "status": summary["status"],
        "epoch": int(metrics["epoch"]),
        "train_loss": float(metrics["train_loss"]),
        "class_prior": float(metrics["class_prior"]) if algorithm == "pu" else None,
        "onnx_exists": onnx_exists,
        "onnx_size": onnx_size,
    }
    if algorithm in {"dnn_nnpu", "pu"}:
        result["train_optimization_objective"] = float(
            metrics["train_optimization_objective"]
        )
        result["train_observed_label_accuracy"] = float(
            metrics["train_observed_label_accuracy"]
        )
        result["train_acc"] = float(metrics["train_acc"])
        result["val_observed_label_accuracy"] = float(
            metrics["val_observed_label_accuracy"]
        )
        result["val_acc"] = float(metrics["val_acc"])
    if result["status"] != "succeeded":
        raise AssertionError(f"Unexpected run status: {result['status']}")
    if not result["onnx_exists"] or result["onnx_size"] <= 0:
        raise AssertionError(f"Missing ONNX artifact in {summary['bundle_uri']}")
    if result["epoch"] != 1 or not math.isfinite(result["train_loss"]):
        raise AssertionError(f"Invalid training metrics: {result}")
    if algorithm == "pu" and result["class_prior"] != 0.35:
        raise AssertionError(f"Class prior was not preserved: {result}")
    if algorithm in {"dnn_nnpu", "pu"} and (
        result["train_acc"] != result["train_observed_label_accuracy"]
        or result["val_acc"] != result["val_observed_label_accuracy"]
    ):
        raise AssertionError(f"Compatibility metric aliases diverged: {result}")
    return result


def main() -> int:
    """Stage input, execute real Ray Train, and report verifiable facts."""
    algorithm = os.environ["ALGORITHM"]
    ray.init()

    rng = random.Random(42)
    f0 = [rng.gauss(0.0, 1.0) for _ in range(64)]
    f1 = [rng.gauss(0.0, 1.0) for _ in range(64)]
    latent_positive = [left + 0.4 * right > 0 for left, right in zip(f0, f1)]
    labels = [
        float(is_positive and rng.random() < 0.5) for is_positive in latent_positive
    ]
    records = {"f0": f0, "f1": f1, "label": labels}

    root = f"/tmp/tributo-correctness-{uuid.uuid4().hex}"
    data_path = f"{root}/data.parquet"
    output_dir = f"{root}/{algorithm}-model"

    try:
        _on_every_node(_stage_parquet, data_path, records)
        result = ray.get(
            _run_training.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=_worker_node_id(), soft=False
                )
            ).remote(algorithm, data_path, output_dir)
        )
        print(f"RESULT: {json.dumps(result, sort_keys=True)}")
    finally:
        _on_every_node(_remove_fixture, root)
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
