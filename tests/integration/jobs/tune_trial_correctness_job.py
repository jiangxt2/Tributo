"""Ray Job proving fit-only Tune trials and inner Ray Train isolation."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any


def main() -> int:
    """Run two real XGBoost Tune trials without publishing a Bundle."""
    workspace_root = Path("/workspace/tributo-work")
    # Ray storage helpers call os.getcwd() during Tune finalization. Keep the
    # Job driver on the stable Compose volume instead of its packaged runtime cwd.
    os.chdir(workspace_root)

    import ray
    from ray.data import DataContext

    from tributo.training.config import build_effective_config
    from tributo.training.registry import get_trainer
    from tributo.training.tune_config import TuneSearchConfig
    from tributo.training.tune_runner import TuneRunner, extract_best_params
    from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec

    DataContext.get_current().enable_progress_bars = False

    execution_suffix = uuid.uuid4().hex
    root = workspace_root / f"tune-{execution_suffix}"
    tune_output = root / "tune-output"
    tune_output.mkdir(parents=True)
    experiment_name = f"tune-trial-correctness-{execution_suffix[:8]}"

    rows = [
        {
            "feature_a": float(index % 4),
            "feature_b": float(index // 4),
            "label": index % 2,
        }
        for index in range(16)
    ]
    dataset = ray.data.from_items(rows)
    trainer_spec = get_trainer("xgboost")
    effective_config = build_effective_config(
        trainer_spec,
        {
            "data": {
                "label_col": "label",
                "feature_columns": ["feature_a", "feature_b"],
            },
            "model": {
                "objective": "binary:logistic",
                "eta": 0.3,
            },
            "training": {
                "num_rounds": 2,
                "val_size": 0.0,
                "test_size": 0.0,
                "max_rows_per_worker": 32,
                "seed": 42,
            },
            "ray": {
                "num_workers": 1,
                "use_gpu": False,
                "max_failures": 0,
            },
        },
        datasets_supplied=True,
    )
    search_space = SearchSpaceSpec(
        parameters=(
            SearchParamSpec(
                path="model.max_depth",
                kind="grid_search",
                values=(1, 2),
            ),
        )
    )
    runner = TuneRunner(
        trainer_spec,
        TuneSearchConfig(
            metric="train-logloss",
            mode="min",
            num_samples=1,
            max_concurrent_trials=2,
            search_alg="random",
            scheduler="fifo",
            fail_fast=True,
        ),
        search_space,
        effective_config,
    )

    result_grid = runner.run(
        datasets={"train": dataset},
        output_path=str(tune_output),
        experiment_name=experiment_name,
    )
    results = [result_grid[index] for index in range(len(result_grid))]
    assert len(results) == 2
    assert result_grid.num_errors == 0

    target_values: list[float] = []
    for result in results:
        assert result.metrics is not None
        target_values.append(float(result.metrics["train-logloss"]))
    assert all(math.isfinite(value) for value in target_values)
    result_paths = [Path(result.path) for result in results if result.path is not None]
    assert len(result_paths) == 2
    assert len(set(result_paths)) == 2
    trial_namespace = tune_output / "trials"
    assert all(path.is_relative_to(trial_namespace) for path in result_paths)

    checkpoints = [result.checkpoint for result in results]
    assert all(checkpoint is not None for checkpoint in checkpoints)
    checkpoint_paths = [
        str(checkpoint.path) for checkpoint in checkpoints if checkpoint
    ]
    assert len(checkpoint_paths) == 2
    assert len(set(checkpoint_paths)) == 2

    inner_storage_paths: list[Path] = []
    for result_path in result_paths:
        trial_inner_storage = [
            path for path in result_path.rglob("_tributo_ray_train") if path.is_dir()
        ]
        assert len(trial_inner_storage) == 1
        inner_storage_paths.extend(trial_inner_storage)
    inner_storage_roots = sorted(
        str(path.relative_to(tune_output)) for path in inner_storage_paths
    )
    assert len(inner_storage_roots) == 2
    assert len(set(inner_storage_roots)) == 2

    manifest_paths = list(tune_output.rglob("manifest.json"))
    assert manifest_paths == []
    assert not (tune_output / "aliases").exists()
    summary_path = tune_output / "tune_summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text())
    assert summary["num_trials"] == 2
    assert summary["num_errors"] == 0

    best_params = extract_best_params(
        result_grid,
        metric="train-logloss",
        mode="min",
    )
    assert set(best_params) == {"model.max_depth"}

    versions = {
        package: importlib.metadata.version(package) for package in ("ray", "xgboost")
    }
    payload = {
        "status": "succeeded",
        "num_trials": len(results),
        "num_errors": result_grid.num_errors,
        "target_values": target_values,
        "best_params": best_params,
        "result_paths": [str(path) for path in result_paths],
        "checkpoint_paths": checkpoint_paths,
        "inner_storage_roots": inner_storage_roots,
        "bundle_manifests": len(manifest_paths),
        "summary_exists": summary_path.is_file(),
        "alive_nodes": len([node for node in ray.nodes() if node["Alive"]]),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "versions": versions,
    }
    print(f"RESULT: {json.dumps(payload, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
