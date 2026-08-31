"""Focused multi-node Gate for the highest-value classical and causal paths."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import cast

import ray

sys.path.insert(0, str(Path(__file__).parent))
from official_algorithm_gate_job import (
    _execute,
    _execute_baseline_equivalence,
    _execute_checkpoint_recovery,
    _execute_distributed_inference,
    _stage_data,
    _stage_inference_data,
)


def main() -> None:
    root = Path(os.environ["TRIBUTO_OFFICIAL_ALGORITHM_GATE_ROOT"])
    if root.parent != Path("/workspace/tributo-work") or not root.name.startswith(
        "tributo-priority-algorithm-gate-"
    ):
        raise ValueError("priority algorithm Gate root is outside the owned workspace")
    if root.exists():
        raise FileExistsError(f"refusing to reuse Gate root: {root}")
    root.mkdir(parents=True)

    def run_case(**kwargs: object) -> dict[str, object]:
        """Run one algorithm with a fresh driver connection to the cluster."""
        algorithm = str(kwargs.get("algorithm", "unknown"))
        if ray.is_initialized():
            ray.shutdown()
        gc.collect()
        ray.init(address="auto")
        kwargs["require_onnx"] = True
        try:
            result = _execute(**kwargs)
            if result is None:
                raise AssertionError(f"priority Gate unexpectedly skipped {algorithm}")
        except Exception as exc:
            print(
                "CASE_FAILURE: "
                + json.dumps(
                    {
                        "algorithm": algorithm,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:4096],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
        print(
            "CASE_RESULT: "
            + json.dumps(
                {
                    "algorithm": algorithm,
                    "status": result["status"],
                    "onnx_exported": result["onnx_exported"],
                    "inference_roundtrip": result["inference_roundtrip"],
                    "cluster_distributed": result["receipt"]["cluster_distributed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return cast(dict[str, object], result)

    ray.init(address="auto")
    try:
        data_path = _stage_data(root)
        inference_path = _stage_inference_data(root)
        records = [
            run_case(
                algorithm="random_forest",
                implementation_id="tributo.official.random_forest.joblib",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "task": "classification",
                    "n_estimators": 8,
                    "seed": 7,
                    "output": {"bundle_uri": str(root / "rf-joblib-bundle")},
                },
            ),
            run_case(
                algorithm="logistic_regression",
                implementation_id=None,
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "feature_count": 2,
                    "C": 1.0,
                    "learning_rate": 1.0,
                    "tolerance": 0.2,
                    "runtime": {"checkpoint_dir": str(root / "lr-checkpoint")},
                    "output": {"bundle_uri": str(root / "lr-bundle")},
                },
            ),
            run_case(
                algorithm="xgboost",
                implementation_id="tributo.official.boosting.xgboost",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "label_col": "label",
                        "feature_columns": ["x0", "x1"],
                    },
                    "model": {
                        "objective": "binary:logistic",
                        "tree_method": "hist",
                        "max_depth": 2,
                        "eta": 0.3,
                        "eval_metric": "logloss",
                    },
                    "training": {"num_rounds": 3},
                    "ray": {"storage_path": str(root / "xgboost-ray-results")},
                    "output": {"bundle_uri": str(root / "xgboost-bundle")},
                },
            ),
            run_case(
                algorithm="linear_dml_ate",
                implementation_id=None,
                feature_names=("treatment", "x0"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "treatment_col": "treatment",
                    "cross_fit_folds": 5,
                    "policy_cost": 0.5,
                    "output": {"bundle_uri": str(root / "causal-dml-bundle")},
                },
            ),
            run_case(
                algorithm="doubly_robust_ate",
                implementation_id=None,
                feature_names=("x0", "x1", "treatment"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "feature_columns": ["x0", "x1"],
                        "treatment_col": "treatment",
                        "outcome_col": "outcome",
                    },
                    "model": {},
                    # The implementation defaults to five folds.  This focused
                    # multi-node Gate uses two to keep the number of nested
                    # XGBoost worker groups bounded on a shared Docker VM.
                    "training": {"num_rounds": 1, "cross_fit_folds": 2},
                    "ray": {"storage_path": str(root / "causal-dr-ray-results")},
                    "output": {"bundle_uri": str(root / "causal-dr-bundle")},
                },
            ),
            run_case(
                algorithm="x_learner",
                implementation_id="tributo.official.causal_xlearner.xgboost",
                feature_names=("x0", "x1", "treatment", "identity"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "feature_columns": ["x0", "x1"],
                        "treatment_col": "treatment",
                        "outcome_col": "outcome",
                        "identity_col": "identity",
                    },
                    "model": {},
                    # The implementation defaults to five folds.  This focused
                    # multi-node Gate uses two to keep the number of nested
                    # XGBoost worker groups bounded on a shared Docker VM.
                    "training": {"num_rounds": 1, "cross_fit_folds": 2},
                    "ray": {"storage_path": str(root / "xlearner-ray-results")},
                    "output": {"bundle_uri": str(root / "xlearner-bundle")},
                },
            ),
        ]
        baseline = _execute_baseline_equivalence(data_path=data_path, records=records)
        recovery = _execute_checkpoint_recovery(data_path=data_path, root=root)
        xgboost_bundle = cast(
            str,
            next(
                record["bundle_uri"]
                for record in records
                if record["algorithm"] == "xgboost"
            ),
        )
        inference = _execute_distributed_inference(
            bundle_uri=xgboost_bundle,
            input_path=inference_path,
            root=root,
        )
        print("BASELINE_RESULT: " + json.dumps(baseline, sort_keys=True), flush=True)
        print("RECOVERY_RESULT: " + json.dumps(recovery, sort_keys=True), flush=True)
        print("INFERENCE_RESULT: " + json.dumps(inference, sort_keys=True), flush=True)
        print("RESULT: " + json.dumps(records, sort_keys=True), flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
