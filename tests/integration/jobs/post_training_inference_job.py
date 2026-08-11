"""Cluster job exercising the inline post-training inference adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow.dataset as pads
import ray

from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.exporting.models import BundleOutputConfig, BundleRef, ExportTarget
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    RayExecutionPolicy,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.inference.post_training import (
    PostTrainingInferenceAction,
    run_post_training_inference,
)


def main() -> None:
    ray.init(address="auto")
    bundle_ref = _train_bundle()
    if os.environ.get("POST_TRAINING_MODE", "inline") == "train-only":
        result_path = Path(os.environ["TRAINING_RESULT_PATH"])
        result_path.write_text(bundle_ref.model_dump_json(), encoding="utf-8")
        print("RESULT: " + bundle_ref.model_dump_json())
        return

    action = PostTrainingInferenceAction(
        input=IngestionRequest(
            source=ParquetSourceConfig(path=os.environ["INPUT_URI"]),
            engine="ray",
        ),
        input_binding=InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="float_input",
                    columns=("feature_a", "feature_b"),
                    dtype="float32",
                ),
            ),
            passthrough_columns=("entity_id",),
        ),
        output_binding=OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="label",
                    column="prediction",
                    semantic="label",
                    squeeze_singleton=True,
                ),
                TensorOutputBinding(
                    tensor_name="probabilities",
                    column="score",
                    semantic="probability",
                ),
            )
        ),
        result_sink=ParquetResultSinkRequest(uri=os.environ["OUTPUT_URI"]),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        mode="inline",
    )
    result = run_post_training_inference(
        action,
        bundle_ref,
        parent_run_id=os.environ["PARENT_RUN_ID"],
    )
    if result.status != "succeeded":
        raise RuntimeError(result.model_dump_json())

    table = pads.dataset(os.environ["OUTPUT_URI"], format="parquet").to_table()
    print(
        "RESULT: "
        + json.dumps(
            {
                "status": result.status,
                "parent_run_id": result.parent_run_id,
                "rows": table.num_rows,
                "columns": table.column_names,
                "bundle_id": bundle_ref.bundle_id,
            },
            sort_keys=True,
        )
    )


def _train_bundle() -> BundleRef:
    from tributo.training.data_loader import load_ray_dataset_from_source
    from tributo.training.xgboost_trainer import XGBoostTrainerImpl

    input_uri = os.environ["INPUT_URI"]
    source = {
        "provider": "tributo.parquet",
        "uri": input_uri,
        "options": {},
    }
    dataset = load_ray_dataset_from_source(source)
    trainer = XGBoostTrainerImpl(
        datasets={"train": dataset},
        config=_training_config(source, os.environ["TRAINING_RESULTS_URI"]),
    )
    summary = trainer.run(
        bundle_config=BundleOutputConfig(
            bundle_uri=os.environ["TRAINING_BUNDLE_URI"],
            targets=[
                ExportTarget(
                    name="xgboost-onnx",
                    format="onnx",
                    options={"opset": 12},
                )
            ],
            roles={"inference": "xgboost-onnx"},
        )
    )
    if summary["status"] != "succeeded":
        raise RuntimeError("Training Bundle publication did not succeed")
    return BundleRef(
        canonical_uri=summary["canonical_uri"],
        bundle_id=summary["bundle_id"],
        manifest_sha256=summary["manifest_sha256"],
    )


def _training_config(source: dict[str, Any], storage_path: str) -> dict[str, Any]:
    return {
        "data": {
            "source": source,
            "label_col": "label",
            "feature_columns": ["feature_a", "feature_b"],
        },
        "model": {
            "objective": "binary:logistic",
            "max_depth": 1,
            "eta": 0.5,
        },
        "training": {
            "num_rounds": 2,
            "val_size": 0.0,
            "test_size": 0.0,
            "max_rows_per_worker": 16,
            "seed": 7,
        },
        "ray": {
            "num_workers": 1,
            "use_gpu": False,
            "storage_path": storage_path,
            "max_failures": 0,
        },
    }


if __name__ == "__main__":
    main()
