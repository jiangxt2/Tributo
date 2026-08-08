"""Cluster job exercising the inline post-training inference adapter."""

from __future__ import annotations

import json
import os

import pyarrow.dataset as pads
import ray

from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.exporting.models import BundleRef
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
        BundleRef(
            canonical_uri=os.environ["BUNDLE_URI"],
            bundle_id=os.environ["BUNDLE_ID"],
            manifest_sha256=os.environ["MANIFEST_SHA256"],
        ),
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
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
