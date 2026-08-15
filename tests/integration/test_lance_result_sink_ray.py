"""Ray-cluster verification for custom Predictors and Lance result writes."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pytest
import ray
import ray.data

from tests.support.lance_predictor import HFLikePredictor
from tributo.inference.contracts import (
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
)
from tributo.integrations.sinks.lance import LanceResultSink

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_inference_compose() -> None:
    if os.environ.get("TRIBUTO_DOCKER_INFERENCE_IT") != "1":
        pytest.fail("Lance Ray integration must run through the inference Docker suite")


def _vector_spec() -> LanceVectorColumnSpec:
    return LanceVectorColumnSpec(name="vector", dimension=2)


def _dataset(start: int, stop: int) -> ray.data.Dataset:
    return ray.data.from_items(
        [{"id": index, "value": float(index)} for index in range(start, stop)]
    )


def _predict(dataset: ray.data.Dataset) -> ray.data.Dataset:
    return dataset.map_batches(
        HFLikePredictor,
        fn_constructor_args=("hf-like-model", {"bias": 1.0}),
        batch_format="pyarrow",
        batch_size=2,
        compute=ray.data.ActorPoolStrategy(size=1),
        num_cpus=1,
    )


def test_custom_predictor_and_lance_create_append_overwrite() -> None:
    output = Path("/workspace/tributo-work") / f"lance-sink-{uuid.uuid4().hex}"
    output_uri = str(output)
    try:
        sink = LanceResultSink()
        create = sink.write(
            _predict(_dataset(1, 3)),
            LanceResultSinkRequest(
                uri=output_uri,
                mode="create",
                min_rows_per_file=1,
                vector_columns=(_vector_spec(),),
            ),
            run_id="run-create",
            plan_digest="a" * 64,
        )
        import lance

        table = lance.dataset(output_uri).to_table()
        assert create.metadata["format"] == "lance"
        assert table.num_rows == 2
        assert table.schema.field("vector").type == pa.list_(pa.float32(), 2)
        nearest = lance.dataset(output_uri).to_table(
            nearest={
                "column": "vector",
                "q": pa.array([2.0, 2.0], type=pa.float32()),
                "k": 1,
                "metric": "L2",
            }
        )
        assert nearest["id"].to_pylist() == [1]

        sink.write(
            _predict(_dataset(3, 4)),
            LanceResultSinkRequest(
                uri=output_uri,
                mode="append",
                min_rows_per_file=1,
                vector_columns=(_vector_spec(),),
            ),
            run_id="run-append",
            plan_digest="b" * 64,
        )
        assert lance.dataset(output_uri).to_table().num_rows == 3

        sink.write(
            _predict(_dataset(9, 10)),
            LanceResultSinkRequest(
                uri=output_uri,
                mode="overwrite",
                min_rows_per_file=1,
                vector_columns=(_vector_spec(),),
            ),
            run_id="run-overwrite",
            plan_digest="c" * 64,
        )
        assert lance.dataset(output_uri).to_table()["id"].to_pylist() == [9]

    finally:
        shutil.rmtree(output, ignore_errors=True)
