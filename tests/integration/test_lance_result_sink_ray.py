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
from tributo.data.base import WriteMode
from tributo.data.lance import LanceDataConnector
from tributo.exceptions import ResultMaterializationError
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

        append = sink.write(
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
        assert int(append.metadata["dataset_version"]) > int(
            create.metadata["dataset_version"]
        )
        assert lance.dataset(output_uri).to_table().num_rows == 3

        overwrite = sink.write(
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
        assert int(overwrite.metadata["dataset_version"]) > int(
            append.metadata["dataset_version"]
        )
        assert lance.dataset(output_uri).to_table()["id"].to_pylist() == [9]

    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_connector_uses_the_same_distributed_writer() -> None:
    output = Path("/workspace/tributo-work") / f"lance-connector-{uuid.uuid4().hex}"
    output_uri = str(output)
    try:
        dataset = _predict(_dataset(1, 3))
        LanceDataConnector().write(
            dataset,
            path=output_uri,
            mode=WriteMode.CREATE,
            min_rows_per_file=1,
            max_rows_per_file=10,
        )
        import lance

        table = lance.dataset(output_uri).to_table()
        assert table.num_rows == 2
        assert table.schema.field("vector").type == pa.list_(pa.float32(), 2)
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_connector_always_writes_lance_for_non_vector_data() -> None:
    output = Path("/workspace/tributo-work") / f"lance-plain-{uuid.uuid4().hex}"
    output_uri = str(output)
    try:
        LanceDataConnector().write(
            _dataset(1, 3),
            path=output_uri,
            mode=WriteMode.CREATE,
            min_rows_per_file=1,
            max_rows_per_file=10,
        )
        import lance

        table = lance.dataset(output_uri).to_table()
        assert table.column_names == ["id", "value"]
        assert table.num_rows == 2
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_sink_materializes_empty_inputs_and_preserves_append_version() -> None:
    output = Path("/workspace/tributo-work") / f"lance-empty-{uuid.uuid4().hex}"
    output_uri = str(output)
    empty = ray.data.from_arrow(
        pa.table(
            {
                "id": pa.array([], type=pa.int64()),
                "vector": pa.array([], type=pa.list_(pa.float32(), 2)),
            }
        )
    )
    request = LanceResultSinkRequest(
        uri=output_uri,
        min_rows_per_file=1,
        vector_columns=(_vector_spec(),),
    )
    try:
        import lance

        sink = LanceResultSink()
        created = sink.write(
            empty,
            request,
            run_id="run-empty-create",
            plan_digest="d" * 64,
        )
        assert lance.dataset(output_uri).to_table().num_rows == 0

        appended = sink.write(
            empty,
            request.model_copy(update={"mode": "append"}),
            run_id="run-empty-append",
            plan_digest="e" * 64,
        )
        assert (
            appended.metadata["dataset_version"] == created.metadata["dataset_version"]
        )

        overwritten = sink.write(
            empty,
            request.model_copy(update={"mode": "overwrite"}),
            run_id="run-empty-overwrite",
            plan_digest="f" * 64,
        )
        assert int(overwritten.metadata["dataset_version"]) > int(
            created.metadata["dataset_version"]
        )
        assert lance.dataset(output_uri).to_table().num_rows == 0
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_sink_overwrite_supports_schema_evolution() -> None:
    output = Path("/workspace/tributo-work") / f"lance-schema-{uuid.uuid4().hex}"
    output_uri = str(output)
    try:
        import lance

        sink = LanceResultSink()
        sink.write(
            ray.data.from_arrow(pa.table({"id": pa.array([1], type=pa.int64())})),
            LanceResultSinkRequest(
                uri=output_uri,
                min_rows_per_file=1,
            ),
            run_id="run-schema-create",
            plan_digest="1" * 64,
        )
        sink.write(
            ray.data.from_arrow(
                pa.table(
                    {
                        "id": pa.array([2], type=pa.int64()),
                        "extra": pa.array(["evolved"]),
                    }
                )
            ),
            LanceResultSinkRequest(
                uri=output_uri,
                mode="overwrite",
                min_rows_per_file=1,
            ),
            run_id="run-schema-overwrite",
            plan_digest="2" * 64,
        )

        table = lance.dataset(output_uri).to_table()
        assert table.column_names == ["id", "extra"]
        assert table["extra"].to_pylist() == ["evolved"]
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_sink_create_rejects_existing_dataset() -> None:
    output = Path("/workspace/tributo-work") / f"lance-conflict-{uuid.uuid4().hex}"
    output_uri = str(output)
    dataset = ray.data.from_arrow(pa.table({"id": pa.array([1], type=pa.int64())}))
    request = LanceResultSinkRequest(uri=output_uri, min_rows_per_file=1)
    try:
        sink = LanceResultSink()
        sink.write(dataset, request, run_id="run-conflict-1", plan_digest="3" * 64)
        with pytest.raises(ResultMaterializationError):
            sink.write(
                dataset,
                request,
                run_id="run-conflict-2",
                plan_digest="4" * 64,
            )
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_sink_append_rejects_a_missing_dataset() -> None:
    output = Path("/workspace/tributo-work") / f"lance-missing-{uuid.uuid4().hex}"
    output_uri = str(output)
    dataset = ray.data.from_arrow(pa.table({"id": pa.array([1], type=pa.int64())}))
    try:
        with pytest.raises(ResultMaterializationError):
            LanceResultSink().write(
                dataset,
                LanceResultSinkRequest(
                    uri=output_uri,
                    mode="append",
                    min_rows_per_file=1,
                ),
                run_id="run-missing-append",
                plan_digest="5" * 64,
            )
        assert not output.exists()
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_lance_writer_honors_minimum_rows_per_file() -> None:
    output = Path("/workspace/tributo-work") / f"lance-files-{uuid.uuid4().hex}"
    output_uri = str(output)
    try:
        LanceDataConnector().write(
            _dataset(0, 5),
            path=output_uri,
            mode=WriteMode.CREATE,
            min_rows_per_file=2,
            max_rows_per_file=3,
        )
        import lance

        fragments = list(lance.dataset(output_uri).get_fragments())
        assert sorted(fragment.count_rows() for fragment in fragments) == [1, 2, 2]
    finally:
        shutil.rmtree(output, ignore_errors=True)
