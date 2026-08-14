"""Unit and conformance tests for the distributed Lance ResultSink."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pytest
import ray.data

from tests.inference.conformance.test_result_sink_contract import (
    assert_result_sink_conformance,
)
from tributo._common.lance_write import LanceWriteConfigurationError
from tributo._common.storage_profiles import StorageProfile
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference.contracts import (
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
)
from tributo.integrations.sinks.lance import (
    LanceResultSink,
    validate_vector_batch,
)


def _vector_schema(
    *,
    dimension: int = 3,
    dtype: pa.DataType = pa.float32(),
    nullable: bool = True,
) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("vector", pa.list_(dtype, dimension), nullable=nullable),
        ]
    )


class _Dataset:
    def __init__(self, schema: pa.Schema, error: Exception | None = None) -> None:
        self._schema = schema
        self.error = error
        self.write_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.map_calls: list[tuple[object, dict[str, object]]] = []

    def schema(self) -> pa.Schema:
        return self._schema

    def map_batches(self, fn: Any, **kwargs: Any) -> _Dataset:
        self.map_calls.append((fn, kwargs))
        return self

    def write_lance(self, *args: Any, **kwargs: Any) -> None:
        self.write_calls.append((args, kwargs))
        if self.error is not None:
            raise self.error


class _Profiles:
    def resolve(self, profile: str | None) -> StorageProfile:
        del profile
        return StorageProfile(
            endpoint="http://minio:9000",
            region="us-east-1",
            access_key_id="sink-key",
            secret_access_key="sink-secret",
        )


def _request(**kwargs: Any) -> LanceResultSinkRequest:
    values: dict[str, Any] = {
        "uri": "file:///tmp/results.lance",
        "vector_columns": (LanceVectorColumnSpec(name="vector", dimension=3),),
    }
    values.update(kwargs)
    return LanceResultSinkRequest(
        **values,
    )


def test_lance_sink_runs_result_sink_conformance() -> None:
    dataset = _Dataset(_vector_schema())
    with (
        patch("tributo.integrations.sinks.lance.write_lance_dataset") as write_lance,
        patch("tributo.integrations.sinks.lance._dataset_version", return_value=7),
    ):
        assert_result_sink_conformance(
            LanceResultSink(), request=_request(), dataset=dataset
        )

    write_lance.assert_called()
    assert write_lance.call_args.kwargs["mode"] == ray.data.SaveMode.CREATE.value
    assert write_lance.call_args.kwargs["data_storage_version"] is None
    assert dataset.map_calls


def test_lance_sink_is_explicit_and_returns_schema_metadata() -> None:
    dataset = _Dataset(_vector_schema())
    request = _request(mode="overwrite", data_storage_version="2.1")
    with (
        patch("tributo.integrations.sinks.lance.write_lance_dataset") as write_lance,
        patch("tributo.integrations.sinks.lance._dataset_version", return_value=12),
    ):
        receipt = LanceResultSink().write(
            dataset, request, run_id="run-1", plan_digest="a" * 64
        )

    call = write_lance.call_args
    assert call.kwargs["uri"] == "/tmp/results.lance"
    assert call.kwargs["mode"] == ray.data.SaveMode.OVERWRITE.value
    assert call.kwargs["data_storage_version"] == "2.1"
    assert receipt.metadata["format"] == "lance"
    assert receipt.metadata["dataset_version"] == "12"
    assert len(receipt.metadata["schema_fingerprint"]) == 64


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (_vector_schema(dimension=2), "dimension"),
        (pa.schema([pa.field("vector", pa.list_(pa.float32()))]), "fixed_size_list"),
        (_vector_schema(dtype=pa.float64()), "dtype"),
        (pa.schema([pa.field("id", pa.int64())]), "missing"),
    ],
)
def test_lance_sink_rejects_invalid_declared_vector_schema(
    schema: pa.Schema, message: str
) -> None:
    with pytest.raises(ResultWriteError, match=message):
        LanceResultSink().write(
            _Dataset(schema), _request(), run_id="run-1", plan_digest="a" * 64
        )


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, None, 3.0]],
        [[1.0, float("nan"), 3.0]],
        [[1.0, float("inf"), 3.0]],
        [None],
    ],
)
def test_vector_value_validation_rejects_null_and_non_finite(
    values: list[list[float | None] | None],
) -> None:
    table = pa.table(
        {
            "vector": pa.array(values, type=pa.list_(pa.float32(), 3)),
        }
    )
    with pytest.raises(ResultWriteError, match="null or non-finite"):
        validate_vector_batch(table, _request())


def test_vector_value_validation_accepts_chunked_finite_vectors() -> None:
    vector_type = pa.list_(pa.float32(), 3)
    table = pa.table(
        {
            "vector": pa.chunked_array(
                [
                    pa.array([[1.0, 2.0, 3.0]], type=vector_type),
                    pa.array([[4.0, 5.0, 6.0]], type=vector_type),
                ]
            )
        }
    )

    assert validate_vector_batch(table, _request()) is table


def test_lance_sink_sanitizes_materialization_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset = _Dataset(
        _vector_schema(),
        RuntimeError("permission denied at /Users/example/output"),
    )
    with (
        patch(
            "tributo.integrations.sinks.lance.write_lance_dataset",
            side_effect=RuntimeError("permission denied at /Users/example/output"),
        ),
        pytest.raises(ResultMaterializationError) as error,
    ):
        LanceResultSink().write(
            dataset, _request(), run_id="run-1", plan_digest="a" * 64
        )

    assert error.value.source_error_type == "RuntimeError"
    assert error.value.__cause__ is None
    assert "RuntimeError" in caplog.text


def test_lance_sink_classifies_backend_value_error_as_materialization_failure() -> None:
    with (
        patch(
            "tributo.integrations.sinks.lance.write_lance_dataset",
            side_effect=ValueError("dataset not found at /private/result"),
        ),
        pytest.raises(ResultMaterializationError) as error,
    ):
        LanceResultSink().write(
            _Dataset(_vector_schema()),
            _request(mode="append"),
            run_id="run-1",
            plan_digest="a" * 64,
        )

    assert error.value.source_error_type == "ValueError"
    assert error.value.__cause__ is None


def test_lance_sink_classifies_writer_configuration_error_as_write_failure() -> None:
    with (
        patch(
            "tributo.integrations.sinks.lance.write_lance_dataset",
            side_effect=LanceWriteConfigurationError("unsupported storage version"),
        ),
        pytest.raises(ResultWriteError, match="unsupported storage version"),
    ):
        LanceResultSink().write(
            _Dataset(_vector_schema()),
            _request(),
            run_id="run-1",
            plan_digest="a" * 64,
        )


def test_lance_sink_uses_only_the_sink_storage_profile() -> None:
    dataset = _Dataset(_vector_schema())
    with patch(
        "tributo.integrations.sinks.lance.to_lance_storage_options",
        return_value={"endpoint": "http://minio:9000"},
    ) as options:
        with (
            patch(
                "tributo.integrations.sinks.lance.write_lance_dataset"
            ) as write_lance,
            patch("tributo.integrations.sinks.lance._dataset_version", return_value=1),
        ):
            LanceResultSink(_Profiles()).write(
                dataset,
                _request(uri="s3://bucket/results"),
                run_id="run-1",
                plan_digest="a" * 64,
            )

    options.assert_called_once()
    assert write_lance.call_args.kwargs["storage_options"] == {
        "endpoint": "http://minio:9000"
    }
