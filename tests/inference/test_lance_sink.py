"""Unit and conformance tests for the distributed Lance ResultSink."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pytest

from tests.inference.conformance.test_result_sink_contract import (
    assert_result_sink_conformance,
)
from tributo._common.storage_profiles import StorageProfile
from tributo.data.base import WriteMode
from tributo.data.writing.contracts import WriteBindingError, WriteCapabilityError
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
        self.map_calls: list[tuple[object, dict[str, object]]] = []

    def schema(self) -> pa.Schema:
        return self._schema

    def map_batches(self, fn: Any, **kwargs: Any) -> _Dataset:
        self.map_calls.append((fn, kwargs))
        return self


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
    with patch(
        "tributo.integrations.sinks.lance.default_write_gateway"
    ) as default_gateway:
        assert_result_sink_conformance(
            LanceResultSink(), request=_request(), dataset=dataset
        )

    request, handle = default_gateway.return_value.execute.call_args.args
    assert request.binding_id == "tributo.ray.lance"
    assert request.mode == WriteMode.CREATE
    assert "data_storage_version" not in request.options
    assert handle.dataset is dataset
    assert dataset.map_calls


def test_lance_sink_is_explicit_and_returns_schema_metadata() -> None:
    dataset = _Dataset(_vector_schema())
    request = _request(mode="overwrite", data_storage_version="2.1")
    with patch(
        "tributo.integrations.sinks.lance.default_write_gateway"
    ) as default_gateway:
        receipt = LanceResultSink().write(
            dataset, request, run_id="run-1", plan_digest="a" * 64
        )

    gateway_request, handle = default_gateway.return_value.execute.call_args.args
    assert gateway_request.target == "file:///tmp/results.lance"
    assert gateway_request.mode == WriteMode.OVERWRITE
    assert gateway_request.options["data_storage_version"] == "2.1"
    assert handle.dataset is dataset
    assert receipt.metadata["format"] == "lance"
    assert "dataset_version" not in receipt.metadata
    assert receipt.metadata["data_storage_version"] == "2.1"
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
            "tributo.integrations.sinks.lance.default_write_gateway"
        ) as default_gateway,
        pytest.raises(ResultMaterializationError) as error,
    ):
        default_gateway.return_value.execute.side_effect = RuntimeError(
            "permission denied at /Users/example/output"
        )
        LanceResultSink().write(
            dataset, _request(), run_id="run-1", plan_digest="a" * 64
        )

    assert error.value.source_error_type == "RuntimeError"
    assert error.value.__cause__ is None
    assert "RuntimeError" in caplog.text


def test_lance_sink_classifies_backend_value_error_as_materialization_failure() -> None:
    with (
        patch(
            "tributo.integrations.sinks.lance.default_write_gateway"
        ) as default_gateway,
        pytest.raises(ResultMaterializationError) as error,
    ):
        default_gateway.return_value.execute.side_effect = ValueError(
            "dataset not found at /private/result"
        )
        LanceResultSink().write(
            _Dataset(_vector_schema()),
            _request(mode="append"),
            run_id="run-1",
            plan_digest="a" * 64,
        )

    assert error.value.source_error_type == "ValueError"
    assert error.value.__cause__ is None


def test_lance_sink_classifies_gateway_capability_error_as_sink_failure() -> None:
    with (
        patch(
            "tributo.integrations.sinks.lance.default_write_gateway"
        ) as default_gateway,
        pytest.raises(ResultWriteError) as error,
    ):
        default_gateway.return_value.execute.side_effect = WriteCapabilityError(
            "missing binding at /private/result"
        )
        LanceResultSink().write(
            _Dataset(_vector_schema()),
            _request(),
            run_id="run-1",
            plan_digest="a" * 64,
        )

    assert str(error.value) == (
        "Lance result sink cannot satisfy the requested write capability"
    )
    assert error.value.__cause__ is None


def test_lance_sink_preserves_native_source_error_classification() -> None:
    with (
        patch(
            "tributo.integrations.sinks.lance.default_write_gateway"
        ) as default_gateway,
        pytest.raises(ResultMaterializationError) as error,
    ):
        default_gateway.return_value.execute.side_effect = WriteBindingError(
            "native write failed", source_error_type="ValueError"
        )
        LanceResultSink().write(
            _Dataset(_vector_schema()),
            _request(),
            run_id="run-1",
            plan_digest="a" * 64,
        )
    assert error.value.source_error_type == "ValueError"


def test_lance_sink_uses_only_the_sink_storage_profile() -> None:
    dataset = _Dataset(_vector_schema())
    with patch(
        "tributo.integrations.sinks.lance.default_write_gateway"
    ) as default_gateway:
        LanceResultSink(_Profiles()).write(
            dataset,
            _request(uri="s3://bucket/results"),
            run_id="run-1",
            plan_digest="a" * 64,
        )

    gateway_request = default_gateway.return_value.execute.call_args.args[0]
    profile = gateway_request.runtime_options["s3"]
    assert profile.endpoint == "http://minio:9000"
    assert profile.access_key_id == "sink-key"
    assert "sink-key" not in gateway_request.model_dump_json()
    assert "sink-secret" not in gateway_request.model_dump_json()
