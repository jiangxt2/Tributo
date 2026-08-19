"""Unit and conformance tests for the distributed Lance ResultSink."""

from __future__ import annotations

import subprocess
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest
from ray.air.util.tensor_extensions.arrow import ArrowTensorType, ArrowTensorTypeV2

from tests.inference.conformance.test_result_sink_contract import (
    assert_result_sink_conformance,
)
from tributo._common.storage_profiles import StorageProfile
from tributo.data.base import WriteMode
from tributo.data.refs import schema_fingerprint
from tributo.data.writing.contracts import WriteBindingError, WriteCapabilityError
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference.contracts import (
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
)
from tributo.integrations.sinks.lance import (
    LanceResultSink,
    _canonical_vector_schema,
    _normalize_vector_batch,
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


def test_lance_sink_module_imports_in_fresh_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tributo.integrations.sinks.lance import LanceResultSink",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


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


def test_lance_sink_fingerprints_normalized_ray_tensor_schema() -> None:
    source_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field(
                "vector",
                ArrowTensorTypeV2((3,), pa.float32()),
                metadata={b"role": b"embedding"},
            ),
        ],
        metadata={b"dataset": b"inference"},
    )
    dataset = _Dataset(source_schema)
    with patch(
        "tributo.integrations.sinks.lance.default_write_gateway"
    ) as default_gateway:
        receipt = LanceResultSink().write(
            dataset, _request(), run_id="run-1", plan_digest="a" * 64
        )

    target_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field(
                "vector",
                pa.list_(pa.float32(), 3),
                metadata={b"role": b"embedding"},
            ),
        ],
        metadata={b"dataset": b"inference"},
    )
    assert receipt.metadata["schema_fingerprint"] == schema_fingerprint(target_schema)
    _, handle = default_gateway.return_value.execute.call_args.args
    assert handle.dataset is dataset
    assert dataset.map_calls[0][1]["batch_format"] == "pyarrow"


def test_lance_sink_without_declared_vectors_preserves_existing_path() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("payload", pa.list_(pa.float32())),
        ]
    )
    dataset = _Dataset(schema)
    with patch(
        "tributo.integrations.sinks.lance.default_write_gateway"
    ) as default_gateway:
        receipt = LanceResultSink().write(
            dataset,
            _request(vector_columns=()),
            run_id="run-1",
            plan_digest="a" * 64,
        )

    assert dataset.map_calls == []
    assert receipt.metadata["schema_fingerprint"] == schema_fingerprint(schema)
    _, handle = default_gateway.return_value.execute.call_args.args
    assert handle.dataset is dataset


@pytest.mark.parametrize(
    "data_type",
    [
        ArrowTensorType((3,), pa.float32()),
        ArrowTensorTypeV2((3,), pa.float32()),
        pa.fixed_shape_tensor(pa.float32(), (3,)),
        pa.list_(pa.float32(), 3),
    ],
)
def test_vector_schema_accepts_supported_fixed_shape_types(
    data_type: pa.DataType,
) -> None:
    source_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field(
                "vector",
                data_type,
                nullable=False,
                metadata={b"role": b"embedding"},
            ),
        ],
        metadata={b"dataset": b"inference"},
    )

    target_schema = _canonical_vector_schema(source_schema, _request())

    assert target_schema.names == source_schema.names
    vector_field = target_schema.field("vector")
    assert vector_field.type == pa.list_(pa.float32(), 3)
    assert vector_field.nullable is False
    assert vector_field.metadata == {b"role": b"embedding"}
    assert target_schema.metadata == {b"dataset": b"inference"}


@pytest.mark.parametrize("representation", ["ray-v1", "ray-v2", "arrow-native"])
def test_vector_batch_normalizes_fixed_shape_tensor_and_preserves_values(
    representation: str,
) -> None:
    values = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    if representation == "arrow-native":
        tensor = pa.FixedShapeTensorArray.from_numpy_ndarray(values)
    else:
        tensor_type = (
            ArrowTensorType((3,), pa.float32())
            if representation == "ray-v1"
            else ArrowTensorTypeV2((3,), pa.float32())
        )
        tensor = pa.ExtensionArray.from_storage(
            tensor_type,
            pa.array(values.tolist(), type=tensor_type.storage_type),
        )
    batch = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "vector": tensor,
        }
    )
    target_schema = _canonical_vector_schema(batch.schema, _request())

    normalized = _normalize_vector_batch(
        batch,
        request=_request(),
        target_schema=target_schema,
    )

    assert normalized.schema == target_schema
    assert normalized.column("vector").to_pylist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]


def test_vector_batch_rejects_source_dtype_drift_before_cast() -> None:
    target_schema = _vector_schema()
    batch = pa.table(
        {
            "id": pa.array([1], type=pa.int64()),
            "vector": pa.array(
                [[1.0, 2.0, 3.0]],
                type=pa.list_(pa.float64(), 3),
            ),
        }
    )

    with pytest.raises(ResultWriteError, match="dtype"):
        _normalize_vector_batch(
            batch,
            request=_request(),
            target_schema=target_schema,
        )


def test_vector_batch_rejects_source_rank_drift_before_cast() -> None:
    tensor_type = ArrowTensorTypeV2((1, 3), pa.float32())
    batch = pa.table(
        {
            "id": pa.array([], type=pa.int64()),
            "vector": pa.ExtensionArray.from_storage(
                tensor_type,
                pa.array([], type=tensor_type.storage_type),
            ),
        }
    )

    with pytest.raises(ResultWriteError, match="shape"):
        _normalize_vector_batch(
            batch,
            request=_request(),
            target_schema=_vector_schema(),
        )


@pytest.mark.parametrize(
    ("batch_schema", "message"),
    [
        (
            pa.schema(
                [
                    pa.field("id", pa.int32(), nullable=False),
                    pa.field("vector", pa.list_(pa.float32(), 3)),
                ]
            ),
            "'id'.*int32.*int64",
        ),
        (
            pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=True),
                    pa.field("vector", pa.list_(pa.float32(), 3)),
                ]
            ),
            "'id'.*nullable=True.*nullable=False",
        ),
        (
            pa.schema(
                [
                    pa.field(
                        "id", pa.int64(), nullable=False, metadata={b"role": b"key"}
                    ),
                    pa.field("vector", pa.list_(pa.float32(), 3)),
                ]
            ),
            "'id' metadata differs",
        ),
        (
            pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("vector", pa.list_(pa.float32(), 3)),
                ],
                metadata={b"dataset": b"worker"},
            ),
            "schema metadata differs",
        ),
        (
            pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), 3)),
                    pa.field("id", pa.int64(), nullable=False),
                ]
            ),
            "field names or order",
        ),
    ],
)
def test_vector_batch_rejects_non_vector_schema_drift_before_cast(
    batch_schema: pa.Schema, message: str
) -> None:
    values = {
        "id": pa.array([1], type=batch_schema.field("id").type),
        "vector": pa.array(
            [[1.0, 2.0, 3.0]],
            type=pa.list_(pa.float32(), 3),
        ),
    }
    batch = pa.Table.from_arrays(
        [values[field.name] for field in batch_schema],
        schema=batch_schema,
    )

    with pytest.raises(ResultWriteError, match=message):
        _normalize_vector_batch(
            batch,
            request=_request(),
            target_schema=_vector_schema(),
        )


def test_vector_batch_cast_failure_retains_only_safe_detail() -> None:
    source_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("vector", ArrowTensorTypeV2((3,), pa.float32())),
        ]
    )
    target_schema = _canonical_vector_schema(source_schema, _request())

    class _FailingCastBatch:
        schema = source_schema

        def cast(self, schema: pa.Schema) -> pa.Table:
            del schema
            raise pa.ArrowInvalid("field vector failed; token=secret-value")

    batch: Any = _FailingCastBatch()
    with pytest.raises(ResultWriteError) as exc_info:
        _normalize_vector_batch(
            batch,
            request=_request(),
            target_schema=target_schema,
        )

    message = str(exc_info.value)
    assert "ArrowInvalid: field vector failed" in message
    assert "token=<redacted>" in message
    assert "secret-value" not in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (_vector_schema(dimension=2), "dimension"),
        (pa.schema([pa.field("vector", pa.list_(pa.float32()))]), "fixed_size_list"),
        (_vector_schema(dtype=pa.float64()), "dtype"),
        (
            pa.schema([pa.field("vector", ArrowTensorTypeV2((1, 3), pa.float32()))]),
            "shape",
        ),
        (
            pa.schema([pa.field("vector", ArrowTensorTypeV2((3,), pa.float64()))]),
            "dtype",
        ),
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
