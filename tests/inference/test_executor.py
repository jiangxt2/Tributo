"""Unit tests for the Ray map-batches inference executor."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tributo.data import IngestionDescriptor, IngestionPlanReceipt, IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.exporting.manifest import ManifestSignature, SignatureField
from tributo.exporting.models import BundleRef
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    RayExecutionPolicy,
    ResolvedInference,
    ResolvedInputSelection,
    ResolvedModelSelection,
    ResultSinkReceipt,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.inference.executor import RayMapBatchesExecutor


def _descriptor() -> IngestionDescriptor:
    return IngestionDescriptor(
        request_digest="3" * 64,
        source_ref="4" * 64,
        dataset_ref="5" * 64,
        logical_plan_digest="6" * 64,
        engine_id="tributo.ray_data",
        provider_id="tributo.parquet",
        connector_id="parquet",
        binding_id="tributo.ray.parquet",
        scan_kind="file",
        handle_kind="ray_data",
        binding_distribution="tributo",
        binding_distribution_version="1.0.0",
        capability_version=1,
    )


def _receipt() -> IngestionPlanReceipt:
    descriptor = _descriptor()
    return IngestionPlanReceipt(
        request_digest=descriptor.request_digest,
        engine_id=descriptor.engine_id,
        engine_version="2.55.1",
        provider_id=descriptor.provider_id,
        connector_id=descriptor.connector_id,
        binding_id=descriptor.binding_id,
        scan_kind=descriptor.scan_kind,
        logical_plan_version=1,
        logical_plan_digest=descriptor.logical_plan_digest,
        source_ref=descriptor.source_ref,
        dataset_ref=descriptor.dataset_ref,
        transform_ir_version=1,
        transform_digest="7" * 64,
        binding_distribution=descriptor.binding_distribution,
        binding_distribution_version=descriptor.binding_distribution_version,
        reader_api="ray.data.read_parquet",
        transport_id="ray-data",
    )


def _plan() -> ResolvedInference:
    descriptor = _descriptor()
    ingestion_request = IngestionRequest(
        source=ParquetSourceConfig(path="/data/input"),
        engine="ray",
        binding_id=descriptor.binding_id,
    )
    return ResolvedInference(
        plan_digest="1" * 64,
        model=ResolvedModelSelection(
            bundle_ref=BundleRef(
                canonical_uri="/models/bundle",
                bundle_id="bundle-1",
                manifest_sha256="2" * 64,
            ),
            role="inference",
            flavor_id="onnx-runtime-v1",
            source_provenance="tributo-bundle",
        ),
        input=ResolvedInputSelection(
            request=ingestion_request,
            descriptor=descriptor,
        ),
        input_signature=ManifestSignature(
            input_fields=(SignatureField(name="x", dtype="float32"),)
        ),
        output_signature=ManifestSignature(
            output_fields=(SignatureField(name="score", dtype="float32"),)
        ),
        input_binding=InputBindingSpec(
            tensors=(TensorInputBinding(tensor_name="x", columns=("feature",)),)
        ),
        output_binding=OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="score", column="score", semantic="score"
                ),
            )
        ),
        result_sink=ParquetResultSinkRequest(uri="/data/output"),
        execution=RayExecutionPolicy(),
        run_id="run-1",
        attempt_id="attempt-1",
        submission_id="tributo-infer-1234567890abcdef",
    )


class _Dataset:
    def __init__(self) -> None:
        self.map_kwargs = None
        self.predicted = object()

    def map_batches(self, *args, **kwargs):
        self.map_kwargs = (args, kwargs)
        return self.predicted


class _Opened:
    def __init__(self, dataset, *, close_error: Exception | None = None) -> None:
        self.dataset = dataset
        self.receipt = _receipt()
        self.close_error = close_error
        self.close_calls = 0
        self.cancel_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error

    def cancel(self) -> None:
        self.cancel_calls += 1


class _Inputs:
    def __init__(
        self,
        dataset=None,
        error: Exception | None = None,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.error = error
        self.calls = []
        self.opened = _Opened(dataset, close_error=close_error)

    def open(self, selection):
        self.calls.append(selection)
        if self.error is not None:
            raise self.error
        return self.opened


class _Sink:
    api_version = 1
    sink_id = "parquet-v1"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    def write(self, dataset, request, *, run_id, plan_digest):
        self.calls.append((dataset, request, run_id, plan_digest))
        if self.error is not None:
            raise self.error
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id="8" * 64,
            uri=request.uri,
            rows_written=None,
        )


class TestRayMapBatchesExecutor:
    def test_success_builds_one_lazy_graph_and_writes_through_sink(self) -> None:
        plan = _plan()
        dataset = _Dataset()
        inputs = _Inputs(dataset)
        sink = _Sink()
        strategy = object()

        with patch("ray.data.ActorPoolStrategy", return_value=strategy) as actor_pool:
            result = RayMapBatchesExecutor(inputs).execute(plan, sink)

        assert result.status == "succeeded"
        assert result.input_rows is None
        assert result.output_rows is None
        assert result.ingestion_receipt == _receipt()
        assert result.sink_receipt is not None
        assert inputs.calls == [plan.input]
        assert inputs.opened.close_calls == 1
        assert inputs.opened.cancel_calls == 0
        args, kwargs = dataset.map_kwargs
        assert args[0].__name__ == "BundleBatchPredictor"
        assert kwargs["batch_format"] == "numpy"
        assert kwargs["compute"] is strategy
        assert sink.calls[0][0] is dataset.predicted
        actor_pool.assert_called_once_with(size=plan.execution.concurrency)

    def test_acquisition_failure_is_structured(self) -> None:
        result = RayMapBatchesExecutor(
            _Inputs(error=RuntimeError("credential detail"))
        ).execute(_plan(), _Sink())

        assert result.status == "failed"
        assert result.failure is not None
        assert result.failure.phase == "acquisition"
        assert result.failure.error_type == "RuntimeError"
        assert result.ingestion_receipt is None
        assert "credential detail" not in result.model_dump_json()

    @pytest.mark.parametrize(
        ("error", "phase", "error_type"),
        [
            (ResultWriteError("secret path"), "sink", "ResultWriteError"),
            (RuntimeError("provider detail"), "sink", "RuntimeError"),
            (
                ResultMaterializationError("RayTaskError"),
                "materialization",
                "RayTaskError",
            ),
        ],
    )
    def test_terminal_failure_is_structured_and_cancels_input(
        self, error: Exception, phase: str, error_type: str
    ) -> None:
        inputs = _Inputs(_Dataset())

        result = RayMapBatchesExecutor(inputs).execute(_plan(), _Sink(error))

        assert result.failure is not None
        assert result.failure.phase == phase
        assert result.failure.error_type == error_type
        assert result.ingestion_receipt == _receipt()
        assert inputs.opened.cancel_calls == 1
        assert inputs.opened.close_calls == 0
        assert "secret path" not in result.model_dump_json()
        assert "provider detail" not in result.model_dump_json()

    def test_map_construction_failure_is_execution_failure_and_cancels(self) -> None:
        class BrokenDataset:
            def map_batches(self, *args, **kwargs):
                raise ValueError("bad execution")

        inputs = _Inputs(BrokenDataset())
        result = RayMapBatchesExecutor(inputs).execute(_plan(), _Sink())

        assert result.failure is not None
        assert result.failure.phase == "execution"
        assert inputs.opened.cancel_calls == 1

    def test_close_failure_preserves_committed_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        inputs = _Inputs(_Dataset(), close_error=RuntimeError("cleanup detail"))

        with caplog.at_level(logging.WARNING, logger="tributo.inference.executor"):
            result = RayMapBatchesExecutor(inputs).execute(_plan(), _Sink())

        assert result.status == "succeeded"
        assert result.failure is None
        assert result.retryable is False
        assert result.ingestion_receipt == _receipt()
        assert result.sink_receipt is not None
        assert inputs.opened.close_calls == 1
        assert "RuntimeError" in caplog.text
        assert "cleanup detail" not in caplog.text

    def test_mismatched_sink_id_fails_before_data_access(self) -> None:
        sink = _Sink()
        sink.sink_id = "other-v1"
        inputs = _Inputs(_Dataset())

        with pytest.raises(ValueError, match="cannot write"):
            RayMapBatchesExecutor(inputs).execute(_plan(), sink)

    def test_post_resolution_credential_mutation_fails_before_data_access(
        self,
    ) -> None:
        plan = _plan()
        plan.input.request.source.path = "s3://user:must-not-leak@bucket/input"
        inputs = _Inputs(_Dataset())

        with pytest.raises(ValidationError, match="plaintext credentials") as error:
            RayMapBatchesExecutor(inputs).execute(plan, _Sink())

        assert inputs.calls == []
        assert "must-not-leak" not in str(error.value)

        assert inputs.calls == []
