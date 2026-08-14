"""Tests for the first-phase custom Predictor boundary."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from tributo.data.source_config import ParquetSourceConfig
from tributo.inference.base import BasePredictor
from tributo.inference.contracts import (
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
    ResultSinkReceipt,
)
from tributo.inference.pipeline import InferenceConfig, run_batch_inference


class _CustomPredictor(BasePredictor):
    def _load_model(self) -> None:
        self.model = object()

    def __call__(self, batch: Any) -> Any:
        return batch


class _Opened:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def close(self) -> None:
        pass

    def cancel(self) -> None:
        pass


def _run_with_config(config: InferenceConfig) -> tuple[Any, Any, Any]:
    dataset = MagicMock()
    dataset.map_batches.return_value = object()
    with (
        patch("ray.data.ActorPoolStrategy"),
        patch(
            "tributo.inference.input_resolver.IngestionGatewayInputResolver.describe",
            return_value=object(),
        ),
        patch(
            "tributo.inference.input_resolver.IngestionGatewayInputResolver.open",
            return_value=_Opened(dataset),
        ),
        patch(
            "tributo.integrations.sinks.parquet.ParquetResultSink.write",
            return_value=ResultSinkReceipt(
                sink_id="parquet-v1",
                result_id="a" * 64,
                uri=config.output_uri,
            ),
        ) as write,
    ):
        result = run_batch_inference(config, predictor_cls=_CustomPredictor)
    return result, dataset, write


def test_custom_predictor_non_bundle_constructor_contract() -> None:
    config = InferenceConfig(
        source=ParquetSourceConfig(path="data.parquet", columns=["feature"]),
        output_uri="output",
        model_uri="model.onnx",
    )

    result, dataset, _ = _run_with_config(config)

    assert result["status"] == "completed"
    args = dataset.map_batches.call_args
    assert args.args[0] is _CustomPredictor
    assert args.kwargs["fn_constructor_args"] == (
        "model.onnx",
        {"feature_names": ["feature"]},
    )


def test_custom_predictor_bundle_constructor_contract_is_explicit() -> None:
    config = InferenceConfig(
        source=ParquetSourceConfig(path="data.parquet", columns=["feature"]),
        output_uri="output",
        bundle_uri="bundle",
    )

    _, dataset, _ = _run_with_config(config)

    args = dataset.map_batches.call_args
    assert args.args[0] is _CustomPredictor
    assert args.kwargs["fn_constructor_args"] == (
        None,
        {"feature_names": ["feature"]},
        "bundle",
        "inference",
        False,
        None,
    )


def test_batch_pipeline_constructs_explicit_lance_sink() -> None:
    config = InferenceConfig(
        source=ParquetSourceConfig(path="data.parquet", columns=["feature"]),
        output_uri="output.lance",
        model_uri="model.onnx",
        output_format="lance",
        output_mode="append",
        output_data_storage_version="2.1",
        output_vector_columns=[
            LanceVectorColumnSpec(name="vector", dimension=3),
        ],
    )
    dataset = MagicMock()
    dataset.map_batches.return_value = object()
    with (
        patch("ray.data.ActorPoolStrategy"),
        patch(
            "tributo.inference.input_resolver.IngestionGatewayInputResolver.describe",
            return_value=object(),
        ),
        patch(
            "tributo.inference.input_resolver.IngestionGatewayInputResolver.open",
            return_value=_Opened(dataset),
        ),
        patch(
            "tributo.integrations.sinks.lance.LanceResultSink.write",
            return_value=ResultSinkReceipt(
                sink_id="lance-v1",
                result_id="b" * 64,
                uri=config.output_uri,
            ),
        ) as write,
    ):
        run_batch_inference(config, predictor_cls=_CustomPredictor)

    request = write.call_args.args[1]
    assert isinstance(request, LanceResultSinkRequest)
    assert request.mode == "append"
    assert request.data_storage_version == "2.1"
    assert request.vector_columns[0].dimension == 3
