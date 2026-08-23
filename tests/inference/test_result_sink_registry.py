"""Tests for extensible inference result sink selection."""

from __future__ import annotations

from tributo.data import DataWriteTargetRequest
from tributo.inference.contracts import ParquetResultSinkRequest
from tributo.integrations.sinks.data_write import DataWriteResultSink
from tributo.integrations.sinks.parquet import ParquetResultSink
from tributo.integrations.sinks.registry import (
    ResultSinkRegistry,
    default_result_sink_registry,
)


def test_default_registry_creates_builtin_and_generic_sinks() -> None:
    registry = default_result_sink_registry()

    assert isinstance(
        registry.create(ParquetResultSinkRequest(uri="/tmp/results")),
        ParquetResultSink,
    )
    assert isinstance(
        registry.create(
            DataWriteTargetRequest(
                target_kind="doris",
                target="analytics.events",
            )
        ),
        DataWriteResultSink,
    )


def test_custom_registry_rejects_factory_id_drift() -> None:
    registry = ResultSinkRegistry()
    registry.register("custom-v1", lambda: DataWriteResultSink())

    request = ParquetResultSinkRequest(uri="/tmp/results")
    try:
        registry.create(request)
    except ValueError as exc:
        assert "No ResultSink" in str(exc)
    else:
        raise AssertionError("registry should reject an unregistered sink id")
