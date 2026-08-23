"""Unit tests for the generic data-backed inference result sink."""

from __future__ import annotations

from typing import Any

import pytest

from tributo.data import DataWriteTargetRequest
from tributo.data.writing.contracts import WriteReceipt
from tributo.exceptions import ResultWriteError
from tributo.inference.contracts import ParquetResultSinkRequest
from tributo.integrations.sinks.data_write import DataWriteResultSink


class _Gateway:
    def __init__(self) -> None:
        self.request: Any | None = None
        self.handle: Any | None = None

    def execute(self, request: Any, handle: Any) -> WriteReceipt:
        self.request = request
        self.handle = handle
        return WriteReceipt(
            request_digest=request.request_digest,
            engine_id=request.engine,
            binding_id="tributo.ray.test",
            target_kind=request.target_kind,
            target_ref=request.target,
            mode=request.mode,
            committed=True,
            rows_written=3,
        )


def test_data_write_sink_delegates_to_gateway() -> None:
    gateway = _Gateway()
    sink = DataWriteResultSink(gateway)
    request = DataWriteTargetRequest(
        target_kind="test.table",
        target="analytics.events",
        binding_id="tributo.ray.test",
    )

    receipt = sink.write(
        object(),
        request,
        run_id="run-1",
        plan_digest="a" * 64,
    )

    assert gateway.request.target_kind == "test.table"
    assert gateway.request.engine == "tributo.ray_data"
    assert gateway.handle.dataset is not None
    assert receipt.uri == "analytics.events"
    assert receipt.rows_written == 3
    assert receipt.metadata["committed"] == "true"


def test_data_write_sink_rejects_non_generic_request() -> None:
    with pytest.raises(ResultWriteError, match="cannot write"):
        DataWriteResultSink(_Gateway()).write(
            object(),
            ParquetResultSinkRequest(uri="/results"),
            run_id="run-1",
            plan_digest="a" * 64,
        )
