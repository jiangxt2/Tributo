"""Conformance harness for credential-free ResultSink adapters."""

from __future__ import annotations

from tributo.inference.contracts import (
    ParquetResultSinkRequest,
    ResultSink,
    ResultSinkReceipt,
)


class _FakeSink:
    api_version = 1
    sink_id = "parquet-v1"

    def __init__(self, rows_written: int | None) -> None:
        self.rows_written = rows_written

    def write(self, dataset, request, *, run_id, plan_digest):
        del dataset, run_id, plan_digest
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id="c" * 64,
            uri=request.uri,
            rows_written=self.rows_written,
        )


def assert_result_sink_conformance(sink: ResultSink) -> None:
    request = ParquetResultSinkRequest(uri="s3://results/output")
    receipt = sink.write(object(), request, run_id="run-1", plan_digest="a" * 64)
    assert receipt.sink_id == sink.sink_id
    assert receipt.uri == request.uri
    assert len(receipt.result_id) == 64
    assert "credential" not in receipt.model_dump_json()


def test_fake_sink_with_natural_row_count_runs_conformance() -> None:
    sink = _FakeSink(rows_written=4)
    assert_result_sink_conformance(sink)


def test_fake_sink_without_natural_row_count_runs_conformance() -> None:
    sink = _FakeSink(rows_written=None)
    assert_result_sink_conformance(sink)
