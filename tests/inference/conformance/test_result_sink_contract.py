"""Conformance harness for credential-free ResultSink adapters."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from tributo.exceptions import ResultWriteError, UnsupportedArtifactFormat
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
    assert isinstance(sink, ResultSink)
    request = ParquetResultSinkRequest(uri="s3://results/output")
    receipt = sink.write(object(), request, run_id="run-1", plan_digest="a" * 64)
    repeated = sink.write(object(), request, run_id="run-1", plan_digest="a" * 64)
    assert receipt.sink_id == sink.sink_id
    assert receipt.uri == request.uri
    assert len(receipt.result_id) == 64
    assert "credential" not in receipt.model_dump_json()
    assert repeated == receipt


def test_fake_sink_with_natural_row_count_runs_conformance() -> None:
    sink = _FakeSink(rows_written=4)
    assert_result_sink_conformance(sink)


def test_fake_sink_without_natural_row_count_runs_conformance() -> None:
    sink = _FakeSink(rows_written=None)
    assert_result_sink_conformance(sink)


def test_sink_rejects_credential_bearing_receipt() -> None:
    with pytest.raises(ValidationError, match="credential-free"):
        ResultSinkReceipt(
            sink_id="parquet-v1",
            result_id="e" * 64,
            uri="s3://user:password@results/output",
            rows_written=1,
        )


def test_sink_failure_and_unsupported_are_distinct() -> None:
    class _FailingSink(_FakeSink):
        def write(self, dataset, request, *, run_id, plan_digest):
            del dataset, request, run_id, plan_digest
            raise ResultWriteError("classified result write failure")

    class _UnsupportedSink(_FakeSink):
        def write(self, dataset, request, *, run_id, plan_digest):
            del dataset, request, run_id, plan_digest
            raise UnsupportedArtifactFormat("classified unsupported sink")

    request = ParquetResultSinkRequest(uri="s3://results/output")
    with pytest.raises(ResultWriteError, match="write failure"):
        _FailingSink(None).write(
            object(), request, run_id="run-1", plan_digest="a" * 64
        )
    with pytest.raises(UnsupportedArtifactFormat, match="unsupported"):
        _UnsupportedSink(None).write(
            object(), request, run_id="run-1", plan_digest="a" * 64
        )


def test_sink_releases_temporary_staging_state() -> None:
    class _CleanupSink(_FakeSink):
        released_path: Path | None = None

        def write(self, dataset, request, *, run_id, plan_digest):
            del dataset, run_id, plan_digest
            with tempfile.TemporaryDirectory(prefix="sink-conformance-") as raw:
                type(self).released_path = Path(raw)
            return ResultSinkReceipt(
                sink_id=self.sink_id,
                result_id="f" * 64,
                uri=request.uri,
                rows_written=0,
            )

    sink = _CleanupSink(0)
    sink.write(
        object(),
        ParquetResultSinkRequest(uri="s3://results/output"),
        run_id="run-1",
        plan_digest="a" * 64,
    )

    assert _CleanupSink.released_path is not None
    assert not _CleanupSink.released_path.exists()


def test_first_party_parquet_sink_reuses_protocol_metadata() -> None:
    from tributo.integrations.sinks.parquet import ParquetResultSink

    assert ParquetResultSink.api_version == 1
    assert ParquetResultSink.sink_id == "parquet-v1"
