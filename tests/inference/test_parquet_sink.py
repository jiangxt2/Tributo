"""Unit tests for the public-Ray-API Parquet ResultSink."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tributo._common.storage_profiles import StorageProfile
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference.contracts import ParquetResultSinkRequest
from tributo.integrations.sinks.parquet import ParquetResultSink


class _Dataset:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []
        self.count_calls = 0

    def write_parquet(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error

    def count(self) -> int:
        self.count_calls += 1
        raise AssertionError("ResultSink must not call Dataset.count()")


class _Profiles:
    def __init__(self) -> None:
        self.calls = []

    def resolve(self, profile: str | None) -> StorageProfile:
        self.calls.append(profile)
        return StorageProfile(
            endpoint="http://minio:9000",
            region="us-east-1",
            access_key_id="sink-key",
            secret_access_key="sink-secret",
        )


class TestParquetResultSink:
    def test_local_write_has_optional_row_count_without_count_job(self) -> None:
        dataset = _Dataset()
        sink = ParquetResultSink()
        request = ParquetResultSinkRequest(
            uri="file:///tmp/results",
            compression="snappy",
            min_rows_per_file=10,
        )

        first = sink.write(dataset, request, run_id="run-1", plan_digest="a" * 64)
        second = sink.write(_Dataset(), request, run_id="run-1", plan_digest="a" * 64)

        assert dataset.calls == [
            (("/tmp/results",), {"compression": "snappy", "min_rows_per_file": 10})
        ]
        assert dataset.count_calls == 0
        assert first.rows_written is None
        assert first.result_id == second.result_id
        assert first.metadata == {"format": "parquet", "compression": "snappy"}

    def test_file_uri_is_decoded_without_treating_localhost_as_a_path(self) -> None:
        dataset = _Dataset()

        ParquetResultSink().write(
            dataset,
            ParquetResultSinkRequest(uri="file://localhost/tmp/result%20set"),
            run_id="run-1",
            plan_digest="a" * 64,
        )

        assert dataset.calls[0][0] == ("/tmp/result set",)

    def test_s3_write_resolves_only_sink_profile(self) -> None:
        dataset = _Dataset()
        profiles = _Profiles()
        filesystem = object()
        request = ParquetResultSinkRequest(
            uri="s3://bucket/results", storage_profile="sink-domain"
        )

        with patch("pyarrow.fs.S3FileSystem", return_value=filesystem) as factory:
            receipt = ParquetResultSink(profiles).write(
                dataset, request, run_id="run-1", plan_digest="a" * 64
            )

        assert profiles.calls == ["sink-domain"]
        assert dataset.calls[0][0] == ("bucket/results",)
        assert dataset.calls[0][1]["filesystem"] is filesystem
        assert receipt.uri == "s3://bucket/results"
        factory.assert_called_once()

    def test_s3_scheme_is_case_insensitive(self) -> None:
        dataset = _Dataset()
        profiles = _Profiles()
        filesystem = object()

        with patch("pyarrow.fs.S3FileSystem", return_value=filesystem):
            receipt = ParquetResultSink(profiles).write(
                dataset,
                ParquetResultSinkRequest(
                    uri="S3://bucket/results", storage_profile="sink-domain"
                ),
                run_id="run-1",
                plan_digest="a" * 64,
            )

        assert profiles.calls == ["sink-domain"]
        assert dataset.calls[0][0] == ("bucket/results",)
        assert dataset.calls[0][1]["filesystem"] is filesystem
        assert receipt.uri == "S3://bucket/results"

    def test_write_error_is_sanitized(self) -> None:
        dataset = _Dataset(RuntimeError("s3://key:secret@bucket/output"))

        with pytest.raises(ResultMaterializationError) as error:
            ParquetResultSink().write(
                dataset,
                ParquetResultSinkRequest(uri="/output"),
                run_id="run-1",
                plan_digest="a" * 64,
            )

        assert error.value.source_error_type == "RuntimeError"
        assert "secret" not in str(error.value)
        assert error.value.__cause__ is None

    def test_unresolved_named_profile_cannot_fall_back_to_default_domain(self) -> None:
        class NamedProfiles:
            def resolve(self, profile: str | None) -> StorageProfile:
                return StorageProfile(profile_name=profile)

        with pytest.raises(ResultWriteError, match="resolve inside the cluster"):
            ParquetResultSink(NamedProfiles()).write(
                _Dataset(),
                ParquetResultSinkRequest(
                    uri="s3://bucket/results",
                    storage_profile="sink-domain",
                ),
                run_id="run-1",
                plan_digest="a" * 64,
            )
