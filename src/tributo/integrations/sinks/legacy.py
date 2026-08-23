"""Compatibility sink selection for the legacy batch inference facade."""

from __future__ import annotations

from typing import Literal

from tributo._common.storage_profiles import StorageProfile
from tributo.inference.contracts import (
    LanceResultSinkRequest,
    LanceVectorColumnSpec,
    ParquetResultSinkRequest,
    ResultSink,
    ResultSinkRequest,
)
from tributo.integrations.sinks import LanceResultSink, ParquetResultSink


class LegacyStorageProfileResolver:
    """Bind deprecated flat S3 settings only inside the compatibility adapter."""

    def __init__(self, raw: dict[str, str]) -> None:
        self._raw = dict(raw)

    def resolve(self, profile: str | None) -> StorageProfile:
        del profile
        return StorageProfile(
            endpoint=self._raw.get("endpoint"),
            region=self._raw.get("region"),
            access_key_id=self._raw.get("access_key_id"),
            secret_access_key=self._raw.get("secret_access_key"),
        )


def build_legacy_result_sink(
    *,
    output_format: Literal["parquet", "lance"],
    output_uri: str,
    output_storage_profile: str | None,
    legacy_s3_config: dict[str, str],
    output_mode: Literal["create", "append", "overwrite"],
    output_compression: str,
    min_rows_per_file: int | None,
    output_data_storage_version: str | None,
    output_vector_columns: list[LanceVectorColumnSpec],
) -> tuple[ResultSink, ResultSinkRequest]:
    """Build the historical Parquet/Lance sink without inference dispatch."""
    sink_profile = output_storage_profile
    legacy_resolver = None
    if sink_profile is None and legacy_s3_config:
        sink_profile = "legacy-flat-s3-config"
        legacy_resolver = LegacyStorageProfileResolver(legacy_s3_config)

    if output_format == "lance":
        sink = LanceResultSink(storage_resolver=legacy_resolver)
        request: ResultSinkRequest = LanceResultSinkRequest(
            uri=output_uri,
            mode=output_mode,
            storage_profile=sink_profile,
            data_storage_version=output_data_storage_version,
            min_rows_per_file=min_rows_per_file or 1024 * 1024,
            vector_columns=tuple(output_vector_columns),
        )
        return sink, request

    parquet_sink: ResultSink = ParquetResultSink(storage_resolver=legacy_resolver)
    request = ParquetResultSinkRequest(
        uri=output_uri,
        storage_profile=sink_profile,
        compression=output_compression,
        min_rows_per_file=min_rows_per_file,
    )
    return parquet_sink, request
