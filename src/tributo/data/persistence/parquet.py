"""Parquet serialization and inspection for persistence bindings."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from tributo.data.persistence.object_store import default_object_store
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class ParquetInspection:
    """Digest and size evidence for a Parquet output prefix."""

    digest: str
    total_bytes: int
    rows: int


@DeveloperAPI
def write_parquet_table(
    table: pa.Table,
    uri: str,
    *,
    storage_profile: str | None = None,
    exclusive: bool = False,
) -> None:
    """Write one Arrow table to a local or S3 Parquet object."""
    parsed = urlsplit(uri)
    if parsed.scheme.lower() == "s3":
        with tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024,
            mode="w+b",
        ) as body:
            pq.write_table(table, body)
            body.seek(0)
            payload = body.read()
        default_object_store().write_bytes(
            uri,
            payload,
            storage_profile=storage_profile,
            exclusive=exclusive,
            content_type="application/vnd.apache.parquet",
        )
        return

    if parsed.scheme and parsed.scheme.lower() not in {"file"}:
        raise ValueError(f"unsupported Parquet output URI scheme: {parsed.scheme!r}")
    path = Path(parsed.path if parsed.scheme else uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as output:
            pq.write_table(table, output)
    else:
        pq.write_table(table, path)


@DeveloperAPI
def inspect_parquet_output(
    uri: str,
    *,
    storage_profile: str | None = None,
) -> ParquetInspection:
    """Inspect Parquet files under a local or S3 output prefix."""
    files = tuple(
        item
        for item in default_object_store().list_files(
            uri,
            storage_profile=storage_profile,
        )
        if item.relative_path.lower().endswith(".parquet")
    )
    if not files:
        raise ValueError(f"output contains no Parquet files at {uri!r}")

    digest = hashlib.sha256()
    total_bytes = 0
    rows = 0
    store = default_object_store()
    for file in files:
        payload = store.read_bytes(file.uri, storage_profile=storage_profile)
        digest.update(file.relative_path.encode())
        digest.update(payload)
        total_bytes += file.size
        rows += pq.ParquetFile(pa.BufferReader(payload)).metadata.num_rows
    return ParquetInspection(
        digest=digest.hexdigest(),
        total_bytes=total_bytes,
        rows=rows,
    )


__all__ = ["ParquetInspection", "inspect_parquet_output", "write_parquet_table"]
