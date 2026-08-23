"""Unit tests for data-owned object and Parquet persistence adapters."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tributo.data.persistence import (
    LocalS3ObjectStore,
    inspect_parquet_output,
    write_parquet_table,
)


def test_local_object_store_round_trip_lists_and_deletes_prefix(
    tmp_path: Path,
) -> None:
    store = LocalS3ObjectStore()
    root = tmp_path / "objects"
    store.write_bytes(str(root / "nested" / "payload.bin"), b"payload")

    assert store.read_bytes(str(root / "nested" / "payload.bin")) == b"payload"
    files = store.list_files(str(root))
    assert [(item.relative_path, item.size) for item in files] == [
        ("nested/payload.bin", 7)
    ]

    store.delete_tree(str(root))
    assert not root.exists()


def test_write_and_inspect_parquet_are_data_persistence_operations(
    tmp_path: Path,
) -> None:
    uri = str(tmp_path / "results" / "part.parquet")
    table = pa.table({"id": [1, 2], "value": ["a", "b"]})

    write_parquet_table(table, uri, exclusive=True)
    inspection = inspect_parquet_output(uri)

    assert inspection.rows == 2
    assert inspection.total_bytes == Path(uri).stat().st_size
    assert pq.read_table(uri).to_pylist() == table.to_pylist()


def test_parquet_writer_preserves_exclusive_create_semantics(tmp_path: Path) -> None:
    uri = str(tmp_path / "result.parquet")
    write_parquet_table(pa.table({"id": [1]}), uri, exclusive=True)

    with pytest.raises(FileExistsError):
        write_parquet_table(pa.table({"id": [2]}), uri, exclusive=True)
