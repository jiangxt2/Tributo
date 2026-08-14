"""Unit tests for the shared distributed Lance writer."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from tributo._common.lance_write import write_lance_dataset


@pytest.fixture
def fake_lance(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a minimal Lance module for dev-only unit environments."""
    module = MagicMock()
    monkeypatch.setitem(sys.modules, "lance", module)
    return module


class _FragmentRows:
    def __init__(self, fragments: list[object]) -> None:
        self._rows = [{"fragment": pickle.dumps(fragment)} for fragment in fragments]

    def take_all(self) -> list[dict[str, bytes]]:
        return self._rows


class _Dataset:
    def __init__(self, fragments: list[object]) -> None:
        self._fragments = fragments
        self.repartition_kwargs: dict[str, Any] | None = None
        self.map_kwargs: dict[str, Any] | None = None

    def repartition(self, **kwargs: Any) -> _Dataset:
        self.repartition_kwargs = kwargs
        return self

    def map_batches(self, fn: Any, **kwargs: Any) -> _FragmentRows:
        del fn
        self.map_kwargs = kwargs
        return _FragmentRows(self._fragments)


def _write(dataset: _Dataset, *, mode: str) -> None:
    write_lance_dataset(
        dataset,
        uri="/tmp/result.lance",
        schema=pa.schema([pa.field("id", pa.int64())]),
        mode=mode,
        min_rows_per_file=7,
        max_rows_per_file=11,
        data_storage_version=None,
        storage_options=None,
    )


def test_create_uses_strict_version_zero_commit(fake_lance: MagicMock) -> None:
    del fake_lance
    dataset = _Dataset(["fragment"])
    operation = object()
    with (
        patch("lance.LanceOperation.Overwrite", return_value=operation),
        patch("lance.LanceDataset.commit") as commit,
    ):
        _write(dataset, mode="create")

    assert dataset.repartition_kwargs == {
        "target_num_rows_per_block": 7,
        "strict": True,
    }
    assert dataset.map_kwargs == {"batch_format": "pyarrow"}
    commit.assert_called_once_with(
        "/tmp/result.lance",
        operation,
        storage_options=None,
        read_version=0,
        max_retries=0,
    )


def test_append_requires_an_existing_dataset(fake_lance: MagicMock) -> None:
    del fake_lance
    dataset = _Dataset(["fragment"])
    with (
        patch("lance.dataset", side_effect=OSError("dataset not found")),
        pytest.raises(OSError, match="dataset not found"),
    ):
        _write(dataset, mode="append")

    assert dataset.repartition_kwargs is None


def test_append_uses_the_observed_dataset_version(fake_lance: MagicMock) -> None:
    del fake_lance
    dataset = _Dataset(["fragment"])
    operation = object()
    with (
        patch("lance.dataset", return_value=MagicMock(version=5)),
        patch("lance.LanceOperation.Append", return_value=operation),
        patch("lance.LanceDataset.commit") as commit,
    ):
        _write(dataset, mode="append")

    commit.assert_called_once_with(
        "/tmp/result.lance",
        operation,
        storage_options=None,
        read_version=5,
    )


def test_empty_append_preserves_the_existing_version(fake_lance: MagicMock) -> None:
    del fake_lance
    dataset = _Dataset([])
    with (
        patch("lance.dataset", return_value=MagicMock(version=5)),
        patch("lance.write_dataset") as write_dataset,
        patch("lance.LanceDataset.commit") as commit,
    ):
        _write(dataset, mode="append")

    write_dataset.assert_not_called()
    commit.assert_not_called()


@pytest.mark.parametrize("mode", ["create", "overwrite"])
def test_empty_create_and_overwrite_materialize_the_declared_schema(
    mode: str, fake_lance: MagicMock
) -> None:
    del fake_lance
    dataset = _Dataset([])
    with patch("lance.write_dataset") as write_dataset:
        _write(dataset, mode=mode)

    empty = write_dataset.call_args.args[0]
    assert empty.num_rows == 0
    assert empty.schema == pa.schema([pa.field("id", pa.int64())])
    assert write_dataset.call_args.kwargs["mode"] == mode


def test_writer_rejects_an_unknown_mode_before_distributed_work() -> None:
    dataset = _Dataset(["fragment"])
    with (
        patch.dict(sys.modules, {"lance": None}),
        pytest.raises(ValueError, match="Unsupported Lance write mode"),
    ):
        _write(dataset, mode="unknown")

    assert dataset.repartition_kwargs is None


def test_strict_create_commit_rejects_a_second_version_zero_writer(
    tmp_path: Path,
) -> None:
    lance = pytest.importorskip("lance")

    uri = str(tmp_path / "strict-create.lance")
    schema = pa.schema([pa.field("id", pa.int64())])
    first_fragments = lance.fragment.write_fragments(
        pa.table({"id": [1]}, schema=schema),
        uri,
        schema=schema,
        mode="create",
    )
    second_fragments = lance.fragment.write_fragments(
        pa.table({"id": [2]}, schema=schema),
        uri,
        schema=schema,
        mode="create",
    )

    lance.LanceDataset.commit(
        uri,
        lance.LanceOperation.Overwrite(schema, first_fragments),
        read_version=0,
        max_retries=0,
    )
    with pytest.raises(OSError):
        lance.LanceDataset.commit(
            uri,
            lance.LanceOperation.Overwrite(schema, second_fragments),
            read_version=0,
            max_retries=0,
        )

    assert lance.dataset(uri).to_table()["id"].to_pylist() == [1]
