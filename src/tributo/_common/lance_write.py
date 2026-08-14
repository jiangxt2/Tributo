"""Distributed Lance writing with strict Tributo save-mode semantics."""

from __future__ import annotations

import pickle
from functools import partial
from typing import Any, Literal, cast

import pyarrow as pa

_LanceStorageVersion = Literal[
    "stable", "2.0", "2.1", "2.2", "2.3", "next", "legacy", "0.1"
]
_LANCE_STORAGE_VERSIONS = frozenset(
    {"stable", "2.0", "2.1", "2.2", "2.3", "next", "legacy", "0.1"}
)


class LanceWriteConfigurationError(ValueError):
    """Invalid configuration detected before Lance performs any I/O."""


def write_lance_dataset(
    dataset: Any,
    *,
    uri: str,
    schema: pa.Schema,
    mode: str,
    min_rows_per_file: int,
    max_rows_per_file: int,
    data_storage_version: str | None,
    storage_options: dict[str, str] | None,
) -> None:
    """Write a Ray Dataset with explicit Lance transaction semantics.

    Ray Data still owns distributed block execution and repartitioning.  Lance
    owns fragment creation and the atomic commit.  Enforcing
    ``min_rows_per_file`` uses Ray's no-shuffle streaming repartition; strict
    sizing adds a distributed reblocking stage over all rows before fragment
    writes.  The pinned Ray 2.55.1 writer cannot preserve schema-bearing empty
    writes or fail-closed ``create`` semantics, so both public Tributo adapters
    share this narrow implementation instead of exposing two behaviorally
    different writers.
    """
    _write_distributed(
        dataset,
        uri=uri,
        schema=schema,
        mode=mode,
        min_rows_per_file=min_rows_per_file,
        max_rows_per_file=max_rows_per_file,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )


def _write_distributed(
    dataset: Any,
    *,
    uri: str,
    schema: pa.Schema,
    mode: str,
    min_rows_per_file: int,
    max_rows_per_file: int,
    data_storage_version: str | None,
    storage_options: dict[str, str] | None,
) -> None:
    if mode not in {"create", "append", "overwrite"}:
        raise LanceWriteConfigurationError(f"Unsupported Lance write mode: {mode!r}")
    lance_storage_version = _validated_storage_version(data_storage_version)

    import lance

    read_version: int | None = None
    if mode == "append":
        # APPEND is deliberately fail-closed.  A missing or inaccessible target
        # must not be converted into an implicit create operation.
        read_version = lance.dataset(uri, storage_options=storage_options).version

    fragment_source = dataset.repartition(
        target_num_rows_per_block=min_rows_per_file,
        strict=True,
    )
    fragment_batches = fragment_source.map_batches(
        partial(
            _write_fragment_batch,
            uri=uri,
            schema=schema,
            mode=mode,
            max_rows_per_file=max_rows_per_file,
            data_storage_version=lance_storage_version,
            storage_options=storage_options,
        ),
        batch_format="pyarrow",
    )
    rows = fragment_batches.take_all()
    fragments = [pickle.loads(row["fragment"]) for row in rows]

    if not fragments:
        if mode == "append":
            # An empty append is a no-op and preserves the existing version.
            return
        # CREATE and OVERWRITE promise a schema-bearing result even when no rows
        # are produced.  Lance's high-level writer provides the corresponding
        # atomic create/overwrite operation for an empty Arrow table.
        lance.write_dataset(
            pa.Table.from_batches([], schema=schema),
            uri,
            schema=schema,
            mode=mode,
            data_storage_version=lance_storage_version,
            storage_options=storage_options,
        )
        return

    operation: lance.LanceOperation.BaseOperation
    commit_kwargs: dict[str, Any] = {"storage_options": storage_options}
    if mode == "append":
        operation = lance.LanceOperation.Append(fragments)
        commit_kwargs["read_version"] = read_version
    else:
        operation = lance.LanceOperation.Overwrite(schema, fragments)
        if mode == "create":
            # A create transaction starts from the empty dataset version and
            # must not rebase onto a concurrently-created table.  Lance treats
            # max_retries=0 as strict overwrite, which makes version 0 an atomic
            # create precondition rather than a last-writer-wins overwrite.
            commit_kwargs["read_version"] = 0
            commit_kwargs["max_retries"] = 0
    lance.LanceDataset.commit(
        uri,
        operation,
        **commit_kwargs,
    )


def _write_fragment_batch(
    batch: pa.Table,
    *,
    uri: str,
    schema: pa.Schema,
    mode: str,
    max_rows_per_file: int,
    data_storage_version: str | None,
    storage_options: dict[str, str] | None,
) -> pa.Table:
    from lance.fragment import write_fragments

    reader = pa.RecordBatchReader.from_batches(schema, batch.to_batches())
    fragments = write_fragments(
        reader,
        uri,
        schema=schema,
        mode=mode,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=1024,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
    )
    return pa.table(
        {"fragment": [pickle.dumps(fragment) for fragment in fragments]},
        schema=pa.schema([pa.field("fragment", pa.binary(), nullable=False)]),
    )


def _validated_storage_version(
    value: str | None,
) -> _LanceStorageVersion | None:
    if value is not None and value not in _LANCE_STORAGE_VERSIONS:
        raise LanceWriteConfigurationError(
            f"Unsupported Lance data storage version: {value!r}"
        )
    return cast(_LanceStorageVersion | None, value)
