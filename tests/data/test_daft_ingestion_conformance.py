"""Single-node Daft semantic conformance for the public ingestion gateway."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tributo.data import (
    CastColumn,
    ColumnRename,
    DropColumns,
    FillNull,
    FilterComparison,
    FilterEq,
    FilterIsIn,
    FilterNotEq,
    FilterNotNull,
    FilterNull,
    FilterRange,
    IngestionRequest,
    Limit,
    ParquetSourceConfig,
    RenameColumns,
    SelectColumns,
    TransformPipeline,
    open_ingestion,
)
from tributo.exceptions import DataSourceError


def _table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "score": pa.array([0.2, 0.7, 0.9, 0.8], type=pa.float64()),
            "category": pa.array(["drop", "keep", "keep", None]),
            "active": pa.array([True, True, None, True], type=pa.bool_()),
            "numeric": pa.array(["1", "2", "3", "4"]),
            "unused": pa.array(["a", "b", "c", "d"]),
        }
    )


def _full_pipeline() -> TransformPipeline:
    return TransformPipeline(
        steps=(
            FillNull(column="category", value="unknown"),
            FillNull(column="active", value=False),
            FillNull(column="score", value=0.0),
            FilterNotEq(column="category", value="drop"),
            FilterComparison(column="id", operator="gte", value=2),
            FilterRange(column="score", low=0.5, high=1.0),
            FilterIsIn(column="id", values=(2, 3, 4)),
            FilterNotNull(column="active"),
            RenameColumns(renames=(ColumnRename(source="category", target="label"),)),
            FilterEq(column="label", value="keep"),
            CastColumn(column="numeric", target_type="int64"),
            DropColumns(columns=("active", "unused")),
            SelectColumns(columns=("id", "score", "label", "numeric")),
            Limit(count=10),
        )
    )


def test_daft_native_runner_executes_complete_transform_ir(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    pq.write_table(_table(), path)

    result = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=str(path)),
            engine="daft",
            transforms=_full_pipeline(),
        )
    )

    assert result.handle.dataframe.to_pylist() == [
        {"id": 2, "score": 0.7, "label": "keep", "numeric": 2},
        {"id": 3, "score": 0.9, "label": "keep", "numeric": 3},
    ]
    assert all(
        item.compiled_result == "residual"
        for item in result.receipt.transform_decisions
    )


def test_daft_null_and_empty_result_semantics(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    pq.write_table(_table(), path)
    null_result = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=str(path)),
            engine="daft",
            transforms=TransformPipeline(
                steps=(FilterNull(column="active"), SelectColumns(columns=("id",)))
            ),
        )
    )
    empty_result = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=str(path)),
            engine="daft",
            transforms=TransformPipeline(steps=(FilterIsIn(column="id", values=()),)),
        )
    )

    assert null_result.handle.dataframe.to_pylist() == [{"id": 3}]
    assert empty_result.handle.dataframe.to_pylist() == []


def test_daft_empty_input_and_not_equal_null_semantics(tmp_path: Path) -> None:
    populated_path = tmp_path / "populated.parquet"
    empty_path = tmp_path / "empty.parquet"
    pq.write_table(_table(), populated_path)
    pq.write_table(_table().slice(0, 0), empty_path)

    not_equal = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=str(populated_path)),
            engine="daft",
            transforms=TransformPipeline(
                steps=(
                    FilterNotEq(column="category", value="drop"),
                    SelectColumns(columns=("id",)),
                )
            ),
        )
    )
    empty = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=str(empty_path)),
            engine="daft",
        )
    )

    assert not_equal.handle.dataframe.to_pylist() == [{"id": 2}, {"id": 3}]
    assert empty.handle.dataframe.to_pylist() == []


def test_daft_binding_normalizes_native_compile_error(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    pq.write_table(_table(), path)

    with pytest.raises(
        DataSourceError,
        match=r"during build_native_plan \[unexpected\] with ValueError",
    ) as exc_info:
        open_ingestion(
            IngestionRequest(
                source=ParquetSourceConfig(path=str(path)),
                engine="daft",
                transforms=TransformPipeline(
                    steps=(FilterEq(column="missing", value=1),)
                ),
            )
        )

    assert "missing" not in str(exc_info.value)
    assert exc_info.value.__context__ is None
