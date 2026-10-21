"""Restricted SQL filters, ordered residuals, and public reader delegation."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from types import ModuleType
from typing import Any, Literal

import pyarrow as pa
import pytest

from tributo.data.bindings._sql_filters import compile_sql_filters
from tributo.data.bindings.ray_clickhouse import RayClickHouseBinding
from tributo.data.bindings.ray_doris import RayDorisBinding
from tributo.data.engine_binding import BindingCompileRequest, BindingStageError
from tributo.data.ingestion import IngestionRuntimeContext, ReadOptions
from tributo.data.scan_plan import (
    SqlPredicate,
    SqlPredicateOperator,
    SqlScan,
    SqlTableRead,
)
from tributo.data.transform_compiler import ConcreteTransformCompiler, TransformBackend
from tributo.data.transform_ir import (
    CastColumn,
    FilterComparison,
    FilterEq,
    FilterIsIn,
    FilterNotEq,
    FilterNotNull,
    FilterNull,
    FilterRange,
    Limit,
    SelectColumns,
    TransformPipeline,
    TransformSpec,
)

_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("text", pa.string()),
        ("flag", pa.bool_()),
        ("score", pa.float64()),
        ("small", pa.float32()),
        ("amount", pa.decimal128(20, 4)),
        ("time", pa.timestamp("us")),
    ]
)
_Dialect = Literal["clickhouse", "doris"]


@pytest.mark.parametrize("dialect", ["clickhouse", "doris"])
@pytest.mark.parametrize(
    "spec",
    [
        FilterEq(column="id", value=-(2**63)),
        FilterNotEq(column="id", value=2**63 - 1),
        FilterComparison(column="id", operator="gte", value=80),
        FilterComparison(column="id", operator="lt", value=80),
        FilterRange(column="id", low=10, high=20),
        FilterIsIn(column="id", values=(1, 2, 3)),
        FilterEq(column="flag", value=True),
        FilterNotEq(column="flag", value=False),
        FilterRange(column="score", low=0.1, high=1.0),
        FilterComparison(column="score", operator="gt", value=80),
        FilterNull(column="text"),
        FilterNotNull(column="text"),
        FilterIsIn(column="amount", values=()),
    ],
)
def test_supported_filters_are_exact(dialect: _Dialect, spec: TransformSpec) -> None:
    pipeline = TransformPipeline(steps=(spec,))
    ConcreteTransformCompiler().compile(pipeline, TransformBackend.RAY, _SCHEMA)
    result = compile_sql_filters(pipeline, _SCHEMA, dialect)
    assert result.predicate
    assert result.decisions[0].pushdown_level == "exact"
    assert result.residual_ordinals == set()
    if isinstance(spec, FilterIsIn) and not spec.values:
        assert result.predicate == "(1 = 0)"
        assert not result.parameters


@pytest.mark.parametrize("value", ["", "中文", "a'\\\x00; -- # /* */", "{p:String}"])
def test_strings_are_values_and_nullable_text_is_preserved(value: str) -> None:
    pipeline = TransformPipeline(steps=(FilterEq(column="text", value=value),))
    clickhouse = compile_sql_filters(pipeline, _SCHEMA, "clickhouse")
    assert clickhouse.predicate == (
        "(CAST(`text` AS Nullable(String)) = {tributo_filter_0_0:String})"
    )
    assert clickhouse.parameters == {"tributo_filter_0_0": value}
    doris = compile_sql_filters(pipeline, _SCHEMA, "doris")
    assert doris.predicate == (
        f"(CAST(`text` AS STRING) = UNHEX('{value.encode('utf-8').hex()}'))"
    )
    assert not doris.parameters
    assert "UNHEX" not in repr(doris)
    if value:
        assert value not in repr(doris)


def test_parameters_keep_original_ordinals_and_do_not_collide() -> None:
    pipeline = TransformPipeline(
        steps=(
            FilterRange(column="id", low=1, high=10),
            FilterIsIn(column="text", values=("east", "west")),
        )
    )
    result = compile_sql_filters(pipeline, _SCHEMA, "clickhouse")
    assert result.parameters == {
        "tributo_filter_0_0": 1,
        "tributo_filter_0_1": 10,
        "tributo_filter_1_0": "east",
        "tributo_filter_1_1": "west",
    }
    assert result.predicate and " AND " in result.predicate
    assert [item.ordinal for item in result.decisions] == [0, 1]


@pytest.mark.parametrize(
    "barrier",
    [
        Limit(count=2),
        SelectColumns(columns=("id",)),
        CastColumn(column="id", target_type="int64"),
        FilterEq(column="amount", value=Decimal("1.0000")),
        FilterEq(column="time", value=datetime(2026, 1, 1)),
        FilterEq(column="small", value=0.1),
        FilterComparison(column="text", operator="gt", value="a"),
        FilterEq(column="id", value=1.5),
    ],
)
def test_no_filter_crosses_a_barrier(barrier: TransformSpec) -> None:
    pipeline = TransformPipeline(
        steps=(FilterEq(column="id", value=1), barrier, FilterEq(column="id", value=2))
    )
    result = compile_sql_filters(pipeline, _SCHEMA, "clickhouse")
    assert result.parameters == {"tributo_filter_0_0": 1}
    assert result.residual_ordinals == {1, 2}
    assert result.decisions[2].diagnostic == "after_transform_barrier"


def test_unrepresentable_identifier_is_residual() -> None:
    column = "column` OR 1=1 --"
    result = compile_sql_filters(
        TransformPipeline(steps=(FilterEq(column=column, value=1),)),
        pa.schema([(column, pa.int64())]),
        "doris",
    )
    assert result.predicate is None
    assert result.residual_ordinals == {0}
    assert result.decisions[0].diagnostic == "unsupported_sql_identifier"


class _Dataset:
    def __init__(self, schema: pa.Schema | None = _SCHEMA) -> None:
        self.input_schema = schema
        self.fetches: list[bool] = []
        self.operations: list[tuple[str, Any]] = []

    def schema(self, fetch_if_missing: bool = True) -> pa.Schema | None:
        self.fetches.append(fetch_if_missing)
        assert fetch_if_missing is False, "metadata discovery must not consume rows"
        return self.input_schema

    def select_columns(self, columns: list[str]) -> _Dataset:
        self.operations.append(("select", columns))
        return self

    def filter(self, *, expr: Any) -> _Dataset:
        self.operations.append(("filter", expr))
        return self

    def limit(self, count: int) -> _Dataset:
        self.operations.append(("limit", count))
        return self


def _request(dialect: _Dialect, transforms: TransformPipeline) -> BindingCompileRequest:
    return BindingCompileRequest(
        plan=SqlScan(
            provider_id=f"tributo.{dialect}",
            connector_id=dialect,
            target=SqlTableRead(schema="analytics", table="events"),
        ),
        runtime_options={"host": "db.example", "database": "analytics"},
        transforms=transforms,
        read_options=ReadOptions(),
        source_ref="0" * 64,
        runtime_context=IngestionRuntimeContext(),
    )


@pytest.mark.parametrize("dialect", ["clickhouse", "doris"])
def test_binding_replans_only_the_filtered_read_and_runs_residuals(
    monkeypatch: pytest.MonkeyPatch, dialect: _Dialect
) -> None:
    calls: list[dict[str, Any]] = []
    datasets: list[_Dataset] = []

    def read(**options: Any) -> _Dataset:
        calls.append(options)
        dataset = _Dataset()
        datasets.append(dataset)
        return dataset

    module = ModuleType(f"ray_{dialect}")
    setattr(module, f"read_{dialect}", read)
    monkeypatch.setitem(sys.modules, f"ray_{dialect}", module)
    pipeline = TransformPipeline(
        steps=(
            FilterComparison(column="id", operator="gte", value=9900),
            SelectColumns(columns=("id",)),
            Limit(count=100),
        )
    )
    binding = RayClickHouseBinding() if dialect == "clickhouse" else RayDorisBinding()
    result = binding.compile(_request(dialect, pipeline))
    assert len(calls) == 2
    assert "filter" not in calls[0]
    assert "`id` >= " in calls[1]["filter"]
    if dialect == "clickhouse":
        assert calls[1]["query_parameters"] == {"tributo_filter_0_0": 9900}
    assert not datasets[0].operations
    assert datasets[1].operations == [("select", ["id"]), ("limit", 100)]
    assert datasets[0].fetches == datasets[1].fetches == [False]
    assert [item.pushdown_level for item in result.transform_decisions] == [
        "exact",
        "none",
        "none",
    ]


@pytest.mark.parametrize("dialect", ["clickhouse", "doris"])
@pytest.mark.parametrize("failure", ["changed_schema", "missing_schema", "read_error"])
def test_binding_fails_closed_without_an_unfiltered_retry(
    monkeypatch: pytest.MonkeyPatch, dialect: _Dialect, failure: str
) -> None:
    calls = []

    def read(**options: Any) -> _Dataset:
        calls.append(options)
        if len(calls) == 2:
            if failure == "read_error":
                raise RuntimeError("reader failed")
            return _Dataset(None if failure == "missing_schema" else pa.schema([]))
        return _Dataset()

    module = ModuleType(f"ray_{dialect}")
    setattr(module, f"read_{dialect}", read)
    monkeypatch.setitem(sys.modules, f"ray_{dialect}", module)
    binding = RayClickHouseBinding() if dialect == "clickhouse" else RayDorisBinding()
    with pytest.raises(BindingStageError):
        binding.compile(
            _request(
                dialect, TransformPipeline(steps=(FilterEq(column="id", value=1),))
            )
        )
    assert len(calls) == 2


def test_whole_pipeline_is_validated_before_filtered_reader_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    module = ModuleType("ray_doris")

    def read(**options: Any) -> _Dataset:
        calls.append(options)
        return _Dataset()

    module.read_doris = read
    monkeypatch.setitem(sys.modules, "ray_doris", module)
    with pytest.raises(BindingStageError):
        RayDorisBinding().compile(
            _request(
                "doris",
                TransformPipeline(
                    steps=(
                        FilterEq(column="id", value=1),
                        SelectColumns(columns=("missing",)),
                    )
                ),
            )
        )
    assert len(calls) == 1


@pytest.mark.parametrize("dialect", ["clickhouse", "doris"])
def test_separate_scan_predicates_are_rejected_before_reader_io(
    monkeypatch: pytest.MonkeyPatch, dialect: _Dialect
) -> None:
    def read(**options: Any) -> _Dataset:
        pytest.fail("Unsupported predicate input must fail before reader I/O")

    module = ModuleType(f"ray_{dialect}")
    setattr(module, f"read_{dialect}", read)
    monkeypatch.setitem(sys.modules, f"ray_{dialect}", module)
    request = _request(dialect, TransformPipeline())
    plan = replace(
        request.plan,
        target=SqlTableRead(
            schema="analytics",
            table="events",
            predicates=(SqlPredicate("id", SqlPredicateOperator.EQ, (1,)),),
        ),
    )
    binding = RayClickHouseBinding() if dialect == "clickhouse" else RayDorisBinding()
    with pytest.raises(BindingStageError) as error:
        binding.compile(replace(request, plan=plan))
    assert error.value.stage == "validate_capabilities"
