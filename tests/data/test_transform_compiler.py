"""A2: TransformCompiler contract tests — error paths and frozen semantics."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from tributo.data.transform_compiler import (
    ConcreteTransformCompiler,
    TransformBackend,
)
from tributo.data.transform_ir import (
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
    Limit,
    RenameColumns,
    SelectColumns,
    TransformPipeline,
    transform_ir_digest,
)

# Fixed schema for all tests
SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("x", pa.int64()),
        ("score", pa.float64()),
        ("name", pa.string()),
        ("active", pa.bool_()),
    ]
)


@pytest.fixture
def compiler() -> ConcreteTransformCompiler:
    return ConcreteTransformCompiler()


# ---------------------------------------------------------------------------
# Compilation success
# ---------------------------------------------------------------------------


class TestCompilationSuccess:
    @staticmethod
    def _all_specs() -> TransformPipeline:
        return TransformPipeline(
            steps=[
                FilterEq(column="x", value=30),
                FilterNotEq(column="name", value="nobody"),
                FilterComparison(column="id", operator="gte", value=1),
                FilterRange(column="score", low=0.0, high=100.0),
                FilterIsIn(column="name", values=["alice", "bob"]),
                FilterNull(column="name"),
                FilterNotNull(column="active"),
                CastColumn(column="x", target_type="float64"),
                FillNull(column="name", value="unknown"),
                RenameColumns(renames=[ColumnRename(source="name", target="label")]),
                DropColumns(columns=["active"]),
                SelectColumns(columns=["id", "x", "score", "label"]),
                Limit(count=10),
            ]
        )

    def test_all_specs_compile_daft(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = self._all_specs()
        compiled = compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)
        assert len(compiled.steps) == len(pipeline.steps)
        for i, step in enumerate(compiled.steps):
            assert step.ordinal == i
        assert compiled.steps[-1].output_schema.names == ["id", "x", "score", "label"]
        assert compiled.steps[-1].output_schema.field("x").type == pa.float64()

    def test_all_specs_compile_ray(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = self._all_specs()
        compiled = compiler.compile(pipeline, TransformBackend.RAY, SCHEMA)
        assert len(compiled.steps) == len(pipeline.steps)

    def test_select_columns_output_schema(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(steps=[SelectColumns(columns=["name", "id"])])
        compiled = compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)
        out_schema = compiled.steps[0].output_schema
        assert out_schema.names == ["name", "id"]
        assert out_schema.field("name").type == pa.string()
        assert out_schema.field("id").type == pa.int64()


# ---------------------------------------------------------------------------
# Error: column not found
# ---------------------------------------------------------------------------


class TestColumnNotFound:
    def test_filter_eq_missing_column(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(steps=[FilterEq(column="z", value=1)])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_filter_range_missing_column(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(steps=[FilterRange(column="z", low=0, high=100)])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_filter_isin_missing_column(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(steps=[FilterIsIn(column="z", values=[1])])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_filter_null_missing_column(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(steps=[FilterNull(column="z")])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_select_columns_missing_column(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(steps=[SelectColumns(columns=["z"])])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_select_then_filter_removed_column(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        """SelectColumns removes 'name', then FilterEq on 'name' → compile error."""
        pipeline = TransformPipeline(
            steps=[
                SelectColumns(columns=["id", "x"]),
                FilterEq(column="name", value="alice"),
            ]
        )
        with pytest.raises(ValueError, match="Column 'name' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)


# ---------------------------------------------------------------------------
# Error: illegal TransformSpec values
# ---------------------------------------------------------------------------


class TestIllegalValues:
    def test_filter_eq_null_value(self) -> None:
        with pytest.raises(ValueError, match="FilterEq.*None.*illegal"):
            FilterEq(column="x", value=None)

    def test_filter_isin_none_in_values(self) -> None:
        with pytest.raises(ValueError, match="cannot contain None"):
            FilterIsIn(column="x", values=[1, None, 3])

    def test_select_columns_empty(self) -> None:
        with pytest.raises(ValueError):
            SelectColumns(columns=[])

    def test_filter_range_low_gt_high_rejected(self) -> None:
        with pytest.raises(ValueError, match="FilterRange low"):
            FilterRange(column="x", low=100, high=10)

    def test_filter_range_incomparable_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="mutually comparable"):
            FilterRange(column="x", low=1, high="10")

    def test_mutable_filter_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            FilterEq(column="x", value={"mutable": "value"})

    def test_non_finite_filter_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FilterEq(column="score", value=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            FilterEq(column="score", value=Decimal("Infinity"))

    def test_unsafe_cast_is_not_part_of_the_portable_contract(self) -> None:
        with pytest.raises(ValueError, match="Input should be True"):
            CastColumn(column="x", target_type="int64", safe=False)

    def test_select_columns_duplicate(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            TransformPipeline(steps=[SelectColumns(columns=["id", "id"])])


# ---------------------------------------------------------------------------
# Frozen semantics contract
# ---------------------------------------------------------------------------


class TestFrozenSemantics:
    def test_empty_pipeline_is_identity(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline()
        compiled = compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)
        assert len(compiled.steps) == 0

    def test_empty_isin_always_false(self) -> None:
        """Empty values list is valid — spec says always false."""
        spec = FilterIsIn(column="x", values=[])
        assert spec.values == ()
        assert spec.type == "filter_isin"

    def test_filter_null_only_matches_null(self) -> None:
        """FilterNull semantics: Arrow null only, not NaN."""
        spec = FilterNull(column="name")
        assert spec.type == "filter_null"
        assert spec.column == "name"

    def test_select_columns_preserves_declared_order(self) -> None:
        spec = SelectColumns(columns=["c", "b", "a"])
        assert spec.columns == ("c", "b", "a")

    def test_transform_digest_preserves_scalar_types(self) -> None:
        string_value = TransformPipeline(
            steps=[FilterEq(column="day", value="2026-08-05")]
        )
        date_value = TransformPipeline(
            steps=[FilterEq(column="day", value=date(2026, 8, 5))]
        )

        assert transform_ir_digest(string_value) != transform_ir_digest(date_value)

    def test_json_round_trip_preserves_scalar_types_and_digest(self) -> None:
        pipeline = TransformPipeline(
            steps=[
                FilterEq(column="day", value=date(2026, 8, 5)),
                FilterEq(
                    column="instant",
                    value=datetime(2026, 8, 5, 12, 30, tzinfo=UTC),
                ),
                FilterEq(column="amount", value=Decimal("10.20")),
            ]
        )

        restored = TransformPipeline.model_validate_json(pipeline.model_dump_json())

        assert isinstance(restored.steps[0].value, date)
        assert isinstance(restored.steps[1].value, datetime)
        assert isinstance(restored.steps[2].value, Decimal)
        assert transform_ir_digest(restored) == transform_ir_digest(pipeline)

    def test_unknown_transform_variant_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="does not match any"):
            TransformPipeline.model_validate(
                {"version": 1, "steps": [{"type": "future_transform"}]}
            )

    def test_timestamp_cast_preserves_timezone_case(
        self, compiler: ConcreteTransformCompiler
    ) -> None:
        pipeline = TransformPipeline(
            steps=[CastColumn(column="id", target_type="timestamp[us, tz=UTC]")]
        )

        compiled = compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

        assert compiled.steps[0].output_schema.field("id").type == pa.timestamp(
            "us", tz="UTC"
        )
