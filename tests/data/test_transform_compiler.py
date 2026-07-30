"""A2: TransformCompiler contract tests — error paths and frozen semantics."""

from __future__ import annotations

import pyarrow as pa
import pytest

from tributo.data.provider import (
    FilterEq,
    FilterIsIn,
    FilterNull,
    FilterRange,
    SelectColumns,
    TransformBackend,
    TransformPipeline,
)
from tributo.data.transform_compiler import ConcreteTransformCompiler

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
    def test_all_five_specs_compile_daft(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(
            steps=[
                FilterEq(column="x", value=30),
                FilterRange(column="score", low=0.0, high=100.0),
                FilterIsIn(column="name", values=["alice", "bob"]),
                FilterNull(column="name"),
                SelectColumns(columns=["id", "x"]),
            ]
        )
        compiled = compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)
        assert len(compiled.steps) == 5
        for i, step in enumerate(compiled.steps):
            assert step.ordinal == i

    def test_all_five_specs_compile_ray(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(
            steps=[
                FilterEq(column="x", value=30),
                FilterRange(column="score", low=0.0, high=100.0),
                FilterIsIn(column="name", values=["alice", "bob"]),
                FilterNull(column="name"),
                SelectColumns(columns=["id", "x"]),
            ]
        )
        compiled = compiler.compile(pipeline, TransformBackend.RAY, SCHEMA)
        assert len(compiled.steps) == 5

    def test_select_columns_output_schema(self, compiler: ConcreteTransformCompiler) -> None:
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
    def test_filter_eq_missing_column(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(steps=[FilterEq(column="z", value=1)])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_filter_range_missing_column(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(steps=[FilterRange(column="z", low=0, high=100)])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_filter_isin_missing_column(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(steps=[FilterIsIn(column="z", values=[1])])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_filter_null_missing_column(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(steps=[FilterNull(column="z")])
        with pytest.raises(ValueError, match="Column 'z' not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_select_columns_missing_column(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(steps=[SelectColumns(columns=["z"])])
        with pytest.raises(ValueError, match="Column 'z' in SelectColumns not found"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)

    def test_select_then_filter_removed_column(self, compiler: ConcreteTransformCompiler) -> None:
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

    def test_select_columns_duplicate(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline(steps=[SelectColumns(columns=["id", "id"])])
        with pytest.raises(ValueError, match="Duplicate column"):
            compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)


# ---------------------------------------------------------------------------
# Frozen semantics contract
# ---------------------------------------------------------------------------


class TestFrozenSemantics:
    def test_empty_pipeline_is_identity(self, compiler: ConcreteTransformCompiler) -> None:
        pipeline = TransformPipeline()
        compiled = compiler.compile(pipeline, TransformBackend.DAFT, SCHEMA)
        assert len(compiled.steps) == 0

    def test_empty_isin_always_false(self) -> None:
        """Empty values list is valid — spec says always false."""
        spec = FilterIsIn(column="x", values=[])
        assert spec.values == []
        assert spec.type == "filter_isin"

    def test_filter_null_only_matches_null(self) -> None:
        """FilterNull semantics: Arrow null only, not NaN."""
        spec = FilterNull(column="name")
        assert spec.type == "filter_null"
        assert spec.column == "name"

    def test_select_columns_preserves_declared_order(self) -> None:
        spec = SelectColumns(columns=["c", "b", "a"])
        assert spec.columns == ["c", "b", "a"]
