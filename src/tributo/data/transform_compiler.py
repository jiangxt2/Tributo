"""Phase 2a TransformCompiler implementation for DAFT and RAY backends.

This module translates ``TransformSpec`` objects into engine-native
expressions and applies them to SourcePlans.

These are validation prototypes — stay in worktree, do NOT merge to master.
"""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa

from tributo.data.provider import (
    CompiledPipeline,
    CompiledStep,
    FilterEq,
    FilterIsIn,
    FilterNull,
    FilterRange,
    SelectColumns,
    TransformBackend,
    TransformCompiler,
    TransformPipeline,
    TransformSpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class ConcreteTransformCompiler(TransformCompiler):
    """Compile ``TransformPipeline`` → ``CompiledPipeline`` for a backend."""

    def compile(
        self,
        pipeline: TransformPipeline,
        backend: TransformBackend,
        input_schema: pa.Schema,
    ) -> CompiledPipeline:
        steps: list[CompiledStep] = []
        current_schema = input_schema

        for ordinal, spec in enumerate(pipeline.steps):
            step = self._compile_one(ordinal, spec, backend, current_schema)
            current_schema = step.output_schema
            steps.append(step)

        return CompiledPipeline(steps)

    def _compile_one(
        self,
        ordinal: int,
        spec: TransformSpec,
        backend: TransformBackend,
        input_schema: pa.Schema,
    ) -> CompiledStep:
        _validate_spec(spec, input_schema)

        if backend == TransformBackend.DAFT:
            backend_op, output_schema = _compile_daft(spec, input_schema)
        elif backend == TransformBackend.RAY:
            backend_op, output_schema = _compile_ray(spec, input_schema)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return CompiledStep(
            ordinal=ordinal,
            spec=spec,
            input_schema=input_schema,
            output_schema=output_schema,
            backend_op=backend_op,
        )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_spec(spec: TransformSpec, schema: pa.Schema) -> None:
    """Validate that spec columns exist in schema. Raises ValueError on failure."""
    columns: list[str] = _spec_columns(spec)
    for col in columns:
        idx = schema.get_field_index(col)
        if idx == -1:
            raise ValueError(
                f"Column {col!r} not found in schema for {spec.type}. "
                f"Available: {schema.names}"
            )


def _spec_columns(spec: TransformSpec) -> list[str]:
    if isinstance(spec, SelectColumns):
        return []  # SelectColumns validation is handled separately below
    if isinstance(spec, (FilterEq, FilterRange, FilterIsIn, FilterNull)):
        return [spec.column]
    raise ValueError(f"Unknown spec type: {type(spec)}")


def _validate_select_columns(spec: SelectColumns, schema: pa.Schema) -> None:
    """Validate SelectColumns: at least one column, no duplicates, all exist."""
    if len(set(spec.columns)) != len(spec.columns):
        raise ValueError(f"Duplicate column names in SelectColumns: {spec.columns}")
    for col in spec.columns:
        if schema.get_field_index(col) == -1:
            raise ValueError(
                f"Column {col!r} in SelectColumns not found in schema. "
                f"Available: {schema.names}"
            )


# ---------------------------------------------------------------------------
# DAFT backend
# ---------------------------------------------------------------------------


def _compile_daft(spec: TransformSpec, schema: pa.Schema) -> tuple[Any, pa.Schema]:
    """Compile a TransformSpec to a Daft expression + output schema."""
    import daft
    from daft import col as daft_col

    if isinstance(spec, FilterEq):
        return daft_col(spec.column) == spec.value, schema

    if isinstance(spec, FilterRange):
        c = daft_col(spec.column)
        return (c >= spec.low) & (c <= spec.high), schema

    if isinstance(spec, FilterIsIn):
        if len(spec.values) == 0:
            # Always false: use literal False
            return daft.lit(False), schema
        return daft_col(spec.column).is_in(spec.values), schema

    if isinstance(spec, FilterNull):
        return daft_col(spec.column).is_null(), schema

    if isinstance(spec, SelectColumns):
        _validate_select_columns(spec, schema)
        output_fields = [schema.field(col) for col in spec.columns]
        return spec.columns, pa.schema(output_fields)

    raise ValueError(f"Unsupported spec: {type(spec).__name__}")


# ---------------------------------------------------------------------------
# RAY backend
# ---------------------------------------------------------------------------


def _compile_ray(spec: TransformSpec, schema: pa.Schema) -> tuple[Any, pa.Schema]:
    """Compile a TransformSpec to a Ray Data expression/op + output schema."""
    from ray.data.expressions import col as ray_col

    if isinstance(spec, FilterEq):
        return ray_col(spec.column) == spec.value, schema

    if isinstance(spec, FilterRange):
        c = ray_col(spec.column)
        return (c >= spec.low) & (c <= spec.high), schema

    if isinstance(spec, FilterIsIn):
        if len(spec.values) == 0:
            # Always false: use an impossible condition
            return ray_col(spec.column).is_null() & ray_col(
                spec.column
            ).is_not_null(), schema
        return ray_col(spec.column).is_in(spec.values), schema

    if isinstance(spec, FilterNull):
        return ray_col(spec.column).is_null(), schema

    if isinstance(spec, SelectColumns):
        _validate_select_columns(spec, schema)
        output_fields = [schema.field(col) for col in spec.columns]
        return spec.columns, pa.schema(output_fields)

    raise ValueError(f"Unsupported spec: {type(spec).__name__}")


# ---------------------------------------------------------------------------
# Source plan application (applies compiled pipeline to native plans)
# ---------------------------------------------------------------------------


def apply_pipeline_to_daft_df(
    pipeline: CompiledPipeline,
    df: Any,  # daft.DataFrame
    *,
    ordinals: set[int] | None = None,
) -> Any:  # daft.DataFrame
    """Apply compiled steps to a Daft DataFrame.

    Filter steps use ``df.where(expr)``; SelectColumns uses ``df.select(*cols)``.
    """
    import daft

    result = df
    for step in pipeline.steps:
        if ordinals is not None and step.ordinal not in ordinals:
            continue

        if isinstance(step.spec, SelectColumns):
            result = result.select(*step.backend_op)
        elif isinstance(step.backend_op, daft.Expression):
            result = result.where(step.backend_op)
        else:
            raise ValueError(
                f"Unknown backend op type for Daft: {type(step.backend_op)}"
            )
    return result


def apply_pipeline_to_ray_ds(
    pipeline: CompiledPipeline,
    ds: Any,  # ray.data.Dataset
    *,
    ordinals: set[int] | None = None,
) -> Any:  # ray.data.Dataset
    """Apply compiled steps to a Ray Dataset.

    Filter steps use ``ds.filter(expr=...)``; SelectColumns uses ``ds.select_columns(...)``.
    """
    from ray.data.expressions import Expr

    result = ds
    for step in pipeline.steps:
        if ordinals is not None and step.ordinal not in ordinals:
            continue

        if isinstance(step.spec, SelectColumns):
            result = result.select_columns(step.backend_op)
        elif isinstance(step.backend_op, Expr):
            result = result.filter(expr=step.backend_op)
        else:
            raise ValueError(
                f"Unknown backend op type for Ray: {type(step.backend_op)}"
            )
    return result
