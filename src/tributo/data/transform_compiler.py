"""Internal translation of Transform IR into Ray Data or Daft expressions."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, cast

import pyarrow as pa
import pyarrow.compute as pc

from tributo.data.transform_ir import (
    CastColumn,
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
    TransformScalar,
    TransformSpec,
)
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class TransformBackend(Enum):
    """Native execution target for internal Transform compilation."""

    DAFT = "daft"
    RAY = "ray"


@DeveloperAPI
@dataclass(frozen=True)
class CompiledStep:
    """A single compiled TransformSpec with schema and ordinal trace."""

    ordinal: int
    spec: TransformSpec
    input_schema: pa.Schema
    output_schema: pa.Schema
    backend_op: Any


@DeveloperAPI
class CompiledPipeline:
    """Ordered compiled steps tagged with their original ordinals."""

    def __init__(self, steps: Sequence[CompiledStep]) -> None:
        self._steps = tuple(steps)
        ordinals = tuple(step.ordinal for step in self._steps)
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("Compiled pipeline contains duplicate ordinals")

    @property
    def steps(self) -> tuple[CompiledStep, ...]:
        return self._steps


@DeveloperAPI
class TransformCompiler(ABC):
    """Compile a complete Transform pipeline for one native engine."""

    @abstractmethod
    def compile(
        self,
        pipeline: TransformPipeline,
        backend: TransformBackend,
        input_schema: pa.Schema,
    ) -> CompiledPipeline: ...


@DeveloperAPI
class ConcreteTransformCompiler(TransformCompiler):
    """Compile the complete P0 Transform IR to Ray or Daft operations."""

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
        if backend is TransformBackend.DAFT:
            backend_op, output_schema = _compile_daft(spec, input_schema)
        elif backend is TransformBackend.RAY:
            backend_op, output_schema = _compile_ray(spec, input_schema)
        else:
            raise ValueError(f"Unsupported backend: {backend!r}")
        return CompiledStep(
            ordinal=ordinal,
            spec=spec,
            input_schema=input_schema,
            output_schema=output_schema,
            backend_op=backend_op,
        )


_COLUMN_TRANSFORMS = (
    FilterEq,
    FilterNotEq,
    FilterComparison,
    FilterRange,
    FilterIsIn,
    FilterNull,
    FilterNotNull,
    CastColumn,
    FillNull,
)


def _validate_spec(spec: TransformSpec, schema: pa.Schema) -> None:
    """Validate referenced columns and scalar compatibility against Arrow."""
    if isinstance(spec, SelectColumns):
        _require_columns(spec.columns, schema, spec.type)
        return
    if isinstance(spec, DropColumns):
        _require_columns(spec.columns, schema, spec.type)
        return
    if isinstance(spec, RenameColumns):
        sources = tuple(item.source for item in spec.renames)
        _require_columns(sources, schema, spec.type)
        source_set = set(sources)
        for item in spec.renames:
            if item.target in schema.names and item.target not in source_set:
                raise ValueError(
                    f"Rename target {item.target!r} already exists and is not renamed"
                )
        return
    if isinstance(spec, _COLUMN_TRANSFORMS):
        _require_columns((spec.column,), schema, spec.type)
        field = schema.field(spec.column)
        if isinstance(spec, (FilterEq, FilterNotEq, FilterComparison)):
            _validate_scalar_for_field(spec.value, field)
        elif isinstance(spec, FilterRange):
            _validate_scalar_for_field(spec.low, field)
            _validate_scalar_for_field(spec.high, field)
        elif isinstance(spec, FilterIsIn):
            for value in spec.values:
                _validate_scalar_for_field(value, field)
        elif isinstance(spec, FillNull):
            _validate_scalar_for_field(spec.value, field)
        elif isinstance(spec, CastColumn):
            _parse_arrow_type(spec.target_type)
        return
    if isinstance(spec, Limit):
        return
    raise ValueError(f"Unknown Transform spec {type(spec).__name__}")


def _require_columns(columns: tuple[str, ...], schema: pa.Schema, name: str) -> None:
    for column in columns:
        if schema.get_field_index(column) == -1:
            raise ValueError(
                f"Column {column!r} not found in schema for {name}. "
                f"Available: {schema.names}"
            )


def _validate_scalar_for_field(value: TransformScalar, field: pa.Field) -> None:
    try:
        pa.scalar(value, type=field.type)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Value {value!r} is incompatible with Arrow field "
            f"{field.name!r}:{field.type}"
        ) from exc


def _parse_arrow_type(value: str) -> pa.DataType:
    stripped = value.strip()
    normalized = stripped.lower()
    try:
        return pa.type_for_alias(normalized)
    except (KeyError, ValueError):
        pass
    decimal_match = re.fullmatch(r"decimal(128|256)\((\d+),(\d+)\)", normalized)
    if decimal_match:
        bits, precision, scale = (int(part) for part in decimal_match.groups())
        constructor = pa.decimal128 if bits == 128 else pa.decimal256
        return constructor(precision, scale)
    timestamp_match = re.fullmatch(
        r"timestamp\[(s|ms|us|ns)(?:,\s*tz=([^\]]+))?\]",
        stripped,
        flags=re.IGNORECASE,
    )
    if timestamp_match:
        unit, timezone = timestamp_match.groups()
        return pa.timestamp(unit.lower(), tz=timezone)
    raise ValueError(f"Unsupported canonical Arrow target type {value!r}")


def _replace_field(schema: pa.Schema, column: str, field: pa.Field) -> pa.Schema:
    fields = [field if existing.name == column else existing for existing in schema]
    return pa.schema(fields, metadata=schema.metadata)


def _drop_schema(schema: pa.Schema, columns: tuple[str, ...]) -> pa.Schema:
    dropped = set(columns)
    return pa.schema(
        [field for field in schema if field.name not in dropped],
        metadata=schema.metadata,
    )


def _rename_schema(schema: pa.Schema, spec: RenameColumns) -> pa.Schema:
    renames = {item.source: item.target for item in spec.renames}
    fields = [
        pa.field(
            renames.get(field.name, field.name),
            field.type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        for field in schema
    ]
    return pa.schema(fields, metadata=schema.metadata)


def _comparison_expression(column: Any, operator: str, value: Any) -> Any:
    if operator == "lt":
        return column < value
    if operator == "lte":
        return column <= value
    if operator == "gt":
        return column > value
    if operator == "gte":
        return column >= value
    raise ValueError(f"Unsupported comparison operator {operator!r}")


@dataclass(frozen=True)
class _RayFillNullOperation:
    """Serializable Arrow batch operation executed by Ray Data map_batches."""

    column: str
    fill_value: pa.Scalar

    def __call__(self, table: pa.Table) -> pa.Table:
        index = table.schema.get_field_index(self.column)
        filled = pc.fill_null(table.column(index), self.fill_value)
        return table.set_column(index, table.schema.field(index), filled)


def _compile_daft(spec: TransformSpec, schema: pa.Schema) -> tuple[Any, pa.Schema]:
    import daft

    column: Any = (
        daft.col(spec.column) if isinstance(spec, _COLUMN_TRANSFORMS) else None
    )
    if isinstance(spec, FilterEq):
        return column == spec.value, schema
    if isinstance(spec, FilterNotEq):
        return column != spec.value, schema
    if isinstance(spec, FilterComparison):
        return _comparison_expression(column, spec.operator, spec.value), schema
    if isinstance(spec, FilterRange):
        return (column >= cast(Any, spec.low)) & (
            column <= cast(Any, spec.high)
        ), schema
    if isinstance(spec, FilterIsIn):
        return (
            daft.lit(False) if not spec.values else column.is_in(list(spec.values)),
            schema,
        )
    if isinstance(spec, FilterNull):
        return column.is_null(), schema
    if isinstance(spec, FilterNotNull):
        return ~column.is_null(), schema
    if isinstance(spec, SelectColumns):
        return tuple(spec.columns), pa.schema(
            [schema.field(name) for name in spec.columns], metadata=schema.metadata
        )
    if isinstance(spec, DropColumns):
        return tuple(spec.columns), _drop_schema(schema, spec.columns)
    if isinstance(spec, RenameColumns):
        return {item.source: item.target for item in spec.renames}, _rename_schema(
            schema, spec
        )
    if isinstance(spec, CastColumn):
        target = _parse_arrow_type(spec.target_type)
        output = _replace_field(
            schema,
            spec.column,
            pa.field(
                spec.column,
                target,
                nullable=schema.field(spec.column).nullable,
                metadata=schema.field(spec.column).metadata,
            ),
        )
        return column.cast(daft.DataType.from_arrow_type(target)), output
    if isinstance(spec, FillNull):
        return column.fill_null(spec.value), schema
    if isinstance(spec, Limit):
        return spec.count, schema
    raise ValueError(f"Unsupported spec: {type(spec).__name__}")


def _compile_ray(spec: TransformSpec, schema: pa.Schema) -> tuple[Any, pa.Schema]:
    from ray.data.expressions import DataType, col

    column: Any = col(spec.column) if isinstance(spec, _COLUMN_TRANSFORMS) else None
    if isinstance(spec, FilterEq):
        return column == spec.value, schema
    if isinstance(spec, FilterNotEq):
        return column != spec.value, schema
    if isinstance(spec, FilterComparison):
        return _comparison_expression(column, spec.operator, spec.value), schema
    if isinstance(spec, FilterRange):
        return (column >= spec.low) & (column <= spec.high), schema
    if isinstance(spec, FilterIsIn):
        return (
            column.is_null() & column.is_not_null()
            if not spec.values
            else column.is_in(list(spec.values)),
            schema,
        )
    if isinstance(spec, FilterNull):
        return column.is_null(), schema
    if isinstance(spec, FilterNotNull):
        return column.is_not_null(), schema
    if isinstance(spec, SelectColumns):
        return tuple(spec.columns), pa.schema(
            [schema.field(name) for name in spec.columns], metadata=schema.metadata
        )
    if isinstance(spec, DropColumns):
        return tuple(spec.columns), _drop_schema(schema, spec.columns)
    if isinstance(spec, RenameColumns):
        return {item.source: item.target for item in spec.renames}, _rename_schema(
            schema, spec
        )
    if isinstance(spec, CastColumn):
        target = _parse_arrow_type(spec.target_type)
        output = _replace_field(
            schema,
            spec.column,
            pa.field(
                spec.column,
                target,
                nullable=schema.field(spec.column).nullable,
                metadata=schema.field(spec.column).metadata,
            ),
        )
        return column.cast(DataType.from_arrow(target), safe=spec.safe), output
    if isinstance(spec, FillNull):
        field = schema.field(spec.column)
        fill_value = pa.scalar(spec.value, type=field.type)
        return _RayFillNullOperation(spec.column, fill_value), schema
    if isinstance(spec, Limit):
        return spec.count, schema
    raise ValueError(f"Unsupported spec: {type(spec).__name__}")


@DeveloperAPI
def apply_pipeline_to_daft_df(
    pipeline: CompiledPipeline,
    df: Any,
    *,
    ordinals: set[int] | None = None,
) -> Any:
    """Apply compiled residual steps to a lazy Daft DataFrame."""
    import daft

    result = df
    for step in pipeline.steps:
        if ordinals is not None and step.ordinal not in ordinals:
            continue
        spec = step.spec
        if isinstance(spec, SelectColumns):
            result = result.select(*step.backend_op)
        elif isinstance(spec, DropColumns):
            result = result.exclude(*step.backend_op)
        elif isinstance(spec, RenameColumns):
            result = result.select(
                *[
                    daft.col(name).alias(step.backend_op.get(name, name))
                    for name in step.input_schema.names
                ]
            )
        elif isinstance(spec, (CastColumn, FillNull)):
            result = result.with_column(spec.column, step.backend_op)
        elif isinstance(spec, Limit):
            result = result.limit(step.backend_op)
        elif isinstance(step.backend_op, daft.Expression):
            result = result.where(step.backend_op)
        else:
            raise ValueError(
                f"Unknown backend operation for Daft: {type(step.backend_op).__name__}"
            )
    return result


@DeveloperAPI
def apply_pipeline_to_ray_ds(
    pipeline: CompiledPipeline,
    ds: Any,
    *,
    ordinals: set[int] | None = None,
) -> Any:
    """Apply compiled residual steps to a lazy Ray Dataset."""
    from ray.data.expressions import Expr

    result = ds
    for step in pipeline.steps:
        if ordinals is not None and step.ordinal not in ordinals:
            continue
        spec = step.spec
        if isinstance(spec, SelectColumns):
            result = result.select_columns(list(step.backend_op))
        elif isinstance(spec, DropColumns):
            result = result.drop_columns(list(step.backend_op))
        elif isinstance(spec, RenameColumns):
            result = result.rename_columns(step.backend_op)
        elif isinstance(spec, CastColumn):
            result = result.with_column(spec.column, step.backend_op)
        elif isinstance(spec, FillNull):
            result = result.map_batches(step.backend_op, batch_format="pyarrow")
        elif isinstance(spec, Limit):
            result = result.limit(step.backend_op)
        elif isinstance(step.backend_op, Expr):
            result = result.filter(expr=step.backend_op)
        else:
            raise ValueError(
                f"Unknown backend operation for Ray: {type(step.backend_op).__name__}"
            )
    return result


__all__ = [
    "CompiledPipeline",
    "CompiledStep",
    "ConcreteTransformCompiler",
    "TransformBackend",
    "TransformCompiler",
    "apply_pipeline_to_daft_df",
    "apply_pipeline_to_ray_ds",
]
