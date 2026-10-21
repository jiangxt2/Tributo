"""Restricted, result-preserving SQL compilation of leading filter transforms."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa

from tributo.data.engine_binding import PushdownLevel, classify_transform_decisions
from tributo.data.ingestion import TransformDecision
from tributo.data.scan_plan import SqlScan, SqlTableRead
from tributo.data.transform_ir import (
    FilterComparison,
    FilterEq,
    FilterIsIn,
    FilterNotEq,
    FilterNotNull,
    FilterNull,
    FilterRange,
    TransformPipeline,
    TransformScalar,
)
from tributo.exceptions import JobConfigurationError

_Dialect = Literal["clickhouse", "doris"]
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")
_FILTERS = (
    FilterEq,
    FilterNotEq,
    FilterComparison,
    FilterRange,
    FilterIsIn,
    FilterNull,
    FilterNotNull,
)


def require_transform_filters(plan: SqlScan) -> None:
    """Reject the separate developer predicate form before opening a reader."""
    if isinstance(plan.target, SqlTableRead) and plan.target.predicates:
        raise JobConfigurationError(
            "SQL table predicates must be expressed as ingestion transforms"
        )


@dataclass(frozen=True)
class _SqlFilters:
    predicate: str | None = field(repr=False)
    parameters: dict[str, TransformScalar] = field(repr=False)
    decisions: tuple[TransformDecision, ...]

    @property
    def residual_ordinals(self) -> set[int]:
        return {item.ordinal for item in self.decisions if item.residual_required}


def _scalar_kind(dtype: pa.DataType, values: tuple[TransformScalar, ...]) -> str | None:
    # Do not coerce scalars merely because Arrow can construct a typed scalar:
    # the existing Ray expression still compares against the original value.
    if pa.types.is_integer(dtype) and all(type(value) is int for value in values):
        return ("UInt" if pa.types.is_unsigned_integer(dtype) else "Int") + str(
            dtype.bit_width
        )
    if pa.types.is_boolean(dtype) and all(type(value) is bool for value in values):
        return "Bool"
    if pa.types.is_float64(dtype) and all(
        type(value) is float or (type(value) is int and abs(value) <= 2**53)
        for value in values
    ):
        return "Float64"
    if (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)) and all(
        type(value) is str for value in values
    ):
        return "String"
    return None


def _render_filter(
    spec: object,
    schema: pa.Schema,
    dialect: _Dialect,
    ordinal: int,
) -> tuple[str | None, dict[str, TransformScalar], str]:
    if not isinstance(spec, _FILTERS):
        return None, {}, "ordered_transform_barrier"
    if _IDENTIFIER.fullmatch(spec.column) is None:
        return None, {}, "unsupported_sql_identifier"
    column = f"`{spec.column}`"
    if isinstance(spec, (FilterNull, FilterNotNull)):
        operator = "IS NULL" if isinstance(spec, FilterNull) else "IS NOT NULL"
        return f"{column} {operator}", {}, "source_predicate"
    if isinstance(spec, FilterIsIn):
        values = spec.values
        if not values:
            return "1 = 0", {}, "source_predicate"
    elif isinstance(spec, FilterRange):
        values = (spec.low, spec.high)
    else:
        values = (spec.value,)
    kind = _scalar_kind(schema.field(spec.column).type, values)
    if kind is None or (
        isinstance(spec, (FilterComparison, FilterRange)) and kind in {"Bool", "String"}
    ):
        return None, {}, "unsupported_predicate_type"

    # Compare textual output values, not the native Enum/UUID/IP ordering or
    # implicit conversion rules. Keep NULL when ClickHouse removes Nullable
    # during a CAST by default.
    if kind == "String":
        target = "Nullable(String)" if dialect == "clickhouse" else "STRING"
        column = f"CAST({column} AS {target})"

    parameters: dict[str, TransformScalar] = {}
    literals = []
    for index, value in enumerate(values):
        if dialect == "clickhouse":
            name = f"tributo_filter_{ordinal}_{index}"
            parameters[name] = value
            literals.append(f"{{{name}:{kind}}}")
        elif isinstance(value, str):
            literals.append(f"UNHEX('{value.encode('utf-8').hex()}')")
        elif isinstance(value, bool):
            literals.append("TRUE" if value else "FALSE")
        elif isinstance(value, float):
            literals.append(f"CAST({value!r} AS DOUBLE)")
        else:
            literals.append(str(value))

    if isinstance(spec, FilterIsIn):
        predicate = f"{column} IN ({', '.join(literals)})"
    elif isinstance(spec, FilterRange):
        predicate = f"({column} >= {literals[0]}) AND ({column} <= {literals[1]})"
    else:
        if isinstance(spec, FilterEq):
            operator = "="
        elif isinstance(spec, FilterNotEq):
            operator = "!="
        else:
            operator = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[spec.operator]
        predicate = f"{column} {operator} {literals[0]}"
    return predicate, parameters, "source_predicate"


def compile_sql_filters(
    pipeline: TransformPipeline, schema: pa.Schema, dialect: _Dialect
) -> _SqlFilters:
    """Compile a supported prefix after the complete pipeline was validated.

    Callers must first use ConcreteTransformCompiler on the source-projected
    schema. No predicate is moved across an unsupported or non-filter step.
    SQL and parameter values remain ephemeral and are excluded from receipts.
    """
    predicates = []
    parameters: dict[str, TransformScalar] = {}
    levels: dict[int, PushdownLevel] = {}
    reasons = []
    barrier = False
    for ordinal, spec in enumerate(pipeline.steps):
        if barrier:
            reasons.append("after_transform_barrier")
            continue
        predicate, values, reason = _render_filter(spec, schema, dialect, ordinal)
        reasons.append(reason)
        if predicate is None:
            barrier = True
            continue
        levels[ordinal] = "exact"
        predicates.append(f"({predicate})")
        parameters.update(values)
    decisions = tuple(
        item.model_copy(update={"diagnostic": reasons[item.ordinal]})
        for item in classify_transform_decisions(pipeline, levels)
    )
    return _SqlFilters(
        predicate=" AND ".join(predicates) or None,
        parameters=parameters,
        decisions=decisions,
    )
