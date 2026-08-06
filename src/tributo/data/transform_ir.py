"""Versioned, engine-neutral transform contracts for bounded ingestion."""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal, Mapping, TypeAlias, Union, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from tributo.data.refs import digest
from tributo.util.annotations import PublicAPI

TransformScalar: TypeAlias = str | int | float | bool | Decimal | datetime | date | time


def _validate_scalar(value: TransformScalar) -> TransformScalar:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Transform scalar floats must be finite")
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("Transform scalar decimals must be finite")
    return value


def _serialize_scalar(value: TransformScalar) -> dict[str, Any]:
    """Encode scalar types explicitly so JSON round-trips preserve identity."""
    value = _validate_scalar(value)
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "value": value}
    if isinstance(value, Decimal):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"kind": "time", "value": value.isoformat()}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    raise TypeError(f"Unsupported Transform scalar: {type(value).__name__}")


def _decode_scalar(value: Any) -> Any:
    """Decode tagged JSON while retaining ergonomic Python scalar construction."""
    if not isinstance(value, Mapping):
        return value
    if set(value) != {"kind", "value"}:
        raise ValueError("Transform scalar mappings require exactly kind and value")
    kind = value["kind"]
    raw = value["value"]
    try:
        if kind == "string" and isinstance(raw, str):
            return raw
        if kind == "bool" and type(raw) is bool:
            return raw
        if kind == "int" and type(raw) is int:
            return raw
        if kind == "float" and type(raw) in {int, float}:
            return _validate_scalar(float(raw))
        if kind == "decimal" and isinstance(raw, str):
            return Decimal(raw)
        if kind == "datetime" and isinstance(raw, str):
            return datetime.fromisoformat(raw)
        if kind == "date" and isinstance(raw, str):
            return date.fromisoformat(raw)
        if kind == "time" and isinstance(raw, str):
            return time.fromisoformat(raw)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {kind!r} Transform scalar") from exc
    raise ValueError(f"Invalid or unsupported Transform scalar kind {kind!r}")


SerializedTransformScalar: TypeAlias = Annotated[
    TransformScalar,
    BeforeValidator(_decode_scalar),
    PlainSerializer(_serialize_scalar, return_type=dict[str, Any], when_used="json"),
]


class _TransformModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@PublicAPI(stability="alpha")
class FilterEq(_TransformModel):
    """Arrow equality. ``value=None`` is illegal; use ``FilterNull``."""

    type: Literal["filter_eq"] = "filter_eq"
    column: str = Field(min_length=1)
    value: SerializedTransformScalar

    @field_validator("value", mode="before")
    @classmethod
    def _reject_null_value(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("FilterEq(value=None) is illegal; use FilterNull")
        return value

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: TransformScalar) -> TransformScalar:
        return _validate_scalar(value)


@PublicAPI(stability="alpha")
class FilterNotEq(_TransformModel):
    """Arrow inequality with NULL values excluded by three-valued logic."""

    type: Literal["filter_not_eq"] = "filter_not_eq"
    column: str = Field(min_length=1)
    value: SerializedTransformScalar

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: TransformScalar) -> TransformScalar:
        return _validate_scalar(value)


@PublicAPI(stability="alpha")
class FilterComparison(_TransformModel):
    """Ordered scalar comparison against a column."""

    type: Literal["filter_comparison"] = "filter_comparison"
    column: str = Field(min_length=1)
    operator: Literal["lt", "lte", "gt", "gte"]
    value: SerializedTransformScalar

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: TransformScalar) -> TransformScalar:
        return _validate_scalar(value)


@PublicAPI(stability="alpha")
class FilterRange(_TransformModel):
    """Inclusive range ``low <= column <= high``."""

    type: Literal["filter_range"] = "filter_range"
    column: str = Field(min_length=1)
    low: SerializedTransformScalar
    high: SerializedTransformScalar

    @field_validator("low", "high")
    @classmethod
    def _validate_bound(cls, value: TransformScalar) -> TransformScalar:
        return _validate_scalar(value)

    @model_validator(mode="after")
    def _reject_inverted_range(self) -> "FilterRange":
        try:
            if cast(Any, self.low) > cast(Any, self.high):
                raise ValueError(
                    f"FilterRange low ({self.low!r}) > high ({self.high!r})"
                )
        except TypeError as exc:
            raise ValueError(
                "FilterRange low and high must be mutually comparable"
            ) from exc
        return self


@PublicAPI(stability="alpha")
class FilterIsIn(_TransformModel):
    """Set membership; an empty tuple is always false."""

    type: Literal["filter_isin"] = "filter_isin"
    column: str = Field(min_length=1)
    values: tuple[SerializedTransformScalar, ...]

    @field_validator("values", mode="before")
    @classmethod
    def _reject_null_in_values(cls, values: Any) -> Any:
        if isinstance(values, (list, tuple)) and any(value is None for value in values):
            raise ValueError("FilterIsIn values cannot contain None; use FilterNull")
        return values

    @field_validator("values")
    @classmethod
    def _validate_values(
        cls, values: tuple[TransformScalar, ...]
    ) -> tuple[TransformScalar, ...]:
        return tuple(_validate_scalar(value) for value in values)


@PublicAPI(stability="alpha")
class FilterNull(_TransformModel):
    """Match Arrow null values, not IEEE NaN values."""

    type: Literal["filter_null"] = "filter_null"
    column: str = Field(min_length=1)


@PublicAPI(stability="alpha")
class FilterNotNull(_TransformModel):
    """Match non-null Arrow values; NaN remains a non-null value."""

    type: Literal["filter_not_null"] = "filter_not_null"
    column: str = Field(min_length=1)


def _validate_columns(columns: tuple[str, ...], transform_name: str) -> tuple[str, ...]:
    if any(not column for column in columns):
        raise ValueError(f"{transform_name} columns must be non-empty")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{transform_name} columns must not contain duplicates")
    return columns


@PublicAPI(stability="alpha")
class SelectColumns(_TransformModel):
    """Select and reorder one or more columns."""

    type: Literal["select_columns"] = "select_columns"
    columns: tuple[str, ...] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def _validate_column_names(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_columns(columns, "SelectColumns")


@PublicAPI(stability="alpha")
class DropColumns(_TransformModel):
    """Drop one or more existing columns."""

    type: Literal["drop_columns"] = "drop_columns"
    columns: tuple[str, ...] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def _validate_column_names(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_columns(columns, "DropColumns")


@PublicAPI(stability="alpha")
class ColumnRename(_TransformModel):
    """One source-to-target column rename."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_noop(self) -> "ColumnRename":
        if self.source == self.target:
            raise ValueError("ColumnRename source and target must differ")
        return self


@PublicAPI(stability="alpha")
class RenameColumns(_TransformModel):
    """Rename columns atomically while preserving declared order."""

    type: Literal["rename_columns"] = "rename_columns"
    renames: tuple[ColumnRename, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_renames(self) -> "RenameColumns":
        sources = tuple(item.source for item in self.renames)
        targets = tuple(item.target for item in self.renames)
        if len(set(sources)) != len(sources):
            raise ValueError("RenameColumns sources must be unique")
        if len(set(targets)) != len(targets):
            raise ValueError("RenameColumns targets must be unique")
        return self


@PublicAPI(stability="alpha")
class CastColumn(_TransformModel):
    """Cast one column to a canonical Arrow data type string."""

    type: Literal["cast_column"] = "cast_column"
    column: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    safe: Literal[True] = True


@PublicAPI(stability="alpha")
class FillNull(_TransformModel):
    """Replace Arrow null values in one column with a typed scalar."""

    type: Literal["fill_null"] = "fill_null"
    column: str = Field(min_length=1)
    value: SerializedTransformScalar

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: TransformScalar) -> TransformScalar:
        return _validate_scalar(value)


@PublicAPI(stability="alpha")
class Limit(_TransformModel):
    """Return at most ``count`` rows without promising row identity or order."""

    type: Literal["limit"] = "limit"
    count: int = Field(ge=0)


TransformSpec: TypeAlias = Annotated[
    Union[
        FilterEq,
        FilterNotEq,
        FilterComparison,
        FilterRange,
        FilterIsIn,
        FilterNull,
        FilterNotNull,
        SelectColumns,
        DropColumns,
        RenameColumns,
        CastColumn,
        FillNull,
        Limit,
    ],
    Field(discriminator="type"),
]


@PublicAPI(stability="alpha")
class TransformPipeline(_TransformModel):
    """Ordered, immutable Transform IR; unknown versions fail closed."""

    version: Literal[1] = 1
    steps: tuple[TransformSpec, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _require_serializable_ir(self) -> "TransformPipeline":
        self.model_dump_json()
        return self


def _typed_digest_value(value: Any) -> Any:
    """Preserve scalar types that canonical JSON would otherwise collapse."""
    if isinstance(value, Mapping):
        return {str(key): _typed_digest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_typed_digest_value(item) for item in value]
    if value is None:
        return {"kind": "null", "value": None}
    if isinstance(value, (str, int, float, bool, Decimal, datetime, date, time)):
        return _serialize_scalar(value)
    raise TypeError(f"Unsupported Transform IR digest value: {type(value).__name__}")


@PublicAPI(stability="alpha")
def transform_ir_digest(pipeline: TransformPipeline) -> str:
    """Return the versioned, type-preserving digest of a Transform pipeline."""
    return digest(
        {
            "algorithm": "tributo.transform-ir.sha256.v1",
            "ir": _typed_digest_value(pipeline.model_dump(mode="python")),
        }
    )


__all__ = [
    "CastColumn",
    "ColumnRename",
    "DropColumns",
    "FillNull",
    "FilterComparison",
    "FilterEq",
    "FilterIsIn",
    "FilterNotEq",
    "FilterNotNull",
    "FilterNull",
    "FilterRange",
    "Limit",
    "RenameColumns",
    "SelectColumns",
    "TransformPipeline",
    "transform_ir_digest",
]
