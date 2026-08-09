"""Framework-neutral materialized tabular input views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tributo._common.immutable import FrozenDict, deep_freeze
from tributo.algorithms.api import AlgorithmInputError
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class InMemoryTabularInputView:
    """Worker-side bounded view used by managed and user algorithms."""

    _columns: Mapping[str, tuple[object, ...]] = field(repr=False)
    feature_names: tuple[str, ...]
    label_name: str | None

    def __post_init__(self) -> None:
        try:
            columns = FrozenDict(deep_freeze(self._columns))
        except TypeError as exc:
            raise AlgorithmInputError(
                "materialized input contains a non-portable column value"
            ) from exc
        lengths = {len(values) for values in columns.values()}
        if not columns or len(lengths) != 1 or lengths == {0}:
            raise AlgorithmInputError(
                "materialized input columns must be non-empty and have equal lengths"
            )
        required = set(self.feature_names)
        if self.label_name is not None:
            required.add(self.label_name)
        missing = sorted(required - set(columns))
        if missing:
            raise AlgorithmInputError(
                f"materialized input is missing required column(s): {missing}"
            )
        object.__setattr__(self, "_columns", columns)

    @property
    def row_count(self) -> int:
        """Return the number of materialized rows."""
        return len(next(iter(self._columns.values())))

    def columns(self) -> Mapping[str, tuple[object, ...]]:
        """Return immutable column values keyed by field name."""
        return self._columns


__all__ = ["InMemoryTabularInputView"]
