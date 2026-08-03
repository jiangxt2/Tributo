"""Single-worker resource budgets and bounded materialization.

The unconditional single-worker safety baseline: every training worker
materializes input data under an explicit bytes-first budget.  A batch
that would exceed the budget fails fast *before* the unbounded
``pd.concat`` / ``pa.concat_tables`` call, and input rows are never silently
truncated.

Two-phase protection:

1. ``preflight_check()`` — rejects inputs that are obviously over budget
   before loading starts (schema-level per-row estimate × known row count).
2. ``BoundedCollector`` — accounts every batch in-flight; the budget is
   checked *before* a batch is appended, so the accumulator never holds
   more than the budget allows and the concat copy peak is accounted for.

The budget is a materialization working-set budget for the input, not a
full Ray worker RSS limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import Field

from tributo._common.config import StrictConfigModel
from tributo.exceptions import ResourceBudgetExceededError

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

# Bounded-batch hint used by trainers that previously collected with
# ``batch_size=None`` (an unbounded single batch).  The real guard is the
# byte budget; this only keeps individual batches at a sane size.
DEFAULT_BATCH_SIZE = 4096


class ResourceBudget(StrictConfigModel):
    """Bytes-first materialization budget for a single training worker.

    Defaults are always active — an unconditional safety baseline, so
    omitting configuration never disables the budget.
    """

    max_batch_bytes: int = Field(
        default=64 * MIB,
        ge=1,
        description=(
            "Upper bound for a single collected batch (bytes). "
            "A batch larger than this fails before it enters the accumulator."
        ),
    )
    max_worker_materialization_bytes: int = Field(
        default=1024 * MIB,
        ge=1,
        description=(
            "Upper bound for the worker materialization working set (bytes), "
            "covering the accumulated payload plus the concat-copy peak. "
            "Not a full Ray worker RSS limit."
        ),
    )
    max_input_rows_per_worker: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional row-count guard per worker.  Exceeding it fails fast; "
            "rows are never silently truncated."
        ),
    )


@dataclass(frozen=True)
class CollectSummary:
    """Accounting result of a bounded collection."""

    rows_seen: int
    payload_bytes: int
    estimated_peak_bytes: int


def _batch_rows(batch: Any) -> int:
    """Row count of a collected batch (pyarrow Table / pandas DataFrame)."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    if isinstance(batch, pa.Table):
        num_rows: int = batch.num_rows
        return num_rows
    if isinstance(batch, pd.DataFrame):
        return len(batch)
    if isinstance(batch, np.ndarray):
        shape: tuple[int, ...] = batch.shape
        return shape[0]
    raise TypeError(
        f"unsupported batch type for budget accounting: {type(batch).__name__}"
    )


def _batch_bytes(batch: Any) -> int:
    """In-memory payload size of a collected batch (pyarrow Table / pandas)."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    if isinstance(batch, pa.Table):
        nbytes: int = batch.nbytes
        return nbytes
    if isinstance(batch, pd.DataFrame):
        return int(batch.memory_usage(deep=True).sum())
    if isinstance(batch, np.ndarray):
        return int(batch.nbytes)
    raise TypeError(
        f"unsupported batch type for budget accounting: {type(batch).__name__}"
    )


def _field_width(type_: Any) -> int:
    """Conservative per-row byte estimate for one pyarrow field type.

    Fixed-width types use their exact byte width; variable-width fields use a
    32-byte default; complex types (list/struct/map) use a 64-byte default;
    unknown types fall back to 16 bytes.  Estimates only — the in-flight
    collector remains the hard guarantee.
    """
    import pyarrow as pa

    if pa.types.is_null(type_):
        return 0
    if pa.types.is_boolean(type_):
        return 1
    if pa.types.is_integer(type_) or pa.types.is_floating(type_):
        bit_width: int = type_.bit_width
        return bit_width // 8
    if pa.types.is_decimal(type_):
        return 16
    if pa.types.is_date32(type_) or pa.types.is_date64(type_):
        return 8
    if pa.types.is_time(type_) or pa.types.is_timestamp(type_):
        return 8
    if (
        pa.types.is_string(type_)
        or pa.types.is_large_string(type_)
        or pa.types.is_binary(type_)
        or pa.types.is_large_binary(type_)
    ):
        return 32
    if (
        pa.types.is_list(type_)
        or pa.types.is_large_list(type_)
        or pa.types.is_fixed_size_list(type_)
        or pa.types.is_struct(type_)
        or pa.types.is_map(type_)
    ):
        return 64
    return 16


def estimate_row_bytes_from_schema(schema: Any) -> int | None:
    """Estimate per-row payload bytes from a pyarrow-style schema.

    Accepts a ``pyarrow.Schema`` or a Ray Data schema exposing
    ``base_schema``.  Returns None when the schema is not pyarrow-based.

    Args:
        schema: pyarrow Schema or Ray Data Schema (``base_schema``).

    Returns:
        Estimated per-row bytes, or None when the schema is unsupported.
    """
    import pyarrow as pa

    base = None
    if isinstance(schema, pa.Schema):
        base = schema
    elif hasattr(schema, "base_schema") and isinstance(schema.base_schema, pa.Schema):
        base = schema.base_schema
    if base is None:
        return None
    return sum(_field_width(f.type) for f in base)


def preflight_check(
    *,
    rows: int | None,
    row_bytes: int | None,
    budget: ResourceBudget,
    algorithm: str,
    split: str,
    worker_rank: str | int = "driver",
) -> None:
    """Phase-1 guard: reject inputs that are obviously over budget pre-load.

    The single-row check runs without a row count — a row that alone
    exceeds the batch budget makes every batch over budget, so it fails
    regardless of how many rows there are.  The total-size check requires
    both *rows* and *row_bytes*; when either is unknown it is skipped and
    the in-flight ``BoundedCollector`` remains the hard guarantee (it never
    silently truncates).

    Args:
        rows: Estimated row count (None when unknown — avoids an eager
            ``count()`` that would execute the pipeline a second time).
        row_bytes: Estimated per-row bytes, e.g. from
            ``estimate_row_bytes_from_schema()``.
        budget: Active resource budget.
        algorithm: Trainer name, included in the error context.
        split: Dataset split name (``train``/``val``/``test``).
        worker_rank: Worker/rank context for diagnostics.

    Raises:
        ResourceBudgetExceededError: When the estimated input is obviously
            over budget (or a single row alone exceeds the batch budget).
    """
    if row_bytes is None:
        return
    if row_bytes > budget.max_batch_bytes:
        raise ResourceBudgetExceededError(
            f"{algorithm} ({split}, worker={worker_rank}): single row is "
            f"{row_bytes} bytes, already above max_batch_bytes="
            f"{budget.max_batch_bytes}; refusing to load",
            algorithm=algorithm,
            split=split,
            worker_rank=worker_rank,
            observed_bytes=row_bytes,
            budget_bytes=budget.max_batch_bytes,
            observed_rows=1,
            max_rows=budget.max_input_rows_per_worker,
        )
    if rows is None:
        return
    estimated = rows * row_bytes
    if estimated > budget.max_worker_materialization_bytes:
        raise ResourceBudgetExceededError(
            f"{algorithm} ({split}, worker={worker_rank}): estimated input "
            f"{rows} rows × {row_bytes} bytes ≈ {estimated} bytes exceeds "
            f"max_worker_materialization_bytes="
            f"{budget.max_worker_materialization_bytes}; failing before load",
            algorithm=algorithm,
            split=split,
            worker_rank=worker_rank,
            observed_bytes=estimated,
            budget_bytes=budget.max_worker_materialization_bytes,
            observed_rows=rows,
            max_rows=budget.max_input_rows_per_worker,
        )


class BoundedCollector:
    """Accumulates batches under a :class:`ResourceBudget`.

    Every batch is accounted *before* it is appended: a single batch above
    ``max_batch_bytes`` fails immediately, and a batch that would push the
    materialization peak over ``max_worker_materialization_bytes`` fails
    without being appended — the unbounded concat never runs.
    ``estimated_peak_bytes`` models the concat copy: during
    ``pd.concat`` / ``pa.concat_tables`` the input batch list and the
    concatenated output coexist, so the peak is approximately twice the
    accumulated payload (2 × (payload + newest batch)).

    Args:
        budget: Active resource budget.
        algorithm: Trainer name, included in error context.
        split: Dataset split name (``train``/``val``/``test``).
        worker_rank: Worker/rank context for diagnostics.
        max_rows: Optional explicit row guard.  When both this and
            ``budget.max_input_rows_per_worker`` are set the stricter value
            applies.
    """

    def __init__(
        self,
        budget: ResourceBudget,
        *,
        algorithm: str,
        split: str,
        worker_rank: str | int = "driver",
        max_rows: int | None = None,
    ) -> None:
        self._budget = budget
        self._algorithm = algorithm
        self._split = split
        self._worker_rank = worker_rank
        budget_rows = budget.max_input_rows_per_worker
        self._max_rows = (
            min(budget_rows, max_rows)
            if budget_rows is not None and max_rows is not None
            else budget_rows
            if budget_rows is not None
            else max_rows
        )
        self._rows_seen = 0
        self._payload_bytes = 0
        self._peak_bytes = 0

    def add(self, batch: Any, *, split: str | None = None) -> None:
        """Account one batch, failing before append when the budget is exceeded.

        One collector may span multiple splits (DNN keeps train and val in
        memory together): pass *split* on individual batches to label errors
        with the current split.

        Args:
            batch: pyarrow Table or pandas DataFrame batch.
            split: Split label for this batch; defaults to the collector's
                split.  Accounting (rows/bytes) is shared across splits.

        Raises:
            ResourceBudgetExceededError: When the batch would exceed the
                single-batch, row or materialization budget.
        """
        n_rows = _batch_rows(batch)
        batch_bytes = _batch_bytes(batch)
        # Concat-copy model: the input batch list and the concatenated
        # output coexist during pd.concat / pa.concat_tables, so the peak
        # is ~2× the accumulated payload.  Checked *before* the batch is
        # appended — the unbounded concat never runs.
        peak = (self._payload_bytes + batch_bytes) * 2
        split_label = split or self._split

        if batch_bytes > self._budget.max_batch_bytes:
            self._fail(
                split=split_label,
                reason="single batch exceeds max_batch_bytes",
                observed_bytes=batch_bytes,
                budget_bytes=self._budget.max_batch_bytes,
                observed_rows=n_rows,
            )
        if self._max_rows is not None and self._rows_seen + n_rows > self._max_rows:
            self._fail(
                split=split_label,
                reason=(
                    f"row limit exceeded "
                    f"({self._rows_seen + n_rows} > {self._max_rows}); "
                    "rows are never truncated"
                ),
                observed_bytes=peak,
                budget_bytes=None,
                observed_rows=self._rows_seen + n_rows,
            )
        if peak > self._budget.max_worker_materialization_bytes:
            self._fail(
                split=split_label,
                reason="materialization would exceed max_worker_materialization_bytes",
                observed_bytes=peak,
                budget_bytes=self._budget.max_worker_materialization_bytes,
                observed_rows=self._rows_seen + n_rows,
            )

        self._rows_seen += n_rows
        self._payload_bytes += batch_bytes
        self._peak_bytes = max(self._peak_bytes, peak)

    def add_bytes(self, nbytes: int, *, split: str | None = None) -> None:
        """Account auxiliary in-memory data that coexists with the batches.

        Used for evaluation artifacts (e.g. rank-0 test label arrays) that
        are not input batches and carry no row semantics — only the
        materialization peak is checked and updated, no row accounting.

        Args:
            nbytes: Payload size of the auxiliary data.
            split: Split label for diagnostics.

        Raises:
            ResourceBudgetExceededError: When the auxiliary data would push
                the materialization peak over the worker budget.
        """
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        split_label = split or self._split
        peak = (self._payload_bytes + nbytes) * 2
        if peak > self._budget.max_worker_materialization_bytes:
            self._fail(
                split=split_label,
                reason=(
                    "auxiliary data (e.g. evaluation labels) would exceed "
                    "max_worker_materialization_bytes"
                ),
                observed_bytes=peak,
                budget_bytes=self._budget.max_worker_materialization_bytes,
                observed_rows=self._rows_seen,
            )
        self._payload_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, peak)

    @property
    def rows_seen(self) -> int:
        """Rows accounted so far."""
        return self._rows_seen

    @property
    def summary(self) -> CollectSummary:
        """Accounting snapshot after the batches added so far."""
        return CollectSummary(
            rows_seen=self._rows_seen,
            payload_bytes=self._payload_bytes,
            estimated_peak_bytes=self._peak_bytes,
        )

    def _fail(
        self,
        *,
        split: str,
        reason: str,
        observed_bytes: int,
        budget_bytes: int | None,
        observed_rows: int,
    ) -> None:
        observed_mib = observed_bytes / MIB
        budget_text = (
            f"{budget_bytes} bytes ({budget_bytes / MIB:.1f}MiB)"
            if budget_bytes is not None
            else "unset"
        )
        raise ResourceBudgetExceededError(
            f"{self._algorithm} training (split={split}, "
            f"worker={self._worker_rank}): {reason} — would reach "
            f"{observed_bytes} bytes ({observed_mib:.1f}MiB) with "
            f"{observed_rows} rows (budget={budget_text}); "
            f"failing before materialization",
            algorithm=self._algorithm,
            split=split,
            worker_rank=self._worker_rank,
            observed_bytes=observed_bytes,
            budget_bytes=budget_bytes,
            observed_rows=observed_rows,
            max_rows=self._max_rows,
        )


def collect_bounded(
    batches: Iterable[Any],
    budget: ResourceBudget,
    *,
    algorithm: str,
    split: str,
    worker_rank: str | int = "driver",
    max_rows: int | None = None,
) -> tuple[list[Any], CollectSummary]:
    """Collect batches under the budget; returns ``(batches, summary)``.

    The returned list is ready for ``pd.concat`` / ``pa.concat_tables`` —
    it is guaranteed to be within budget, or this call raised before the
    unbounded concat could run.

    Args:
        batches: Iterable of pyarrow Table / pandas DataFrame batches.
        budget: Active resource budget.
        algorithm: Trainer name, included in error context.
        split: Dataset split name.
        worker_rank: Worker/rank context for diagnostics.
        max_rows: Optional explicit row guard (stricter of this and the
            budget value applies).

    Returns:
        Tuple of the collected batches and their accounting summary.

    Raises:
        ResourceBudgetExceededError: When any batch would exceed the budget.
    """
    collector = BoundedCollector(
        budget,
        algorithm=algorithm,
        split=split,
        worker_rank=worker_rank,
        max_rows=max_rows,
    )
    collected: list[Any] = []
    for batch in batches:
        collector.add(batch)
        collected.append(batch)
    return collected, collector.summary
