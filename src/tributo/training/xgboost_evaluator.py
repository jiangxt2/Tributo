"""XGBoost training evaluation utilities: data filtering, splitting, metric extraction."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import ray.data

logger = logging.getLogger(__name__)


def filter_invalid_labels(
    ds: ray.data.Dataset,
    label_col: str = "label",
    invalid_value: int = -1,
) -> ray.data.Dataset:
    """Filter samples with invalid labels.

    Args:
        ds: Input dataset.
        label_col: Label column name.
        invalid_value: Label value to filter out.

    Returns:
        Filtered dataset.
    """
    import pyarrow as pa

    def _filter(batch: pa.Table) -> pa.Table:
        mask = batch.column(label_col).to_numpy() != invalid_value
        return batch.filter(mask)

    before = ds.count()
    ds = ds.map_batches(_filter, batch_format="pyarrow")
    after = ds.count()
    logger.info(
        "Label filter applied (label==%s): %d rows before, %d rows after",
        invalid_value,
        before,
        after,
    )
    return ds


def split_dataset(
    ds: ray.data.Dataset,
    val_size: float = 0.2,
    test_size: float = 0.0,
    seed: int = 42,
) -> tuple[ray.data.Dataset, ray.data.Dataset | None, ray.data.Dataset | None]:
    """Three-way split: training / validation / test sets.

    Args:
        ds: Input dataset.
        val_size: Validation set proportion, 0 means no split.
        test_size: Test set proportion, 0 means no split.
        seed: Random seed.

    Returns:
        (train_ds, val_ds, test_ds), with None for unused splits.
    """
    if val_size <= 0 and test_size <= 0:
        return ds, None, None

    ds = ds.random_shuffle(seed=seed)
    train_frac = 1.0 - val_size - test_size

    if test_size > 0 and val_size > 0:
        train_ds, val_ds, test_ds = ds.split_proportionately([train_frac, val_size])
        return train_ds, val_ds, test_ds
    elif val_size > 0:
        train_ds, val_ds = ds.split_proportionately([train_frac])
        return train_ds, val_ds, None
    else:
        train_ds, test_ds = ds.split_proportionately([train_frac])
        return train_ds, None, test_ds


def compute_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert raw metrics reported by Ray Train into a serializable summary.

    Args:
        metrics: Metrics dictionary from ray.train.report.

    Returns:
        Recursively JSON-safe metrics. Numeric values are converted to floats;
        scalar metadata and nested metric arrays/mappings are preserved.
    """

    def json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("training metrics must contain only finite numbers")
            return numeric
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]

        # numpy/pandas scalars and arrays expose one of these conversion hooks.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if converted is not value:
                return json_safe(converted)
        item = getattr(value, "item", None)
        if callable(item):
            converted = item()
            if converted is not value:
                return json_safe(converted)
        raise TypeError(
            f"training metric value {type(value).__name__!r} is not JSON-safe"
        )

    return {str(key): json_safe(value) for key, value in metrics.items()}
