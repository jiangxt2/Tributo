"""Reusable semantic assertions for every bounded ingestion Connector."""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Mapping, Sequence

from tributo.data.ingestion import IngestionOpenResult


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, Decimal):
        return ("decimal", str(value))
    if isinstance(value, (datetime, date, time)):
        return (type(value).__name__, value.isoformat())
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _canonical_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item) for item in value)
    return value


def row_multiset(rows: Sequence[Mapping[str, Any]]) -> Counter[Any]:
    """Normalize rows without assuming physical order or Python container types."""
    return Counter(_canonical_value(dict(row)) for row in rows)


def assert_dual_engine_conformance(
    ray_result: IngestionOpenResult,
    ray_rows: Sequence[Mapping[str, Any]],
    daft_result: IngestionOpenResult,
    daft_rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    limit: int | None = None,
    require_worker_validation: bool = True,
) -> None:
    """Assert the cross-engine invariants shared by all Connector fixtures."""
    assert row_multiset(ray_rows) == row_multiset(daft_rows)
    assert row_multiset(ray_rows) == row_multiset(expected_rows)
    assert ray_result.receipt.schema_fingerprint == (
        daft_result.receipt.schema_fingerprint
    )
    assert ray_result.receipt.transform_digest == (daft_result.receipt.transform_digest)
    assert ray_result.receipt.dataset_ref == daft_result.receipt.dataset_ref
    for result in (ray_result, daft_result):
        decisions = result.receipt.transform_decisions
        assert tuple(item.ordinal for item in decisions) == tuple(range(len(decisions)))
        assert all(item.compiled_result == "residual" for item in decisions)
        assert all(item.residual_required for item in decisions)
        if require_worker_validation:
            assert all(
                item.worker_validation_complete
                for item in result.receipt.component_versions
            )
        else:
            assert all(
                item.driver_version for item in result.receipt.component_versions
            )
        serialized = result.receipt.model_dump_json()
        assert "secret_access_key" not in serialized
        assert "access_key_id" not in serialized
    if limit is not None:
        assert len(ray_rows) <= limit
        for result in (ray_result, daft_result):
            limit_decision = result.receipt.transform_decisions[-1]
            assert limit_decision.transform_type == "limit"
            assert limit_decision.pushdown_level == "none"


__all__ = ["assert_dual_engine_conformance", "row_multiset"]
