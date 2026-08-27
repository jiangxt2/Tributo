"""Single-worker resource budget tests.

Covers the algorithm-agnostic bounded collector: budget defaults, in-flight
bytes accounting, concat-copy peak estimation, fail-fast before unbounded
concat, row guards (never silently truncate), schema-level preflight
estimation and structured error context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from tributo.exceptions import ResourceBudgetExceededError
from tributo.training.resource import (
    MIB,
    BoundedCollector,
    ResourceBudget,
    collect_bounded,
    estimate_row_bytes_from_schema,
    preflight_check,
)


def _df(n_rows: int) -> pd.DataFrame:
    """Small int64 DataFrame — 8 bytes per row (plus fixed overhead)."""
    return pd.DataFrame({"a": np.arange(n_rows, dtype=np.int64)})


def _table(n_rows: int) -> pa.Table:
    """Small int64 pyarrow Table — exactly 8 bytes per row (nbytes linear)."""
    return pa.table({"a": np.arange(n_rows, dtype=np.int64)})


class TestResourceBudgetDefaults:
    """预算默认启用，默认值不允许为 None。"""

    def test_defaults_are_always_active(self):
        budget = ResourceBudget()
        assert budget.max_batch_bytes == 64 * MIB
        assert budget.max_worker_materialization_bytes == 1024 * MIB
        assert budget.max_input_rows_per_worker is None

    def test_rejects_zero_budgets(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResourceBudget(max_batch_bytes=0)
        with pytest.raises(ValidationError):
            ResourceBudget(max_worker_materialization_bytes=0)


class TestBoundedCollector:
    """加载中累计 bytes，加入前检查，超限不 append 不 concat。"""

    def test_collect_within_budget_returns_summary(self):
        budget = ResourceBudget(
            max_batch_bytes=64 * MIB, max_worker_materialization_bytes=1024 * MIB
        )
        batches, summary = collect_bounded(
            [_table(3), _table(5)],
            budget,
            algorithm="pu",
            split="train",
            worker_rank=0,
        )
        assert summary.rows_seen == 8
        assert summary.payload_bytes == 8 * 8  # int64 nbytes 精确线性
        assert summary.estimated_peak_bytes >= summary.payload_bytes
        assert len(batches) == 2

    def test_single_batch_over_max_batch_bytes_fails_before_append(self):
        budget = ResourceBudget(
            max_batch_bytes=10, max_worker_materialization_bytes=10**9
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            collect_bounded(
                [_df(100)],
                budget,
                algorithm="dnn",
                split="train",
                worker_rank="rank-0",
            )
        assert excinfo.value.budget_bytes == 10
        assert excinfo.value.split == "train"

    def test_accumulated_across_batches_fails_before_concat(self):
        # 预算 = 2×单批 payload（含 concat 输出副本）：第一批通过，第二批峰值超限。
        budget = ResourceBudget(
            max_batch_bytes=10**9,
            max_worker_materialization_bytes=2 * _df(10).memory_usage(deep=True).sum(),
        )
        collector = BoundedCollector(
            budget, algorithm="dnn", split="train", worker_rank=0
        )
        collector.add(_df(10))  # peak = 2×payload(10) ≤ 预算
        with pytest.raises(ResourceBudgetExceededError):
            collector.add(_df(10))  # peak = 2×payload(20) > 预算 → 不 append
        assert collector.rows_seen == 10  # 第二个 batch 未被加入

    def test_row_guard_fails_fast_and_never_truncates(self):
        budget = ResourceBudget(
            max_batch_bytes=10**9,
            max_worker_materialization_bytes=10**9,
            max_input_rows_per_worker=6,
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            collect_bounded(
                [_df(4), _df(4)],
                budget,
                algorithm="dnn",
                split="val",
                worker_rank=0,
            )
        assert excinfo.value.observed_rows == 8  # 不截断为 6
        assert excinfo.value.max_rows == 6
        assert "truncated" in str(excinfo.value) or "truncat" in str(excinfo.value)

    def test_stricter_row_guard_wins(self):
        budget = ResourceBudget(max_input_rows_per_worker=100)
        collector = BoundedCollector(
            budget, algorithm="xgboost", split="train", worker_rank=0, max_rows=5
        )
        collector.add(_df(4))
        with pytest.raises(ResourceBudgetExceededError):
            collector.add(_df(4))  # 4+4=8 > 5（显式 max_rows 更严格）

    def test_shared_collector_across_splits_accounts_together(self):
        # DNN train/val 共享 worker 预算。
        # 预算 = 2×6 行 int64（含 concat 副本）：train+val 通过，第三个超限。
        budget = ResourceBudget(
            max_batch_bytes=10**9,
            max_worker_materialization_bytes=6 * 8 * 2,  # 96
        )
        collector = BoundedCollector(
            budget, algorithm="dnn", split="train", worker_rank=0
        )
        collector.add(_table(3), split="train")  # peak = 2×24 = 48 ≤ 96 ✓
        collector.add(_table(3), split="val")  # peak = 2×48 = 96 ≤ 96 ✓
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            collector.add(_table(3), split="val")  # peak = 2×72 > 96 → 标注 split
        assert excinfo.value.split == "val"

    def test_error_context_carries_all_diagnostics(self):
        budget = ResourceBudget(
            max_batch_bytes=10**9,
            max_worker_materialization_bytes=100,
            max_input_rows_per_worker=1000,
        )
        collector = BoundedCollector(
            budget, algorithm="pu", split="train", worker_rank="rank-3"
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            collector.add(_df(50))
        err = excinfo.value
        assert err.algorithm == "pu"
        assert err.split == "train"
        assert err.worker_rank == "rank-3"
        assert err.observed_bytes > 0
        assert err.budget_bytes == 100
        assert err.observed_rows == 50
        assert err.max_rows == 1000
        assert "rank-3" in str(err)

    def test_unsupported_batch_type_raises(self):
        collector = BoundedCollector(
            ResourceBudget(), algorithm="pu", split="train", worker_rank=0
        )
        with pytest.raises(TypeError):
            collector.add("not-a-batch")

    def test_empty_resource_config_uses_defaults(self):
        """worker 内 ``config.get("resource") or {}`` 的解析路径：空配置 = 默认预算。"""
        budget = ResourceBudget.model_validate({})
        assert budget.max_batch_bytes == 64 * MIB
        assert budget.max_worker_materialization_bytes == 1024 * MIB


class TestPreflightEstimate:
    """加载前 schema 级估算，明显超限拒绝。"""

    def test_estimate_row_bytes_from_pyarrow_schema(self):
        schema = pa.schema(
            [
                ("id", pa.int64()),  # 8
                ("score", pa.float32()),  # 4
                ("name", pa.string()),  # 32 (variable-width default)
                ("tags", pa.list_(pa.string())),  # 64 (complex default)
                ("flag", pa.bool_()),  # 1
                ("ts", pa.timestamp("ms")),  # 8
            ]
        )
        assert estimate_row_bytes_from_schema(schema) == 8 + 4 + 32 + 64 + 1 + 8

    def test_estimate_row_bytes_accepts_ray_style_schema_wrapper(self):
        class FakeRaySchema:
            base_schema = pa.schema([("a", pa.int64())])

        assert estimate_row_bytes_from_schema(FakeRaySchema()) == 8

    def test_estimate_row_bytes_none_for_unsupported(self):
        assert estimate_row_bytes_from_schema(None) is None
        assert estimate_row_bytes_from_schema("not-a-schema") is None

    def test_preflight_rejects_obvious_over_budget(self):
        budget = ResourceBudget(
            max_batch_bytes=10**9, max_worker_materialization_bytes=10**6
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            preflight_check(
                rows=10**9,  # 10 亿行 × 8B = 8GB > 1MiB
                row_bytes=8,
                budget=budget,
                algorithm="dnn",
                split="train",
                worker_rank=0,
            )
        assert excinfo.value.observed_bytes == 8 * 10**9

    def test_preflight_skips_when_rows_unknown(self):
        budget = ResourceBudget(max_worker_materialization_bytes=10**6)
        # rows=None → 不报错，加载中 collector 兜底
        preflight_check(
            rows=None,
            row_bytes=8,
            budget=budget,
            algorithm="dnn",
            split="train",
            worker_rank=0,
        )

    def test_preflight_rejects_single_row_over_batch_budget(self):
        budget = ResourceBudget(max_batch_bytes=10)
        with pytest.raises(ResourceBudgetExceededError):
            preflight_check(
                rows=10,
                row_bytes=32,  # 单行即超单批预算
                budget=budget,
                algorithm="pu",
                split="train",
                worker_rank=0,
            )

    def test_preflight_rejects_absurd_row_even_when_rows_unknown(self):
        """单行超预算不依赖行数——worker 不知道行数时仍拒绝。"""
        budget = ResourceBudget(max_batch_bytes=8)
        with pytest.raises(ResourceBudgetExceededError, match="single row"):
            preflight_check(
                rows=None,  # worker 场景：行数未知
                row_bytes=32,
                budget=budget,
                algorithm="dnn",
                split="train",
                worker_rank=0,
            )


class TestConcatCopyPeak:
    """estimated_peak_bytes 必须覆盖 concat 输出副本（≈2×payload）。"""

    def test_peak_models_concat_output_copy(self):
        budget = ResourceBudget(
            max_batch_bytes=10**9, max_worker_materialization_bytes=10**9
        )
        collector = BoundedCollector(
            budget, algorithm="pu", split="train", worker_rank=0
        )
        collector.add(_table(3))
        collector.add(_table(5))
        summary = collector.summary
        assert summary.payload_bytes == 8 * 8  # int64 nbytes 精确线性
        # concat 时输入列表与输出并存 → 峰值 ≈ 2 × payload
        assert summary.estimated_peak_bytes == 2 * summary.payload_bytes

    def test_concat_peak_exceeds_budget_while_payload_fits(self):
        """回归：payload 在预算内、但 concat 峰值（2×）超限时仍须失败。"""
        budget = ResourceBudget(
            max_batch_bytes=10**9,
            max_worker_materialization_bytes=6 * 8,  # 可容纳 payload 48
        )
        with pytest.raises(ResourceBudgetExceededError) as excinfo:
            collect_bounded(
                [_table(6)],  # payload = 48 ≤ 48，但峰值 96 > 48
                budget,
                algorithm="pu",
                split="train",
                worker_rank=0,
            )
        assert excinfo.value.observed_bytes == 2 * 6 * 8  # 峰值 96
        assert excinfo.value.budget_bytes == 6 * 8
