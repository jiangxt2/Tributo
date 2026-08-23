"""Unit tests for the explainability data-persistence adapter."""

from __future__ import annotations

from types import SimpleNamespace

from tributo.explainability.protocols import ExplainabilityResultStore
from tributo.integrations.sinks import explainability as result_store_module
from tributo.integrations.sinks.explainability import (
    ParquetExplainabilityResultStore,
)


def test_parquet_result_store_materializes_through_the_sink_port(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def write(self, dataset, request, *, run_id, plan_digest):
        del self
        calls.update(
            dataset=dataset,
            request=request,
            run_id=run_id,
            plan_digest=plan_digest,
        )

    monkeypatch.setattr(result_store_module.ParquetResultSink, "write", write)
    monkeypatch.setattr(
        result_store_module,
        "inspect_parquet_output",
        lambda uri, *, storage_profile: SimpleNamespace(
            digest="a" * 64,
            total_bytes=17,
            rows=3,
        ),
    )
    store = ParquetExplainabilityResultStore()

    materialized = store.materialize(
        object(),
        uri="/tmp/explanations",
        storage_profile=None,
        max_bytes=100,
        run_id="operation-1",
        plan_digest="b" * 64,
    )

    assert isinstance(store, ExplainabilityResultStore)
    assert calls["request"].uri == "/tmp/explanations"
    assert calls["request"].max_bytes == 100
    assert calls["run_id"] == "operation-1"
    assert materialized.digest == "a" * 64
    assert materialized.total_bytes == 17
    assert materialized.rows == 3
