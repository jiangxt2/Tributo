from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from tributo.data.bindings.ray_clickhouse import RayClickHouseBinding
from tributo.data.bindings.ray_hive import RayHiveBinding
from tributo.data.engine_binding import BindingCompileRequest
from tributo.data.ingestion import IngestionRuntimeContext, RayDataHandle, ReadOptions
from tributo.data.scan_plan import (
    ParameterizedQuery,
    SqlScan,
    SqlShardMode,
    SqlShardRequirement,
)
from tributo.data.transform_ir import TransformPipeline


class _Dataset:
    def schema(self, fetch_if_missing: bool = True) -> pa.Schema:
        del fetch_if_missing
        return pa.schema([("entity_id", pa.int64()), ("value", pa.float64())])


def _request(connector: str, runtime: dict[str, Any]) -> BindingCompileRequest:
    return BindingCompileRequest(
        plan=SqlScan(
            provider_id=f"tributo.{connector}",
            connector_id=connector,
            target=ParameterizedQuery("a" * 64),
            sharding=SqlShardRequirement(mode=SqlShardMode.AUTO),
        ),
        runtime_options=runtime,
        transforms=TransformPipeline(),
        read_options=ReadOptions(target_parallelism=4, batch_size=512),
        source_ref="b" * 64,
        runtime_context=IngestionRuntimeContext(),
    )


def test_ray_clickhouse_binding_preserves_runtime_query_and_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "tributo.data.bindings.ray_clickhouse.ClickHouseDataConnector.read",
        lambda _self, **kwargs: calls.append(kwargs) or _Dataset(),
    )

    compilation = RayClickHouseBinding().compile(
        _request(
            "clickhouse",
            {
                "host": "ch.internal",
                "database": "features",
                "sql": "SELECT * FROM training",
                "params": {"tenant": "acme"},
                "sort_key": "entity_id",
            },
        )
    )

    assert isinstance(compilation.handle, RayDataHandle)
    assert calls[0]["sql"] == "SELECT * FROM training"
    assert calls[0]["sort_key"] == "entity_id"
    assert calls[0]["parallelism"] == 4


def test_ray_hive_binding_preserves_auth_and_bounded_read_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "tributo.data.bindings.ray_hive.HiveDataConnector.read",
        lambda _self, **kwargs: calls.append(kwargs) or _Dataset(),
    )

    compilation = RayHiveBinding().compile(
        _request(
            "hive",
            {
                "host": "hive.internal",
                "database": "warehouse",
                "sql": "SELECT * FROM training",
                "auth": "NOSASL",
                "hash_column": "entity_id",
            },
        )
    )

    assert isinstance(compilation.handle, RayDataHandle)
    assert calls[0]["auth"] == "NOSASL"
    assert calls[0]["hash_column"] == "entity_id"
    assert calls[0]["batch_size"] == 512
    assert calls[0]["parallelism"] == 4
