"""Real source-side filtering on one Docker Ray node and owned SQL services."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import ray

from tests.integrations.database_filter_trace import (
    execution_records,
    setup_sql_observers,
)
from tributo.data import (
    CastColumn,
    FilterComparison,
    FilterEq,
    FilterIsIn,
    FilterNotEq,
    FilterNotNull,
    FilterNull,
    FilterRange,
    IngestionRequest,
    Limit,
    SqlPartitioning,
    SqlSourceConfig,
    TransformPipeline,
    open_ingestion,
)
from tributo.data.transform_compiler import (
    ConcreteTransformCompiler,
    TransformBackend,
    apply_pipeline_to_ray_ds,
)
from tributo.data.transform_ir import TransformSpec

pytestmark = pytest.mark.integration
_DATABASE = os.environ.get("TRIBUTO_FILTER_DATABASE", "")
_MODES = (
    ["single", "partition", "range"]
    if _DATABASE == "clickhouse"
    else ["mysql", "flight"]
)
_ARTIFACTS = Path("/workspace/tributo-work/artifacts")
_TEXT = [None, "east", "west", "", "中文", "a'\\;--\x00"]
_ROWS = [
    {
        "id": index,
        "text": _TEXT[index % len(_TEXT)],
        "score": [None, 0.1, 1.0, -0.0, 100.5][index % 5],
        "flag": [None, True, False][index % 3],
    }
    for index in range(10_000)
]


@pytest.fixture(scope="module", autouse=True)
def cluster() -> Any:
    assert os.environ.get("TRIBUTO_DOCKER_RAY_TEST") == "1", (
        "Run the owned Docker entrypoint"
    )
    assert _DATABASE in {"clickhouse", "doris"}
    setup_sql_observers()
    ray.init(
        address="auto", runtime_env={"worker_process_setup_hook": setup_sql_observers}
    )
    assert len([node for node in ray.nodes() if node["Alive"]]) == 1
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("ray", "pyarrow", "ray-doris", "ray-clickhouse")
    }
    assert versions["ray"] == "2.59.0"
    (_ARTIFACTS / "versions.json").write_text(
        json.dumps(
            {
                "versions": versions,
                "commit": os.environ["TRIBUTO_FILTER_SOURCE_SHA"],
                "source_snapshot": Path("/workspace/tributo-src/.tributo-source-ready")
                .read_text()
                .strip(),
                "ray_nodes": 1,
            },
            indent=2,
        )
        + "\n"
    )
    deadline = time.monotonic() + 240
    while True:
        try:
            if _DATABASE == "clickhouse":
                client = _clickhouse()
                assert client.query("SELECT 1").result_rows == [(1,)]
                client.close()
            else:
                connection = _mysql()
                with connection.cursor() as cursor:
                    cursor.execute("SHOW BACKENDS")
                    names = [column[0] for column in cursor.description]
                    backends = [
                        dict(zip(names, row, strict=True)) for row in cursor.fetchall()
                    ]
                    assert any(
                        str(row["Alive"]).lower() in {"true", "1"} for row in backends
                    )
                    cursor.execute("CREATE DATABASE IF NOT EXISTS tributo_filters")
                connection.close()
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    yield
    ray.shutdown()


def _clickhouse() -> Any:
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host="clickhouse",
        username="tributo",
        password=os.environ["TRIBUTO_CLICKHOUSE_PASSWORD"],
        database="tributo_filters",
    )


def _mysql() -> Any:
    import pymysql

    return pymysql.connect(host="fe", port=9030, user="root", autocommit=True)


@pytest.fixture(scope="module")
def table(cluster: Any) -> Any:
    name = "filter_fixture_" + uuid.uuid4().hex
    if _DATABASE == "clickhouse":
        client = _clickhouse()
        client.command(
            f"CREATE TABLE `{name}` (id Int64, text Nullable(String), score Nullable(Float64), flag Nullable(Bool)) ENGINE=MergeTree PARTITION BY id % 4 ORDER BY id"
        )
        client.insert(
            name, [list(row.values()) for row in _ROWS], column_names=list(_ROWS[0])
        )
        yield name
        client.command(f"DROP TABLE `{name}`")
        client.close()
    else:
        connection = _mysql()
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE tributo_filters.`{name}` (id BIGINT, text VARCHAR(1024) NULL, score DOUBLE NULL, flag BOOLEAN NULL) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 4 PROPERTIES('replication_num'='1')"
            )
            for start in range(0, len(_ROWS), 1000):
                cursor.executemany(
                    f"INSERT INTO tributo_filters.`{name}` VALUES (%s, %s, %s, %s)",
                    [tuple(row.values()) for row in _ROWS[start : start + 1000]],
                )
        yield name
        with connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE tributo_filters.`{name}`")
        connection.close()


def _source(table: str, mode: str) -> SqlSourceConfig:
    if _DATABASE == "clickhouse":
        partitioning = (
            SqlPartitioning(mode="single")
            if mode == "single"
            else (
                SqlPartitioning(mode="auto", num_partitions=2)
                if mode == "partition"
                else SqlPartitioning(mode="parallel", column="id", num_partitions=2)
            )
        )
        return SqlSourceConfig(
            dialect="clickhouse",
            host="clickhouse",
            database="tributo_filters",
            table=table,
            user="tributo",
            partitioning=partitioning,
        )
    return SqlSourceConfig(
        dialect="doris",
        host="fe",
        database="tributo_filters",
        table=table,
        user="root",
        protocol=mode,
        tablet_size=1,
    )


def _multiset(rows: list[dict[str, Any]]) -> Counter[Any]:
    return Counter(tuple(sorted(row.items())) for row in rows)


def _original_ray_rows(
    rows: list[dict[str, Any]], pipeline: TransformPipeline
) -> list[dict[str, Any]]:
    """Run the original residual compiler on fully consumed source values."""
    table = pa.Table.from_pylist(rows)
    compiled = ConcreteTransformCompiler().compile(
        pipeline, TransformBackend.RAY, table.schema
    )
    return apply_pipeline_to_ray_ds(compiled, ray.data.from_arrow(table)).take_all()


def _consume(
    source: SqlSourceConfig,
    pipeline: TransformPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], Any]:
    module = importlib.import_module(f"tributo.data.bindings.ray_{_DATABASE}")
    original = module.apply_pipeline_to_ray_ds
    calls = []

    def observe(
        compiled: Any, dataset: Any, *, ordinals: set[int] | None = None
    ) -> Any:
        assert ordinals is not None
        result = original(compiled, dataset, ordinals=ordinals)
        if not ordinals:
            assert result is dataset, (
                "An exact-only request must expose the native reader Dataset"
            )
        calls.append(set(ordinals))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(module, "apply_pipeline_to_ray_ds", observe)
        with open_ingestion(
            IngestionRequest(source=source, engine="ray", transforms=pipeline)
        ) as result:
            rows = result.handle.dataset.take_all()
            receipt = result.receipt
    assert calls
    return rows, receipt


@pytest.mark.parametrize("mode", _MODES)
def test_exact_filter_reaches_real_driver_and_native_output(
    table: str, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(table, mode)
    baseline, baseline_receipt = _consume(source, TransformPipeline(), monkeypatch)
    assert _multiset(baseline) == _multiset(_ROWS)
    pipeline = TransformPipeline(
        steps=(FilterComparison(column="id", operator="gte", value=9900),)
    )
    reference = _original_ray_rows(baseline, pipeline)
    assert _multiset(reference) == _multiset(_ROWS[9900:])
    before = {record["event_id"] for record in execution_records(table)}
    rows, receipt = _consume(
        source,
        pipeline,
        monkeypatch,
    )
    assert _multiset(rows) == _multiset(reference)
    assert receipt.schema_fingerprint == baseline_receipt.schema_fingerprint
    assert all(not item.residual_required for item in receipt.transform_decisions)
    records = [
        record
        for record in execution_records(table)
        if record["event_id"] not in before
    ]
    assert records, "No successful real data-read driver call was observed"
    assert all(
        "WHERE" in record["sql"].upper() and "`id` >= " in record["sql"]
        for record in records
    )
    if _DATABASE == "clickhouse":
        assert all(
            record["parameters"]["tributo_filter_0_0"] == 9900 for record in records
        )
    else:
        assert all("`id` >= 9900" in record["sql"] for record in records)
    evidence = {
        "database": _DATABASE,
        "mode": mode,
        "ray_nodes": 1,
        "baseline_rows": len(baseline),
        "source_rows": len(rows),
        "final_rows": len(rows),
        "residual_ordinals": [
            item.ordinal
            for item in receipt.transform_decisions
            if item.residual_required
        ],
        "evidence_kind": "successful_real_driver_execution_and_native_dataset_output",
        "queries": records,
    }
    with (_ARTIFACTS / "evidence.jsonl").open("a") as output:
        output.write(json.dumps(evidence) + "\n")


_CASES = [
    (FilterEq(column="id", value=7), lambda row: row["id"] == 7),
    (FilterNotEq(column="id", value=7), lambda row: row["id"] != 7),
    (FilterRange(column="id", low=9900, high=9999), lambda row: row["id"] >= 9900),
    (FilterIsIn(column="id", values=(0, 9999)), lambda row: row["id"] in (0, 9999)),
    (FilterEq(column="text", value="中文"), lambda row: row["text"] == "中文"),
    (FilterEq(column="text", value=""), lambda row: row["text"] == ""),
    (
        FilterEq(column="text", value="a'\\;--\x00"),
        lambda row: row["text"] == "a'\\;--\x00",
    ),
    (
        FilterNotEq(column="text", value="east"),
        lambda row: row["text"] is not None and row["text"] != "east",
    ),
    (
        FilterIsIn(column="text", values=("east", "west")),
        lambda row: row["text"] in ("east", "west"),
    ),
    (FilterNull(column="text"), lambda row: row["text"] is None),
    (FilterNotNull(column="text"), lambda row: row["text"] is not None),
    (
        FilterComparison(column="score", operator="gte", value=80),
        lambda row: row["score"] is not None and row["score"] >= 80,
    ),
    (
        FilterRange(column="score", low=0.1, high=1.0),
        lambda row: row["score"] is not None and 0.1 <= row["score"] <= 1.0,
    ),
    (FilterEq(column="flag", value=True), lambda row: row["flag"] is True),
    (FilterIsIn(column="id", values=()), lambda row: False),
]


@pytest.mark.parametrize(
    ("spec", "matches"),
    _CASES,
    ids=[spec.type + "-" + str(index) for index, (spec, _) in enumerate(_CASES)],
)
def test_supported_predicates_match_original_values(
    table: str, spec: TransformSpec, matches: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = "single" if _DATABASE == "clickhouse" else "mysql"
    rows, receipt = _consume(
        _source(table, mode), TransformPipeline(steps=(spec,)), monkeypatch
    )
    assert _multiset(rows) == _multiset([row for row in _ROWS if matches(row)])
    assert receipt.transform_decisions[0].pushdown_level == "exact"


def test_cast_barrier_keeps_original_order(
    table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = "single" if _DATABASE == "clickhouse" else "mysql"
    pipeline = TransformPipeline(
        steps=(
            CastColumn(column="id", target_type="string"),
            FilterEq(column="id", value="9900"),
        )
    )
    rows, receipt = _consume(_source(table, mode), pipeline, monkeypatch)
    expected = [{**_ROWS[9900], "id": "9900"}]
    assert _multiset(rows) == _multiset(expected)
    assert all(item.residual_required for item in receipt.transform_decisions)


def test_limit_barrier_and_mixed_residual(
    table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = "single" if _DATABASE == "clickhouse" else "mysql"
    rows, receipt = _consume(
        _source(table, mode),
        TransformPipeline(
            steps=(
                FilterComparison(column="id", operator="gte", value=9900),
                Limit(count=10),
                FilterComparison(column="id", operator="gte", value=0),
            )
        ),
        monkeypatch,
    )
    assert len(rows) == 10 and all(row["id"] >= 9900 for row in rows)
    assert [item.pushdown_level for item in receipt.transform_decisions] == [
        "exact",
        "none",
        "none",
    ]


def test_no_matching_rows_preserve_schema(
    table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A source-side false predicate exercises the connector's real empty-read
    # path, including Doris's schema-carrying zero-row task.
    mode = "single" if _DATABASE == "clickhouse" else "mysql"
    rows, receipt = _consume(
        _source(table, mode),
        TransformPipeline(
            steps=(FilterComparison(column="id", operator="gt", value=100_000),)
        ),
        monkeypatch,
    )
    assert rows == []
    assert receipt.schema_fingerprint


def test_actual_empty_table_preserves_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    table = "filter_fixture_empty_" + uuid.uuid4().hex
    if _DATABASE == "clickhouse":
        client = _clickhouse()
        client.command(
            f"CREATE TABLE `{table}` (id Int64) ENGINE=MergeTree ORDER BY id"
        )
    else:
        client = _mysql()
        with client.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE tributo_filters.`{table}` (id BIGINT) DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1 PROPERTIES('replication_num'='1')"
            )
    try:
        mode = "single" if _DATABASE == "clickhouse" else "mysql"
        rows, receipt = _consume(
            _source(table, mode),
            TransformPipeline(
                steps=(FilterComparison(column="id", operator="gte", value=0),)
            ),
            monkeypatch,
        )
        assert rows == []
        assert receipt.schema_fingerprint
    finally:
        if _DATABASE == "clickhouse":
            client.command(f"DROP TABLE `{table}`")
        else:
            with client.cursor() as cursor:
                cursor.execute(f"DROP TABLE tributo_filters.`{table}`")
        client.close()
