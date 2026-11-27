"""Exercise Doris writes through the public gateway on the owned Ray cluster."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pymysql
import pytest
import ray

from tributo.data import RayDataHandle
from tributo.data.writing import (
    WriteAmbiguousError,
    WriteMode,
    WriteRequest,
    default_write_gateway,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database() -> Iterator[str]:
    assert os.environ.get("TRIBUTO_DOCKER_RAY_TEST") == "1"
    assert os.environ.get("TRIBUTO_FILTER_DATABASE") == "doris"
    ray.init(address="auto")
    name = "write_fixture_" + uuid.uuid4().hex
    connection = pymysql.connect(host="fe", port=9030, user="root", autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{name}`")
            cursor.execute(
                f"CREATE TABLE `{name}`.`events` "
                "(id BIGINT NOT NULL, category VARCHAR(64), score INT) "
                "DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1 "
                "PROPERTIES ('replication_num'='1')"
            )
            cursor.execute(
                f"CREATE TABLE `{name}`.`crash_events` "
                "(id BIGINT NOT NULL, category VARCHAR(64), score INT) "
                "DUPLICATE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1 "
                "PROPERTIES ('replication_num'='1')"
            )
            cursor.execute(
                f"CREATE TABLE `{name}`.`unique_events` "
                "(id BIGINT NOT NULL, category VARCHAR(64), score INT) "
                "UNIQUE KEY(id) DISTRIBUTED BY HASH(id) BUCKETS 1 "
                "PROPERTIES ('replication_num'='1', 'enable_unique_key_merge_on_write'='true')"
            )
        yield name
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{name}`")
        finally:
            connection.close()
            ray.shutdown()


def _write(
    database: str, table: str, operation: str, rows: dict[str, list[Any]]
) -> Any:
    request = WriteRequest(
        engine="ray",
        target_kind="doris",
        target=f"doris://fe:8030/{database}/{table}",
        mode=WriteMode.APPEND,
        options={
            "operation": operation,
            "batch_rows": 1,
            "redirect_hosts": [socket.gethostbyname("be")],
            "redirect_ports": [8040],
        },
    )
    receipt = default_write_gateway().execute(
        request, RayDataHandle(ray.data.from_arrow(pa.table(rows)))
    )
    assert receipt.committed is True
    assert receipt.binding_id == "tributo.ray.doris"
    assert receipt.target_ref == request.target
    assert receipt.rows_written == len(rows["id"])
    assert receipt.bytes_written is not None and receipt.bytes_written > 0
    assert receipt.metadata["native_status"] == "success"
    assert receipt.metadata["attempted_rows"] == len(rows["id"])
    assert receipt.metadata["filtered_rows"] == 0
    return receipt


def _query(database: str, table: str) -> list[tuple[Any, ...]]:
    connection = pymysql.connect(host="fe", port=9030, user="root", database=database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT id, category, score FROM `{table}` ORDER BY id")
            return list(cursor.fetchall())
    finally:
        connection.close()


def test_load_rows_and_receipt(database: str) -> None:
    receipt = _write(
        database,
        "events",
        "load",
        {"id": [1, 2], "category": ["a", "b"], "score": [10, 20]},
    )
    assert receipt.metadata["batches"] == 2
    assert _query(database, "events") == [(1, "a", 10), (2, "b", 20)]


def test_upsert_replaces_existing_row(database: str) -> None:
    _write(
        database,
        "unique_events",
        "upsert",
        {"id": [11], "category": ["old"], "score": [1]},
    )
    _write(
        database,
        "unique_events",
        "upsert",
        {"id": [11], "category": ["new"], "score": [2]},
    )
    assert [row for row in _query(database, "unique_events") if row[0] == 11] == [
        (11, "new", 2)
    ]


def test_partial_update_preserves_omitted_column(database: str) -> None:
    _write(
        database,
        "unique_events",
        "upsert",
        {"id": [12], "category": ["old"], "score": [10]},
    )
    _write(
        database, "unique_events", "partial_update", {"id": [12], "category": ["new"]}
    )
    assert [row for row in _query(database, "unique_events") if row[0] == 12] == [
        (12, "new", 10)
    ]


def _exit_worker_after_doris_load() -> None:
    """Inject a worker loss only after Doris confirms the load."""
    from ray_doris.write.stream_load import StreamLoadClient

    original = StreamLoadClient.load

    def load_then_exit(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if result.status != "Success":
            raise RuntimeError("The injected load did not receive confirmation")
        os._exit(74)

    client_type: Any = StreamLoadClient
    client_type.load = load_then_exit


def test_worker_crash_after_load_is_ambiguous(database: str) -> None:
    ray.shutdown()
    ray.init(
        address="auto",
        runtime_env={"worker_process_setup_hook": _exit_worker_after_doris_load},
    )
    try:
        with pytest.raises(WriteAmbiguousError) as captured:
            _write(
                database,
                "crash_events",
                "load",
                {"id": [99], "category": ["committed"], "score": [7]},
            )
        assert captured.value.source_error_type == "WorkerCrashedError"
        assert _query(database, "crash_events") == [(99, "committed", 7)]
    finally:
        ray.shutdown()
        ray.init(address="auto")
