"""Exercise ClickHouse writes through the public gateway on the owned Ray node."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import clickhouse_connect
import pytest
import ray

from tributo.data import RayDataHandle
from tributo.data.writing import WriteMode, WriteRequest, default_write_gateway

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def table() -> Iterator[str]:
    assert os.environ.get("TRIBUTO_DOCKER_RAY_TEST") == "1"
    assert os.environ.get("TRIBUTO_FILTER_DATABASE") == "clickhouse"
    ray.init(address="auto")
    name = "write_fixture_" + uuid.uuid4().hex
    client = clickhouse_connect.get_client(
        host="clickhouse",
        database="tributo_filters",
        username="tributo",
        password=os.environ["TRIBUTO_CLICKHOUSE_PASSWORD"],
    )
    try:
        client.command(
            f"CREATE TABLE `{name}` (id Int64, label String) "
            "ENGINE = MergeTree ORDER BY id"
        )
        yield name
    finally:
        try:
            client.command(f"DROP TABLE IF EXISTS `{name}`")
        finally:
            client.close()
            ray.shutdown()


def test_append_rows_and_receipt(table: str) -> None:
    password = os.environ["TRIBUTO_CLICKHOUSE_PASSWORD"]
    request = WriteRequest(
        engine="ray",
        target_kind="clickhouse",
        target=f"clickhouse://clickhouse:8123/tributo_filters/{table}",
        binding_id="tributo.ray.clickhouse",
        mode=WriteMode.APPEND,
        runtime_options={
            "user": "tributo",
            "credential_ref": "env://TRIBUTO_CLICKHOUSE_PASSWORD",
        },
    )
    dataset = ray.data.from_items(
        [{"id": 1, "label": "one"}, {"id": 2, "label": "two"}],
        override_num_blocks=2,
    )
    receipt = default_write_gateway().execute(request, RayDataHandle(dataset))

    assert receipt.committed is True
    assert receipt.binding_id == "tributo.ray.clickhouse"
    assert receipt.target_ref == request.target
    assert receipt.rows_written == 2
    assert receipt.bytes_written is not None and receipt.bytes_written > 0
    assert receipt.metadata["native_status"] == "confirmed"
    assert receipt.metadata["ambiguous_batches"] == 0
    assert password not in receipt.model_dump_json()

    client: Any = clickhouse_connect.get_client(
        host="clickhouse",
        database="tributo_filters",
        username="tributo",
        password=password,
    )
    try:
        rows = client.query(f"SELECT id, label FROM `{table}` ORDER BY id").result_rows
        assert rows == [(1, "one"), (2, "two")]
    finally:
        client.close()
