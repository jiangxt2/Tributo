"""Real Ray HDFS Parquet conformance for the optional HDFS runtime profile."""

from __future__ import annotations

import os

import pyarrow as pa
import pytest
import ray

from tributo.data import (
    IngestionRequest,
    IngestionRuntimeContext,
    ParquetSourceConfig,
    ReadOptions,
    open_ingestion,
    ray_worker_distribution_probe,
)
from tributo.exceptions import DataSourceError

pytestmark = pytest.mark.integration


def _request() -> IngestionRequest:
    return IngestionRequest(
        source=ParquetSourceConfig(
            path=(
                f"hdfs://{os.environ.get('TRIBUTO_HDFS_HOST', 'hdfs')}:"
                f"{os.environ.get('TRIBUTO_HDFS_PORT', '8020')}"
                "/tributo-hdfs/parquet"
            ),
            columns=["id", "part"],
        ),
        engine="ray",
        read_options=ReadOptions(target_parallelism=8),
    )


def test_hdfs_parquet_runs_on_worker_and_preserves_multiple_blocks() -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    driver_ip = ray.util.get_node_ip_address()
    context = IngestionRuntimeContext(
        distribution_probe=ray_worker_distribution_probe,
        require_worker_validation=True,
    )

    with open_ingestion(_request(), context) as result:
        dataset = result.handle.dataset.materialize()
        # Ray may consolidate small files into one reader task; this gate
        # requires worker execution and multiple materialized blocks.
        assert dataset.num_blocks() >= 2
        rows = dataset.take_all()
        assert len(rows) == 1_600
        assert {row["part"] for row in rows} == set(range(8))
        assert result.receipt.binding_id == "tributo.ray.parquet.hdfs"
        assert result.receipt.reader_api == "ray.data.read_parquet"
        assert result.receipt.transport_id == "hdfs"
        evidence = {
            item.distribution_name: item for item in result.receipt.component_versions
        }
        assert evidence["ray"].worker_validation_complete is True
        assert set(evidence["ray"].worker_versions) == {"2.55.1"}

        def record_node(batch: pa.Table) -> pa.Table:
            node = ray.util.get_node_ip_address()
            return pa.table({"node": [node] * batch.num_rows})

        nodes = dataset.map_batches(record_node, batch_format="pyarrow").take_all()
        assert nodes
        assert {row["node"] for row in nodes} != {driver_ip}


def test_hdfs_parquet_unreachable_fails_closed() -> None:
    ray.init(address="auto", ignore_reinit_error=True)
    context = IngestionRuntimeContext()
    request = IngestionRequest(
        source=ParquetSourceConfig(path="hdfs://missing-hdfs:8020/not-found"),
        engine="ray",
    )
    with pytest.raises(DataSourceError):
        with open_ingestion(request, context):
            pass


if __name__ == "__main__":
    test_hdfs_parquet_runs_on_worker_and_preserves_multiple_blocks()
    test_hdfs_parquet_unreachable_fails_closed()
    print("HDFS Parquet ingestion conformance passed")
