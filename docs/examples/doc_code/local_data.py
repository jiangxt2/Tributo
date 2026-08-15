"""Read and write local Parquet through Tributo's bounded-data gateways."""

from __future__ import annotations

import argparse

import ray

from tributo.data import (
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
    WriteMode,
    open_ingestion,
)
from tributo.data.writing import WriteReceipt, WriteRequest, default_write_gateway


def read_local_parquet(input_path: str) -> None:
    """Print schema and planning evidence for a local Parquet dataset."""
    opened = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=input_path),
            engine="ray",
        )
    )
    try:
        assert isinstance(opened.handle, RayDataHandle)
        print(opened.handle.dataset.schema())
        print(opened.receipt)
    finally:
        opened.close()


def write_local_parquet(input_path: str, output_path: str) -> WriteReceipt:
    """Copy a local Parquet dataset through the native write gateway."""
    opened = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path=input_path),
            engine="ray",
        )
    )
    try:
        request = WriteRequest(
            engine="ray",
            target_kind="parquet",
            target=output_path,
            mode=WriteMode.OVERWRITE,
            options={"compression": "zstd"},
        )
        return default_write_gateway().execute(request, opened.handle)
    finally:
        opened.close()


def main() -> None:
    """Run the local read and write example."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    try:
        read_local_parquet(args.input_path)
        receipt = write_local_parquet(args.input_path, args.output_path)
        print(receipt)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
