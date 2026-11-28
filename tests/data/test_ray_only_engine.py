"""Public request validation for the supported ingestion and write engine."""

import pytest
from pydantic import ValidationError

from tributo.data import IngestionRequest, ParquetSourceConfig, WriteMode, WriteRequest


@pytest.mark.parametrize("engine", ["daft", "tributo.daft"])
def test_removed_daft_engine_is_rejected_for_reads_and_writes(engine: str) -> None:
    with pytest.raises(ValidationError, match="engine must be one of"):
        IngestionRequest(source=ParquetSourceConfig(path="data.parquet"), engine=engine)
    with pytest.raises(ValidationError, match="engine must be one of"):
        WriteRequest(
            engine=engine,
            target_kind="parquet",
            target="output",
            mode=WriteMode.APPEND,
        )


@pytest.mark.parametrize("engine", ["ray", "tributo.ray_data"])
def test_ray_engine_aliases_remain_accepted(engine: str) -> None:
    read = IngestionRequest(
        source=ParquetSourceConfig(path="data.parquet"), engine=engine
    )
    write = WriteRequest(
        engine=engine, target_kind="parquet", target="output", mode=WriteMode.APPEND
    )
    assert read.engine == write.engine == "tributo.ray_data"
