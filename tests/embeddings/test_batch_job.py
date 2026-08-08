"""Tests for canonical source handling in the embedding job entrypoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tributo.data import DaftDataFrameHandle, RayDataHandle
from tributo.data.source_config import ProviderSourceConfig
from tributo.embeddings.batch_job import (
    _open_embedding_dataset,
    _parse_args,
    _resolve_embedding_source,
)


def test_parse_args_accepts_canonical_source() -> None:
    args = _parse_args(
        [
            "--source",
            '{"provider":"tributo.parquet","uri":"data.parquet"}',
            "--output",
            "out.lance",
        ]
    )
    assert args.source is not None
    assert args.input is None
    assert args.text_column is None
    assert args.engine == "ray"


def test_source_resolution_applies_native_text_projection() -> None:
    source, text_column = _resolve_embedding_source(
        source_json='{"provider":"tributo.parquet","uri":"data.parquet"}',
        input_path=None,
        text_column="content",
    )
    assert text_column == "content"
    assert source.options == {"columns": ["content"]}


def test_source_resolution_rejects_missing_text_column_for_multi_projection() -> None:
    with pytest.raises(ValueError, match="text-column"):
        _resolve_embedding_source(
            source_json=(
                '{"provider":"tributo.parquet","uri":"data.parquet",'
                '"options":{"columns":["id","text"]}}'
            ),
            input_path=None,
            text_column=None,
        )


def test_source_resolution_requires_exactly_one_input() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _resolve_embedding_source(
            source_json=None,
            input_path=None,
            text_column="text",
        )


@pytest.mark.parametrize("engine", ["ray", "daft"])
def test_embedding_dataset_uses_explicit_engine_and_adapter(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ray_dataset = object()
    result = MagicMock()
    result.handle = (
        RayDataHandle(ray_dataset) if engine == "ray" else DaftDataFrameHandle(object())
    )
    opened: list[Any] = []

    def open_ingestion(request: Any) -> MagicMock:
        opened.append(request)
        return result

    adaptation = MagicMock(handle=RayDataHandle(ray_dataset))
    monkeypatch.setattr("tributo.embeddings.batch_job.open_ingestion", open_ingestion)
    adapt = MagicMock(return_value=adaptation)
    monkeypatch.setattr("tributo.embeddings.batch_job.adapt_daft_result_to_ray", adapt)

    with _open_embedding_dataset(
        ProviderSourceConfig(provider="tributo.parquet", uri="data.parquet"),
        engine,
    ) as dataset:
        assert dataset is ray_dataset
        result.close.assert_not_called()

    assert opened[0].engine == (
        "tributo.ray_data" if engine == "ray" else "tributo.daft"
    )
    assert adapt.call_count == (1 if engine == "daft" else 0)
    result.close.assert_called_once_with()
