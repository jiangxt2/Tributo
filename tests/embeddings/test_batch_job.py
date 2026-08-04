"""Tests for canonical source handling in the embedding job entrypoint."""

from __future__ import annotations

import pytest

from tributo.embeddings.batch_job import _parse_args, _resolve_embedding_source


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
