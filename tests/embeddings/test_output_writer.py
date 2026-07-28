"""Tests for embeddings/output_writer.py."""

from __future__ import annotations

import pyarrow as pa
import pytest

from tributo.embeddings.output_writer import _has_vector_column


def test_has_vector_column_fixed_size_list():
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("embedding", pa.list_(pa.float32(), 512)),
        ]
    )
    assert _has_vector_column(schema) is True


def test_has_vector_column_large_list():
    schema = pa.schema(
        [
            ("name", pa.string()),
            ("vec", pa.large_list(pa.float64())),
        ]
    )
    assert _has_vector_column(schema) is True


def test_has_vector_column_no_vector():
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("text", pa.string()),
            ("score", pa.float32()),
        ]
    )
    assert _has_vector_column(schema) is False


def test_has_vector_column_int_list_not_vector():
    schema = pa.schema(
        [
            ("tags", pa.list_(pa.int64())),
        ]
    )
    assert _has_vector_column(schema) is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
