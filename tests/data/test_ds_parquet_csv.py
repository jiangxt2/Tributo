"""Checks for the row comparison used by file ingestion conformance."""

from tests.data.ingestion_conformance import row_multiset


def test_row_comparison_preserves_multiplicity_and_normalizes_nan() -> None:
    rows = [{"id": 1, "value": float("nan")}, {"id": 2, "value": None}]
    assert row_multiset(rows) == row_multiset(list(reversed(rows)))
    assert row_multiset(rows) != row_multiset([rows[0], rows[0]])
