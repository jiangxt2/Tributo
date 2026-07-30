"""DS1/DS2: Parquet + CSV correctness — Daft vs PyArrow canonical baseline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_pq
import pytest

# ---------------------------------------------------------------------------
# Test data generator
# ---------------------------------------------------------------------------


def make_canonical_table() -> pa.Table:
    """Fixed test table covering int/float/string/bool/date/null/NaN."""
    import pyarrow.compute as pc

    t = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "x": [10, 30, 50, 70, 90, 110],
            "score": [0.5, 1.5, 2.5, 3.5, 4.5, float("nan")],
            "city": ["A", "B", "A", "C", "B", "A"],
            "active": [True, False, True, True, False, True],
            "name": ["alice", None, "charlie", "diana", None, "frank"],
            "ts": pc.strptime(
                pa.array(
                    [
                        "2024-01-01",
                        "2024-02-01",
                        "2024-03-01",
                        "2024-04-01",
                        "2024-05-01",
                        "2024-06-01",
                    ]
                ),
                format="%Y-%m-%d",
                unit="us",
            ),
        }
    )
    return t


# ---------------------------------------------------------------------------
# Correctness comparison helper (per plan: normalization + row_id sort)
# ---------------------------------------------------------------------------


def normalize_table(t: pa.Table) -> pa.Table:
    """Normalize schema: cast large_string → string, strip all metadata."""
    # Build normalized field list — always cast string types
    norm_fields = []
    for f in t.schema:
        target_type = f.type
        if pa.types.is_large_string(target_type):
            target_type = pa.string()
        elif pa.types.is_large_binary(target_type):
            target_type = pa.binary()
        norm_fields.append(pa.field(f.name, target_type, nullable=f.nullable))
    return t.cast(pa.schema(norm_fields))


def assert_tables_equal(
    left: pa.Table, right: pa.Table, row_id_col: str = "id"
) -> None:
    """Compare two Arrow Tables: schema normalization + row_id sorted values."""
    left = normalize_table(left)
    right = normalize_table(right)

    # Schema check: field names, types, nullable (ignore metadata from different producers)
    assert len(left.schema) == len(right.schema), (
        f"Schema length mismatch: {len(left.schema)} vs {len(right.schema)}"
    )
    for i, (lf, rf) in enumerate(zip(left.schema, right.schema)):
        assert lf.name == rf.name, f"Field {i} name: {lf.name} vs {rf.name}"
        assert lf.type == rf.type, f"Field {i} ({lf.name}) type: {lf.type} vs {rf.type}"
        assert lf.nullable == rf.nullable, f"Field {i} ({lf.name}) nullable: {lf.nullable} vs {rf.nullable}"

    # Value check: sort by row_id if present, then compare column-by-column
    # with NaN-safe equality (NaN != NaN in IEEE 754)
    if row_id_col in left.column_names and row_id_col in right.column_names:
        left = left.sort_by([(row_id_col, "ascending")])
        right = right.sort_by([(row_id_col, "ascending")])
    assert left.num_rows == right.num_rows

    for col_name in left.column_names:
        la = left.column(col_name)
        ra = right.column(col_name)
        # Use for-loop comparison for small test data;
        # NaN-safe: NaN == NaN is True for our purposes
        for i in range(left.num_rows):
            lv = la[i].as_py()
            rv = ra[i].as_py()
            if lv == rv:
                continue
            if isinstance(lv, float) and isinstance(rv, float):
                import math
                if math.isnan(lv) and math.isnan(rv):
                    continue
            raise AssertionError(
                f"Column {col_name!r} row {i}: {lv!r} != {rv!r}"
            )


# ---------------------------------------------------------------------------
# DS1: Parquet correctness
# ---------------------------------------------------------------------------


class TestDS1Parquet:
    def setup_method(self) -> None:
        import daft

        self.daft = daft
        self.canonical = make_canonical_table()
        self.tmpdir = tempfile.mkdtemp()
        self.parquet_path = str(Path(self.tmpdir) / "test.parquet")
        pa_pq.write_table(self.canonical, self.parquet_path)

    def test_full_read(self) -> None:
        """Daft read_parquet must match PyArrow canonical."""
        df = self.daft.read_parquet(self.parquet_path)
        arrow_table = df.collect().to_arrow()
        assert_tables_equal(arrow_table, self.canonical)

    def test_select_columns(self) -> None:
        """SelectColumns: Daft must match canonical with column order preserved."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.select("name", "id", "x")
        result = df.collect().to_arrow()
        expected = self.canonical.select(["name", "id", "x"])
        assert_tables_equal(result, expected)

    def test_filter_eq(self) -> None:
        """FilterEq: same rows as PyArrow filter."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.where(self.daft.col("city") == "A")
        result = df.collect().to_arrow()
        expected = self.canonical.filter(
            pa.compute.equal(self.canonical["city"], "A")
        )
        assert_tables_equal(result, expected)

    def test_filter_range(self) -> None:
        """FilterRange: low ≤ x ≤ high."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.where(
            (self.daft.col("x") >= 30) & (self.daft.col("x") <= 90)
        )
        result = df.collect().to_arrow()
        mask = pa.compute.and_(
            pa.compute.greater_equal(self.canonical["x"], 30),
            pa.compute.less_equal(self.canonical["x"], 90),
        )
        expected = self.canonical.filter(mask)
        assert_tables_equal(result, expected)

    def test_filter_isin(self) -> None:
        """FilterIsIn: set membership."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.where(self.daft.col("city").is_in(["A", "B"]))
        result = df.collect().to_arrow()
        expected = self.canonical.filter(
            pa.compute.is_in(self.canonical["city"], pa.array(["A", "B"]))
        )
        assert_tables_equal(result, expected)

    def test_filter_null(self) -> None:
        """FilterNull: only null rows."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.where(self.daft.col("name").is_null())
        result = df.collect().to_arrow()
        expected = self.canonical.filter(
            pa.compute.is_null(self.canonical["name"])
        )
        assert_tables_equal(result, expected)

    def test_empty_filter_result(self) -> None:
        """Filter with no matches → 0 rows, no exception."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.where(self.daft.col("x") == 99999)
        result = df.collect().to_arrow()
        assert result.num_rows == 0
        # Schema check with normalization (Daft uses large_string for strings)
        result_norm = normalize_table(result)
        canonical_norm = normalize_table(self.canonical)
        assert result_norm.schema == canonical_norm.schema

    def test_combined_filter_select(self) -> None:
        """Filter → SelectColumns pipeline."""
        df = self.daft.read_parquet(self.parquet_path)
        df = df.where(self.daft.col("active") == True)  # noqa: E712
        df = df.select("id", "name")
        result = df.collect().to_arrow()
        expected = self.canonical.filter(
            pa.compute.equal(self.canonical["active"], True)  # noqa: E712
        ).select(["id", "name"])
        assert_tables_equal(result, expected)


# ---------------------------------------------------------------------------
# DS1: S3/MinIO Parquet correctness
# ---------------------------------------------------------------------------


@pytest.mark.s3
class TestDS1ParquetS3:
    def setup_method(self) -> None:
        import daft
        from daft.io import IOConfig, S3Config

        self.daft = daft
        self.canonical = make_canonical_table()
        self.tmpdir = tempfile.mkdtemp()

        # Write canonical to local parquet first
        local_path = str(Path(self.tmpdir) / "canonical.parquet")
        pa_pq.write_table(self.canonical, local_path)

        # S3 config
        self.s3_cfg = IOConfig(
            s3=S3Config(
                endpoint_url="http://127.0.0.10:9000",
                key_id="minioadmin",
                access_key="minioadmin123",
                region_name="us-east-1",
                use_ssl=False,
                force_virtual_addressing=False,
            )
        )
        self.s3_path = "s3://test-bucket/ds1-s3-test.parquet"

        # Clean up stale files from previous runs before writing.
        # Daft write_parquet treats the path as a directory — left-over
        # partition files from a prior setup would cause duplicate rows.
        import s3fs

        s3 = s3fs.S3FileSystem(
            key="minioadmin",
            secret="minioadmin123",
            endpoint_url="http://127.0.0.10:9000",
        )
        if s3.exists(self.s3_path):
            s3.rm(self.s3_path, recursive=True)

        # Write to MinIO via Daft
        df = daft.from_arrow(self.canonical)
        df.write_parquet(self.s3_path, io_config=self.s3_cfg)

    def test_s3_full_read(self) -> None:
        """Daft S3 Parquet read must match canonical."""
        df = self.daft.read_parquet(self.s3_path, io_config=self.s3_cfg)
        result = df.collect().to_arrow()
        assert_tables_equal(result, self.canonical)

    def test_s3_filter_then_select(self) -> None:
        """S3: filter + select pipeline."""
        df = self.daft.read_parquet(self.s3_path, io_config=self.s3_cfg)
        df = df.where(self.daft.col("x") > 20)
        df = df.select("id", "city", "x")
        result = df.collect().to_arrow()
        expected = self.canonical.filter(
            pa.compute.greater(self.canonical["x"], 20)
        ).select(["id", "city", "x"])
        assert_tables_equal(result, expected)


# ---------------------------------------------------------------------------
# DS2: CSV correctness
# ---------------------------------------------------------------------------


class TestDS2CSV:
    def setup_method(self) -> None:
        import daft

        self.daft = daft
        self.canonical = make_canonical_table()
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = str(Path(self.tmpdir) / "test.csv")
        pa_csv.write_csv(self.canonical, self.csv_path)

    def test_full_read(self) -> None:
        """Daft CSV read must match PyArrow canonical."""
        df = self.daft.read_csv(self.csv_path)
        result = df.collect().to_arrow()
        # CSV round-trip may change type inference; normalize
        assert result.num_rows == self.canonical.num_rows
        # Columns should be present
        for col in self.canonical.column_names:
            assert col in result.column_names, f"Missing column: {col}"

    def test_select_columns(self) -> None:
        """CSV: select columns."""
        df = self.daft.read_csv(self.csv_path)
        df = df.select("id", "name")
        result = df.collect().to_arrow()
        assert result.num_rows == self.canonical.num_rows
        assert result.column_names == ["id", "name"]

    def test_filter_then_select(self) -> None:
        """CSV: filter + select pipeline."""
        df = self.daft.read_csv(self.csv_path)
        df = df.where(self.daft.col("active") == True)  # noqa: E712
        df = df.select("id", "city")
        result = df.collect().to_arrow()
        assert result.num_rows > 0
        assert result.column_names == ["id", "city"]
