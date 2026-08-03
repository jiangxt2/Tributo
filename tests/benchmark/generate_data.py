"""Deterministic benchmark dataset generator (benchmark-protocol.md §Dataset).

Schema (fixed):
    id       int64    row id
    feature_0..feature_9   float64  bounded noise
    label    int64    binary label derived from feature_0
    ts       int64    synthetic timestamp

Generation is seeded — every run reproduces the same file byte-for-byte.

Usage::

    python tests/benchmark/generate_data.py --rows 2000000 --out tests/benchmark/data/train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def generate(rows: int) -> pa.Table:
    rng = np.random.default_rng(seed=42)
    feature_0 = rng.normal(size=rows).astype(np.float64)
    columns: dict[str, Any] = {
        "id": np.arange(rows, dtype=np.int64),
        "feature_0": feature_0,
        **{
            f"feature_{i}": rng.normal(size=rows).astype(np.float64)
            for i in range(1, 10)
        },
        "label": (feature_0 > 0.0).astype(np.int64),
        "ts": 1_700_000_000_000
        + rng.integers(0, 86_400_000, size=rows).astype(np.int64),
    }
    return pa.table(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument(
        "--out", type=Path, default=Path("tests/benchmark/data/train.parquet")
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = generate(args.rows)
    pq.write_table(table, args.out, compression="zstd")
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} rows={table.num_rows} size={size_mb:.1f} MB")


if __name__ == "__main__":
    main()
