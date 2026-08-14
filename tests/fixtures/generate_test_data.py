"""Generate synthetic test data for CI and local development.

This script generates test datasets in Parquet format for various Tributo
workload types. It requires no external services (S3, MinIO, Redis) — all
data is written to local disk.

Usage::

    # Generate all dataset types (default: --output-dir /tmp/tributo-test-data)
    python scripts/generate_test_data.py

    # Generate with custom output directory
    python scripts/generate_test_data.py --output-dir ./ci-data

    # Control dataset size
    python scripts/generate_test_data.py --scale small

Dataset types:
    training  : Tabular data for XGBoost/PU Learning tests (numeric + label columns)
    serving   : (no-op placeholder — serving tests use programmatic request dicts)

Scale presets:
    small  : ~1,000 records per dataset (fast smoke tests)
    medium : ~10,000 records per dataset (CI unit tests, default)
    large  : ~100,000 records per dataset (local perf testing)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# ── numeric feature generators ──────────────────────────────────────────────

_CATEGORIES: list[str] = ["A", "B", "C", "D", "E"]


def _generate_training_dataset(num_records: int, output_path: str) -> int:
    """Generate a tabular dataset for XGBoost/PU Learning tests."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(42)

    data = {
        "id": list(range(num_records)),
        "feature_a": rng.normal(0, 1, num_records).tolist(),
        "feature_b": rng.normal(5, 2, num_records).tolist(),
        "feature_c": rng.uniform(0, 1, num_records).tolist(),
        "feature_d": rng.choice([0, 1], num_records, p=[0.7, 0.3]).tolist(),
        "feature_e": rng.choice(_CATEGORIES, num_records).tolist(),
        # imbalanced labels (~5 % positive) — mirrors real-world PU Learning use cases
        "label": rng.choice([0, 1], num_records, p=[0.95, 0.05]).tolist(),
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="zstd", compression_level=3)
    file_size = os.path.getsize(output_path)
    print(
        f"  [training]  {num_records:,} rows → {output_path} ({file_size / 1024:.1f} KB)"
    )
    return file_size


# ── scale presets ───────────────────────────────────────────────────────────

_SCALES: dict[str, int] = {
    "small": 1_000,
    "medium": 10_000,
    "large": 100_000,
}

# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for Tributo CI and development.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/tributo-test-data",
        help="Directory for generated files (default: /tmp/tributo-test-data)",
    )
    parser.add_argument(
        "--type",
        choices=["all", "training"],
        default="all",
        help="Dataset type to generate (default: all)",
    )
    parser.add_argument(
        "--scale",
        choices=list(_SCALES),
        default="medium",
        help="Number of records per dataset (default: medium)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    num_records = _SCALES[args.scale]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test data (scale={args.scale}, {num_records:,} rows each)")
    print(f"Output directory: {output_dir}\n")

    total_bytes = 0
    dataset_types = ["training"] if args.type == "all" else [args.type]

    for ds_type in dataset_types:
        filename = f"{ds_type}_{args.scale}.parquet"
        filepath = str(output_dir / filename)
        if ds_type == "training":
            total_bytes += _generate_training_dataset(num_records, filepath)

    print(f"\nDone. Total size: {total_bytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
