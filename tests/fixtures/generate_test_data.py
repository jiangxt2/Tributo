"""Generate synthetic test data for CI and local development.

This script generates test datasets in Parquet format for various Tributo
workload types. It requires no external services (S3, MinIO, Redis) — all
data is written to local disk.

Usage::

    # Generate all dataset types (default: --output-dir /tmp/tributo-test-data)
    python scripts/generate_test_data.py

    # Generate only embedding test data
    python scripts/generate_test_data.py --type embedding

    # Generate with custom output directory
    python scripts/generate_test_data.py --output-dir ./ci-data

    # Control dataset size
    python scripts/generate_test_data.py --scale small

Dataset types:
    embedding : Text data for batch embedding tests (id, text, category columns)
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

# ── sample text corpus ──────────────────────────────────────────────────────
_SAMPLE_TEXTS: list[str] = [
    "Machine learning is a subfield of artificial intelligence that focuses on "
    "developing algorithms that can learn from and make predictions on data.",
    "Deep learning uses multi-layered neural networks to model complex patterns "
    "in large datasets, achieving state-of-the-art results in many domains.",
    "Natural language processing enables computers to understand, interpret, "
    "and generate human language in ways that are both meaningful and useful.",
    "Computer vision systems can identify objects, faces, and scenes in images "
    "and videos with accuracy rivaling human perception.",
    "Reinforcement learning trains agents to make sequential decisions by "
    "rewarding desired behaviors and penalizing undesired ones.",
    "Transfer learning allows models trained on one task to be repurposed for "
    "another related task, drastically reducing the data and compute needed.",
    "Ensemble methods combine predictions from multiple models to produce "
    "more accurate and robust results than any single model alone.",
    "Feature engineering is the process of using domain knowledge to extract "
    "meaningful representations from raw data for machine learning models.",
    "Model evaluation uses metrics like accuracy, precision, recall, and F1-score "
    "to quantify how well a machine learning model performs on unseen data.",
    "Hyperparameter tuning searches for the optimal configuration of a model's "
    "external parameters to maximize its predictive performance.",
    "Data augmentation artificially expands training datasets by applying "
    "transformations such as rotation, cropping, or noise injection.",
    "Federated learning trains models across decentralized devices while keeping "
    "data local, preserving privacy and reducing communication costs.",
    "Explainable AI develops techniques to make machine learning model decisions "
    "interpretable and understandable to human stakeholders.",
    "AutoML automates the end-to-end process of applying machine learning to "
    "real-world problems, from data preprocessing to model selection.",
    "Graph neural networks extend deep learning to graph-structured data, "
    "enabling applications in social networks, biology, and recommendation systems.",
    "Bayesian inference provides a principled framework for reasoning under "
    "uncertainty by updating prior beliefs with observed evidence.",
    "Causal inference goes beyond correlation to understand cause-and-effect "
    "relationships, critical for decision-making in medicine and economics.",
    "Contrastive learning trains models to distinguish between similar and "
    "dissimilar data points, producing powerful representations without labels.",
    "Diffusion models generate high-quality images by iteratively denoising "
    "random patterns, powering state-of-the-art text-to-image systems.",
    "Time series forecasting predicts future values based on historical "
    "observations, essential for finance, weather, and demand planning.",
]

# ── numeric feature generators ──────────────────────────────────────────────

_CATEGORIES: list[str] = ["A", "B", "C", "D", "E"]


def _generate_embedding_dataset(num_records: int, output_path: str) -> int:
    """Generate a text-based dataset for embedding tests."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        "id": list(range(num_records)),
        "text": [_SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)] for i in range(num_records)],
        "category": [_CATEGORIES[i % len(_CATEGORIES)] for i in range(num_records)],
        "timestamp": pd.date_range("2024-01-01", periods=num_records, freq="1min"),
    }
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="zstd", compression_level=3)
    file_size = os.path.getsize(output_path)
    print(
        f"  [embedding] {num_records:,} rows → {output_path} ({file_size / 1024:.1f} KB)"
    )
    return file_size


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
        choices=["all", "embedding", "training"],
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
    dataset_types = ["embedding", "training"] if args.type == "all" else [args.type]

    for ds_type in dataset_types:
        filename = f"{ds_type}_{args.scale}.parquet"
        filepath = str(output_dir / filename)
        if ds_type == "embedding":
            total_bytes += _generate_embedding_dataset(num_records, filepath)
        elif ds_type == "training":
            total_bytes += _generate_training_dataset(num_records, filepath)

    print(f"\nDone. Total size: {total_bytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
