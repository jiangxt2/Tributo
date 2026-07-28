"""Cluster-side entrypoint script for batch embedding.

Invoked as ``python -m tributo.embeddings.batch_job --input ...``
inside the Ray cluster via Jobs API.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import ray

from tributo._common import configure_logging
from tributo.data import get_connector
from tributo.embeddings.batch_processor import Embedder
from tributo.embeddings.output_writer import write_dataset

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed batch text embedding via Ray"
    )
    parser.add_argument("--input", required=True, help="S3 input Parquet path")
    parser.add_argument("--output", required=True, help="S3 output path")
    parser.add_argument(
        "--model",
        default="bge-small-zh",
        help="Short model name from registry",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name containing raw text",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per actor inference call",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of Embedder actors",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = _parse_args()

    configure_logging(log_format="json")

    ray.init(address="auto")

    logger.info("Reading input from %s", args.input)

    ds = get_connector("parquet").read(
        path=args.input,
        columns=[args.text_column],
    )

    logger.info(
        "Embedding with model=%s actors=%d batch_size=%d",
        args.model,
        args.concurrency,
        args.batch_size,
    )

    # Resolve model path:
    #   1. Image-bundled models at /opt/models/ (Dockerfile pre-export)
    #   2. Shared volume at /workspace/shared_models/ (runtime mount)
    image_path = Path(f"/opt/models/{args.model}")
    volume_path = Path(f"/workspace/shared_models/{args.model}")
    model_path = str(image_path if image_path.exists() else volume_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Check image build (/opt/models/) or volume mount (/workspace/shared_models/)"
        )

    ds = ds.map_batches(
        Embedder,
        fn_constructor_args=(model_path, args.text_column),
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        num_cpus=1,  # Reduce to 1 CPU/actor to lower resource usage
    )

    logger.info("Writing output to %s", args.output)
    write_dataset(ds, args.output)

    logger.info("Batch embedding complete: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
