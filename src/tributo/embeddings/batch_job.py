"""Cluster-side entrypoint script for batch embedding.

Invoked as ``python -m tributo.embeddings.batch_job --source ...``
inside the Ray cluster via Jobs API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import ray
from pydantic import TypeAdapter, ValidationError

from tributo._common import configure_logging
from tributo.data import (
    CanonicalSourceInput,
    ProviderSourceConfig,
    apply_source_projection,
    source_projection,
)
from tributo.embeddings.batch_processor import Embedder
from tributo.embeddings.output_writer import write_dataset
from tributo.training.data_loader import load_ray_dataset_from_source

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed batch text embedding via Ray"
    )
    parser.add_argument(
        "--source",
        help="Canonical source configuration as a JSON object",
    )
    parser.add_argument(
        "--input",
        help="Legacy S3 input Parquet path",
    )
    parser.add_argument("--output", required=True, help="S3 output path")
    parser.add_argument(
        "--model",
        default="bge-small-zh",
        help="Short model name from registry",
    )
    parser.add_argument(
        "--text-column",
        default=None,
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


def _resolve_embedding_source(
    *,
    source_json: str | None,
    input_path: str | None,
    text_column: str | None,
) -> tuple[CanonicalSourceInput, str]:
    """Validate the canonical/legacy input shape and resolve text projection."""
    if (source_json is None) == (input_path is None):
        raise ValueError("provide exactly one of --source or --input")

    source: CanonicalSourceInput
    if source_json is not None:
        try:
            source_payload: Any = json.loads(source_json)
            source = TypeAdapter(CanonicalSourceInput).validate_python(source_payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid embedding source JSON: {exc}") from exc
    else:
        assert input_path is not None
        source = ProviderSourceConfig(
            provider="tributo.parquet",
            uri=input_path,
        )

    configured_projection = source_projection(source)
    if text_column is None:
        if configured_projection is not None and len(configured_projection) != 1:
            raise ValueError(
                "--text-column is required when the source projection has "
                "multiple columns"
            )
        text_column = configured_projection[0] if configured_projection else "text"
    source = apply_source_projection(source, [text_column])
    return source, text_column


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = _parse_args()

    configure_logging(log_format="json")

    ray.init(address="auto")

    source, text_column = _resolve_embedding_source(
        source_json=args.source,
        input_path=args.input,
        text_column=args.text_column,
    )
    provider_name = (
        source.provider if isinstance(source, ProviderSourceConfig) else source.type
    )
    logger.info("Reading input from provider=%s", provider_name)

    ds = load_ray_dataset_from_source(source.model_dump(mode="python"))

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
        fn_constructor_args=(model_path, text_column),
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
