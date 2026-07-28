"""Cluster-side entrypoint: batch inference job.

Executed inside the Ray cluster, submitted via Jobs API:
    python -m tributo.inference.batch_job --config inference.json
"""

from __future__ import annotations

import argparse
import logging
import sys

import ray

from tributo._common import configure_logging

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed batch inference via Ray Data + ONNX Runtime"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to inference JSON config file",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = _parse_args()

    configure_logging(log_format="json")

    ray.init(address="auto")

    from tributo.inference.pipeline import run_inference_from_json

    try:
        result = run_inference_from_json(args.config)
        logger.info("Inference completed: %s", result)
        return 0
    except Exception:
        logger.exception("Inference job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
