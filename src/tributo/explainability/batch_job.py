"""Ray Jobs entry point for distributed batch explainability."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import ray

from tributo._common import configure_logging
from tributo.explainability.contracts import ExplainabilityRequest

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tributo batch explainability")
    parser.add_argument("--config", required=True, help="Explainability JSON config")
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or _parse_args()
    configure_logging(log_format="json")
    try:
        with open(args.config, encoding="utf-8") as stream:
            request = ExplainabilityRequest.model_validate(json.load(stream))
        if request.input.engine != "tributo.ray_data":
            raise ValueError(
                "batch explainability currently requires ingestion engine 'ray'"
            )
        if request.operation_store_uri is None:
            raise ValueError(
                "batch explainability Ray Jobs require operation_store_uri"
            )
        ray.init(address="auto")
        from tributo.explainability.executor import run_batch_explainability

        receipt = run_batch_explainability(request)
        logger.info("Explainability completed: %s", receipt.model_dump(mode="json"))
        return 0
    except Exception as exc:
        logger.error(
            "Explainability job failed (%s): %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
