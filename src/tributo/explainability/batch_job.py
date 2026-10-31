"""Ray Jobs entry point for distributed batch explainability."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import ray

from tributo._common.logging import bootstrap_logging, emit_event
from tributo._common.logging_policy import driver_logging_runtime_env
from tributo.explainability.contracts import ExplainabilityRequest

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tributo batch explainability")
    parser.add_argument("--config", required=True, help="Explainability JSON config")
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or _parse_args()
    bootstrap_logging()
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
        ray.init(address="auto", **driver_logging_runtime_env())
        from tributo.explainability.executor import run_batch_explainability

        receipt = run_batch_explainability(request)
        emit_event(
            logger,
            "tributo.stage.finished",
            phase="explainability_driver",
            status=receipt.status,
            operation_id=receipt.operation_id,
            request_id=receipt.request_id,
            bundle_id=receipt.bundle_id,
        )
        return 0
    except Exception as exc:
        emit_event(
            logger,
            "tributo.stage.finished",
            level=logging.ERROR,
            phase="explainability_driver",
            status="failed",
            error_type=type(exc).__name__,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
