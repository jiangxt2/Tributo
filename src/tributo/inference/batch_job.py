"""Cluster-side entrypoint: batch inference job.

Executed inside the Ray cluster, submitted via Jobs API:
    python -m tributo.inference.batch_job --config inference.json
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
from typing import TYPE_CHECKING

import ray

from tributo._common.logging import bootstrap_logging, emit_event
from tributo._common.logging_policy import driver_logging_runtime_env

if TYPE_CHECKING:
    from tributo.inference.contracts import InferenceResult

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed batch inference via Ray Data + ONNX Runtime"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        help="Path to inference JSON config file",
    )
    source.add_argument(
        "--resolved-plan-env",
        help="Environment variable containing a base64 encoded ResolvedInference",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = _parse_args()

    bootstrap_logging()

    ray.init(address="auto", **driver_logging_runtime_env())

    try:
        if args.resolved_plan_env is not None:
            inference_result = _run_resolved_plan(args.resolved_plan_env)
            if inference_result.status != "succeeded":
                failure = inference_result.failure
                emit_event(
                    logger,
                    "tributo.stage.finished",
                    level=(
                        logging.INFO
                        if inference_result.status == "cancelled"
                        else logging.ERROR
                    ),
                    phase=failure.phase if failure is not None else "batch_driver",
                    status=inference_result.status,
                    error_type=failure.error_type if failure is not None else None,
                    run_id=inference_result.run_id,
                    attempt_id=inference_result.attempt_id,
                )
                return 1
        else:
            from tributo.inference.pipeline import run_inference_from_json

            run_inference_from_json(args.config)
        emit_event(
            logger, "tributo.stage.finished", phase="batch_driver", status="succeeded"
        )
        return 0
    except Exception as exc:
        logger.error("Inference job failed (%s)", type(exc).__name__)
        return 1


def _run_resolved_plan(env_name: str) -> InferenceResult:
    """Validate and execute an immutable plan transported by Ray Jobs."""
    from tributo.inference.api import run_resolved_inference
    from tributo.inference.contracts import ResolvedInference

    raw = os.environ.get(env_name)
    if raw is None:
        raise ValueError(f"Required environment variable {env_name!r} is not set")
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8")
    except Exception:
        raise ValueError("Resolved inference plan is not valid base64 UTF-8") from None
    plan = ResolvedInference.model_validate_json(decoded)
    if os.environ.get("TRIBUTO_RUN_ID") != plan.run_id:
        raise ValueError("Resolved plan run_id conflicts with TRIBUTO_RUN_ID")
    if os.environ.get("TRIBUTO_ATTEMPT_ID") != plan.attempt_id:
        raise ValueError("Resolved plan attempt_id conflicts with TRIBUTO_ATTEMPT_ID")
    if os.environ.get("TRIBUTO_SUBMISSION_ID") != plan.submission_id:
        raise ValueError(
            "Resolved plan submission_id conflicts with TRIBUTO_SUBMISSION_ID"
        )
    return run_resolved_inference(plan)


if __name__ == "__main__":
    sys.exit(main())
