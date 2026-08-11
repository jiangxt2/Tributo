"""Ray Data implementation of the InferenceExecutor contract."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal, TypedDict

from tributo.data import IngestionPlanReceipt
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference.bundle_predictor import BundleBatchPredictor
from tributo.inference.contracts import (
    FailureDiagnostic,
    InferenceResult,
    ResolvedInference,
    ResultSink,
    ResultSinkReceipt,
)
from tributo.inference.input_resolver import (
    IngestionGatewayInputResolver,
    InputResolverPort,
    OpenedInferenceInput,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
class RayMapBatchesExecutor:
    """Execute a pinned inference plan with resident Ray Data actors."""

    api_version: ClassVar[int] = 1
    executor_id: ClassVar[str] = "ray-map-batches-v1"

    def __init__(self, input_resolver: InputResolverPort | None = None) -> None:
        self._inputs = input_resolver or IngestionGatewayInputResolver()

    def execute(self, plan: ResolvedInference, sink: ResultSink) -> InferenceResult:
        """Run the lazy Ray graph once, when the sink writes its output."""
        # Rebuild the shallow-frozen transport before data access.  This
        # rejects post-resolution mutation of nested source/options mappings
        # and gives the executor an isolated, fully revalidated plan snapshot.
        plan = ResolvedInference.model_validate(plan.model_dump(mode="python"))
        if plan.execution.executor_id != self.executor_id:
            raise ValueError(
                f"Executor {self.executor_id!r} cannot execute "
                f"{plan.execution.executor_id!r}"
            )
        if sink.sink_id != plan.result_sink.sink_id:
            raise ValueError(
                f"ResultSink {sink.sink_id!r} cannot write {plan.result_sink.sink_id!r}"
            )

        try:
            opened = self._inputs.open(plan.input)
        except Exception as exc:
            return _failed_result(plan, phase="acquisition", exc=exc)

        try:
            import ray.data

            predicted = opened.dataset.map_batches(
                BundleBatchPredictor,
                fn_constructor_args=(
                    plan.model,
                    plan.input_binding,
                    plan.output_binding,
                ),
                batch_format="numpy",
                batch_size=plan.execution.batch_size,
                compute=ray.data.ActorPoolStrategy(size=plan.execution.concurrency),
                num_cpus=plan.execution.num_cpus_per_actor,
                num_gpus=plan.execution.num_gpus_per_actor,
            )
        except Exception as exc:
            _cancel_safely(opened)
            return _failed_result(
                plan,
                phase="execution",
                exc=exc,
                ingestion_receipt=opened.receipt,
            )

        try:
            receipt = sink.write(
                predicted,
                plan.result_sink,
                run_id=plan.run_id,
                plan_digest=plan.plan_digest,
            )
        except ResultWriteError as exc:
            _cancel_safely(opened)
            return _failed_result(
                plan,
                phase="sink",
                exc=exc,
                ingestion_receipt=opened.receipt,
            )
        except ResultMaterializationError as exc:
            _cancel_safely(opened)
            return _failed_result(
                plan,
                phase="materialization",
                exc=exc,
                ingestion_receipt=opened.receipt,
            )
        except Exception as exc:
            _cancel_safely(opened)
            return _failed_result(
                plan,
                phase="sink",
                exc=exc,
                ingestion_receipt=opened.receipt,
            )

        try:
            opened.close()
        except Exception as exc:
            logger.warning(
                "Inference input close failed after result commit (%s); "
                "preserving succeeded status",
                type(exc).__name__,
            )

        return InferenceResult(
            **_result_identity(plan, opened.receipt),
            ingestion_receipt=opened.receipt,
            sink_receipt=receipt,
            output_rows=receipt.rows_written,
            status="succeeded",
        )


def _failed_result(
    plan: ResolvedInference,
    *,
    phase: Literal["acquisition", "execution", "materialization", "sink"],
    exc: BaseException,
    ingestion_receipt: IngestionPlanReceipt | None = None,
    sink_receipt: ResultSinkReceipt | None = None,
) -> InferenceResult:
    return InferenceResult(
        **_result_identity(plan, ingestion_receipt),
        ingestion_receipt=ingestion_receipt,
        sink_receipt=sink_receipt,
        status="failed",
        retryable=False,
        failure=FailureDiagnostic(
            phase=phase,
            code=f"inference_{phase}_failed",
            error_type=getattr(exc, "source_error_type", type(exc).__name__),
            retryable=False,
        ),
    )


class _ResultIdentity(TypedDict):
    run_id: str
    attempt_id: str
    submission_id: str
    parent_run_id: str | None
    plan_digest: str
    bundle_id: str
    manifest_sha256: str
    role: str
    flavor_id: str
    source_ref_id: str


def _result_identity(
    plan: ResolvedInference,
    ingestion_receipt: IngestionPlanReceipt | None = None,
) -> _ResultIdentity:
    source_ref = plan.input.descriptor.source_ref
    if ingestion_receipt is not None:
        source_ref = ingestion_receipt.source_ref
    return {
        "run_id": plan.run_id,
        "attempt_id": plan.attempt_id,
        "submission_id": plan.submission_id,
        "parent_run_id": plan.parent_run_id,
        "plan_digest": plan.plan_digest,
        "bundle_id": plan.model.bundle_ref.bundle_id,
        "manifest_sha256": plan.model.bundle_ref.manifest_sha256,
        "role": plan.model.role,
        "flavor_id": plan.model.flavor_id,
        "source_ref_id": source_ref,
    }


def _cancel_safely(opened: OpenedInferenceInput) -> None:
    try:
        opened.cancel()
    except Exception:
        # Preserve the primary execution failure. The ingestion implementation
        # owns cleanup diagnostics and must keep callbacks idempotent.
        pass


__all__ = ["RayMapBatchesExecutor"]
