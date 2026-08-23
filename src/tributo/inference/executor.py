"""Ray Data implementation of the InferenceExecutor contract."""

from __future__ import annotations

import logging
from typing import ClassVar, Literal, TypedDict

from tributo.data import IngestionPlanReceipt
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference.contracts import (
    BoundResultSink,
    FailureDiagnostic,
    InferenceResult,
    PreparedInferencePlan,
    PreparedModelProvenance,
    ResolvedInference,
    ResultSink,
    ResultSinkReceipt,
    ResultSinkRequest,
)
from tributo.inference.input_resolver import (
    IngestionGatewayInputResolver,
    InputResolverPort,
    OpenedInferenceInput,
)
from tributo.inference.kernel import (
    KernelBatchPredictor,
    ModelKernelProvider,
    PredictionKernelFactory,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
class RayMapBatchesExecutor:
    """Execute a pinned inference plan with resident Ray Data actors."""

    api_version: ClassVar[int] = 1
    executor_id: ClassVar[str] = "ray-map-batches-v1"

    def __init__(
        self,
        input_resolver: InputResolverPort | None = None,
        *,
        kernel_provider: ModelKernelProvider | None = None,
        kernel_factory: PredictionKernelFactory | None = None,
    ) -> None:
        self._inputs = input_resolver or IngestionGatewayInputResolver()
        self._kernel_provider = kernel_provider
        self._kernel_factory = kernel_factory

    def execute(self, plan: ResolvedInference, sink: ResultSink) -> InferenceResult:
        """Compatibility facade that opens data before core execution."""
        return self.execute_bound(
            plan,
            _RequestBoundResultSink(sink, plan.result_sink),
        )

    def execute_bound(
        self,
        plan: ResolvedInference,
        sink: BoundResultSink,
    ) -> InferenceResult:
        """Open data through the compatibility port and execute a bound sink."""
        plan, prepared, kernel_factory = self._prepare_compatibility_execution(
            plan, sink
        )
        try:
            opened = self._inputs.open(plan.input)
        except Exception as exc:
            return _failed_result(prepared, phase="acquisition", exc=exc)
        return self._execute_opened(prepared, sink, opened, kernel_factory)

    def execute_prepared(
        self,
        plan: PreparedInferencePlan,
        sink: BoundResultSink,
        opened: OpenedInferenceInput,
        *,
        kernel_factory: PredictionKernelFactory | None = None,
    ) -> InferenceResult:
        """Execute prepared data, kernel, and output ports without resolving them."""
        plan = self._validate_prepared_execution(plan, sink)
        effective_factory = kernel_factory or self._kernel_factory
        if effective_factory is None:
            raise ValueError("prepared execution requires an injected kernel_factory")
        return self._execute_opened(plan, sink, opened, effective_factory)

    def _prepare_compatibility_execution(
        self, plan: ResolvedInference, sink: BoundResultSink
    ) -> tuple[ResolvedInference, PreparedInferencePlan, PredictionKernelFactory]:
        # Rebuild the shallow-frozen transport before data access.  This
        # rejects post-resolution mutation of nested source/options mappings
        # and gives the executor an isolated, fully revalidated plan snapshot.
        plan = ResolvedInference.model_validate(plan.model_dump(mode="python"))
        prepared = prepared_inference_plan(plan)
        prepared = self._validate_prepared_execution(prepared, sink)
        kernel_factory = self._kernel_factory or (
            self._kernel_provider or _default_model_kernel_provider()
        ).prediction_factory(plan.model)
        return plan, prepared, kernel_factory

    def _validate_prepared_execution(
        self, plan: PreparedInferencePlan, sink: BoundResultSink
    ) -> PreparedInferencePlan:
        plan = PreparedInferencePlan.model_validate(plan.model_dump(mode="python"))
        if plan.execution.executor_id != self.executor_id:
            raise ValueError(
                f"Executor {self.executor_id!r} cannot execute "
                f"{plan.execution.executor_id!r}"
            )
        if sink.sink_id != plan.output_port_id:
            raise ValueError(
                f"ResultSink {sink.sink_id!r} cannot write {plan.output_port_id!r}"
            )
        return plan

    def _execute_opened(
        self,
        plan: PreparedInferencePlan,
        sink: BoundResultSink,
        opened: OpenedInferenceInput,
        kernel_factory: PredictionKernelFactory,
    ) -> InferenceResult:
        """Build and materialize the Ray graph over a prepared input."""
        try:
            import ray.data

            predicted = opened.dataset.map_batches(
                KernelBatchPredictor,
                fn_constructor_args=(
                    kernel_factory,
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
    plan: PreparedInferencePlan,
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


class _RequestBoundResultSink:
    """Compatibility adapter binding one legacy ResultSink request."""

    api_version: ClassVar[int] = 1

    def __init__(self, sink: ResultSink, request: ResultSinkRequest) -> None:
        if sink.sink_id != request.sink_id:
            raise ValueError(
                f"ResultSink {sink.sink_id!r} cannot write {request.sink_id!r}"
            )
        self._sink = sink
        self._request = request
        self._sink_id = str(sink.sink_id)

    @property
    def sink_id(self) -> str:
        return self._sink_id

    def write(
        self,
        dataset: object,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt:
        return self._sink.write(
            dataset,
            self._request,
            run_id=run_id,
            plan_digest=plan_digest,
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
    plan: PreparedInferencePlan,
    ingestion_receipt: IngestionPlanReceipt | None = None,
) -> _ResultIdentity:
    source_ref = plan.source_ref_id
    if ingestion_receipt is not None:
        source_ref = ingestion_receipt.source_ref
    return {
        "run_id": plan.run_id,
        "attempt_id": plan.attempt_id,
        "submission_id": plan.submission_id,
        "parent_run_id": plan.parent_run_id,
        "plan_digest": plan.plan_digest,
        "bundle_id": plan.model_provenance.model_id,
        "manifest_sha256": plan.model_provenance.version_digest,
        "role": plan.model_provenance.role,
        "flavor_id": plan.model_provenance.runtime_id,
        "source_ref_id": source_ref,
    }


def _cancel_safely(opened: OpenedInferenceInput) -> None:
    try:
        opened.cancel()
    except Exception:
        # Preserve the primary execution failure. The ingestion implementation
        # owns cleanup diagnostics and must keep callbacks idempotent.
        pass


def _default_model_kernel_provider() -> ModelKernelProvider:
    """Resolve the compatibility provider through top-level composition."""
    from tributo.runtime import default_model_kernel_provider

    return default_model_kernel_provider()


@PublicAPI(stability="alpha")
def prepared_inference_plan(plan: ResolvedInference) -> PreparedInferencePlan:
    """Strip compatibility data/model/output requests from a resolved plan."""
    plan = ResolvedInference.model_validate(plan.model_dump(mode="python"))
    return PreparedInferencePlan(
        plan_digest=plan.plan_digest,
        model_provenance=PreparedModelProvenance(
            model_id=plan.model.bundle_ref.bundle_id,
            version_digest=plan.model.bundle_ref.manifest_sha256,
            role=plan.model.role,
            runtime_id=plan.model.flavor_id,
        ),
        source_ref_id=plan.input.descriptor.source_ref,
        output_port_id=plan.result_sink.sink_id,
        input_binding=plan.input_binding,
        output_binding=plan.output_binding,
        execution=plan.execution,
        run_id=plan.run_id,
        attempt_id=plan.attempt_id,
        submission_id=plan.submission_id,
        parent_run_id=plan.parent_run_id,
    )


__all__ = ["RayMapBatchesExecutor", "prepared_inference_plan"]
