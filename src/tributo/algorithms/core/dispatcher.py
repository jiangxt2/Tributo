"""Bounded Dispatcher façade and Run Coordinator."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmRequest,
    AlgorithmRunResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
    WorkerExecutionResult,
)
from tributo.algorithms.core.planner import AlgorithmPlanner
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    InputResolverPort,
    InputRuntimeAdapter,
    PortableRuntimeAdapter,
    RuntimeExecutionEnvelope,
    RuntimeInputBinding,
)
from tributo.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)


@DeveloperAPI
class AlgorithmRunCoordinator:
    """Own Driver input/runtime resources around one already-resolved plan."""

    def __init__(
        self,
        *,
        resolvers: Mapping[str, InputResolverPort],
        input_adapters: Mapping[str, InputRuntimeAdapter],
        runtimes: Mapping[str, PortableRuntimeAdapter],
    ) -> None:
        self._resolvers = dict(resolvers)
        self._input_adapters = dict(input_adapters)
        self._runtimes = dict(runtimes)

    def execute(
        self,
        plan: ResolvedAlgorithmPlan,
        context: InputExecutionContext,
        artifacts: tuple[ArtifactDraft, ...] = (),
        cancelled: bool = False,
    ) -> AlgorithmRunResult:
        """Open input, invoke the selected Runtime, and close in reverse order."""
        plan.validate_integrity()
        try:
            resolver = self._resolvers[plan.input_descriptor.resolver_id]
            input_adapter = self._input_adapters[plan.input_descriptor.resolver_id]
            runtime = self._runtimes[plan.runtime.runtime_id]
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                f"missing execution component for {exc.args[0]!r}"
            ) from exc

        lease = resolver.open(
            plan.input_binding,
            plan.input_descriptor,
            context,
        )
        runtime_binding: RuntimeInputBinding | None = None
        worker_result = None
        primary_error: BaseException | None = None
        cleanup_errors: list[Exception] = []
        provenance = lease.provenance
        try:
            runtime_binding = input_adapter.bind(lease, plan)
            runtime_result = runtime.execute(
                RuntimeExecutionEnvelope(
                    plan=plan,
                    input_payloads=runtime_binding.payloads,
                    artifacts=artifacts,
                    cancelled=cancelled,
                )
            )
            if not isinstance(runtime_result, WorkerExecutionResult) or not isinstance(
                runtime_result.execution, AlgorithmExecutionResult
            ):
                raise AlgorithmExecutionError(
                    "runtime returned an invalid WorkerExecutionResult"
                )
            worker_result = runtime_result
        except BaseException as exc:
            primary_error = exc
        finally:
            if runtime_binding is not None:
                try:
                    runtime_binding.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            try:
                worker_failed = (
                    worker_result is not None
                    and worker_result.execution.status == "failed"
                )
                if primary_error is not None or worker_failed or cancelled:
                    lease.cancel()
                else:
                    lease.close()
            except Exception as exc:
                cleanup_errors.append(exc)

        if primary_error is not None:
            for cleanup_error in cleanup_errors:
                primary_error.add_note(
                    f"cleanup also failed: {type(cleanup_error).__name__}"
                )
            raise primary_error.with_traceback(primary_error.__traceback__)
        if worker_result is None:
            raise AlgorithmExecutionError(
                "runtime returned without a WorkerExecutionResult"
            )
        if cleanup_errors:
            if worker_result.execution.status == "failed":
                for cleanup_error in cleanup_errors:
                    logger.error(
                        "cleanup failed after algorithm execution failure: %s: %s",
                        type(cleanup_error).__name__,
                        "message redacted",
                    )
            else:
                first_error = cleanup_errors[0]
                raise AlgorithmExecutionError(
                    f"algorithm execution cleanup failed: {type(first_error).__name__}"
                ) from first_error

        return AlgorithmRunResult(
            run_id=uuid.uuid4().hex,
            plan_id=plan.plan_id,
            execution=worker_result.execution,
            actual_versions=worker_result.actual_versions,
            input_provenance=provenance,
            worker_metadata=worker_result.worker_metadata,
        )


@DeveloperAPI
class AlgorithmDispatcher:
    """Public bounded façade that delegates planning and execution."""

    def __init__(
        self,
        planner: AlgorithmPlanner,
        coordinator: AlgorithmRunCoordinator,
    ) -> None:
        self._planner = planner
        self._coordinator = coordinator

    def plan(
        self,
        request: AlgorithmRequest,
        context: InputResolutionContext | None = None,
    ) -> ResolvedAlgorithmPlan:
        """Resolve a request without loading code or opening runtime input."""
        return self._planner.plan(request, context)

    def explain(
        self,
        request: AlgorithmRequest,
        context: InputResolutionContext | None = None,
    ) -> dict[str, object]:
        """Return a credential-free deterministic plan projection."""
        return self._planner.explain(request, context)

    def execute(
        self,
        request: AlgorithmRequest,
        context: InputExecutionContext,
        *,
        artifacts: tuple[ArtifactDraft, ...] = (),
        resolution_context: InputResolutionContext | None = None,
        cancelled: bool = False,
    ) -> AlgorithmRunResult:
        """Plan and execute one bounded request."""
        plan = self.plan(request, resolution_context)
        return self._coordinator.execute(
            plan,
            context,
            artifacts,
            cancelled=cancelled,
        )


__all__ = ["AlgorithmDispatcher", "AlgorithmRunCoordinator"]
