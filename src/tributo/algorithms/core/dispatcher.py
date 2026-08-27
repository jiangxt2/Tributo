"""Bounded Dispatcher façade and Run Coordinator."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from tributo._common.immutable import deep_thaw
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmRequest,
    AlgorithmRunResult,
    ArtifactDraft,
    ExecutionReceipt,
    ExecutionRequest,
    ResolvedAlgorithmPlan,
    StateCoordinationEvidence,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
    WorkerResources,
)
from tributo.algorithms.core.contracts import validate_contract_value
from tributo.algorithms.core.planner import AlgorithmPlanner
from tributo.algorithms.core.runtime import RayRuntimeManager
from tributo.algorithms.spi import (
    InputExecutionContext,
    InputResolutionContext,
    InputResolverPort,
    InputRuntimeAdapter,
    PortableRuntimeAdapter,
    ResolvedInputLease,
    RuntimeExecutionEnvelope,
    RuntimeInputBinding,
    WorkerInputPayload,
    WorkerInputPayloadSet,
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
        run_id = uuid.uuid4().hex
        try:
            runtime = self._runtimes[plan.runtime.runtime_id]
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                f"missing execution component for {exc.args[0]!r}"
            ) from exc

        leases: list[tuple[str, ResolvedInputLease]] = []
        role_bindings: list[RuntimeInputBinding] = []
        runtime_binding: RuntimeInputBinding | None = None
        worker_result = None
        primary_error: BaseException | None = None
        cleanup_errors: list[Exception] = []
        try:
            for binding, descriptor in zip(
                plan.input_bindings.bindings,
                plan.input_descriptors.descriptors,
                strict=True,
            ):
                try:
                    resolver = self._resolvers[descriptor.resolver_id]
                    input_adapter = self._input_adapters[descriptor.resolver_id]
                except KeyError as exc:
                    raise AlgorithmConfigurationError(
                        f"missing execution component for {exc.args[0]!r}"
                    ) from exc
                lease = resolver.open(binding, descriptor, context)
                lease.attach_binding(binding)
                leases.append((binding.name, lease))
                role_bindings.append(input_adapter.bind(lease, plan))
            runtime_binding = self._combine_role_bindings(plan, role_bindings)
            runtime_result = runtime.execute(
                RuntimeExecutionEnvelope(
                    run_id=run_id,
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
            if (
                plan.contract_bindings is not None
                and runtime_result.execution.status == "succeeded"
            ):
                validate_contract_value(
                    plan.contract_bindings.output,
                    {
                        "status": runtime_result.execution.status,
                        "metrics": deep_thaw(runtime_result.execution.metrics),
                        "outputs": deep_thaw(runtime_result.execution.outputs),
                        "artifacts": [
                            {
                                "name": artifact.name,
                                "kind": artifact.kind,
                                "format": artifact.format,
                                "sha256": artifact.sha256,
                                "trusted": artifact.trusted,
                            }
                            for artifact in runtime_result.execution.artifacts
                        ],
                    },
                )
        except BaseException as exc:
            primary_error = exc
        finally:
            if runtime_binding is not None:
                try:
                    runtime_binding.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            for role_binding in reversed(role_bindings):
                try:
                    role_binding.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            worker_failed = (
                worker_result is not None and worker_result.execution.status == "failed"
            )
            for _, lease in reversed(leases):
                try:
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

        execution_receipt = self._execution_receipt(
            run_id,
            plan,
            worker_result,
        )
        if (
            plan.contract_bindings is not None
            and plan.contract_bindings.coverage is not None
            and execution_receipt is not None
        ):
            validate_contract_value(
                plan.contract_bindings.coverage,
                execution_receipt.to_dict(),
            )
        provenance: Mapping[str, object]
        if len(leases) == 1:
            provenance = leases[0][1].provenance
        else:
            provenance = {role: lease.provenance for role, lease in leases}
        return AlgorithmRunResult(
            run_id=run_id,
            plan_id=plan.plan_id,
            execution=worker_result.execution,
            actual_versions=worker_result.actual_versions,
            input_provenance=provenance,
            worker_metadata=worker_result.worker_metadata,
            execution_receipt=execution_receipt,
        )

    @staticmethod
    def _combine_role_bindings(
        plan: ResolvedAlgorithmPlan,
        bindings: list[RuntimeInputBinding],
    ) -> RuntimeInputBinding:
        """Combine role payloads by rank without materializing their values."""
        if not bindings:
            raise AlgorithmConfigurationError(
                "algorithm execution requires at least one input binding"
            )
        if len(bindings) == 1:
            return RuntimeInputBinding(bindings[0].payloads)
        counts = {len(binding.payloads) for binding in bindings}
        if len(counts) != 1:
            raise AlgorithmConfigurationError(
                "input roles produced different Worker partition counts"
            )
        roles = tuple(item.name for item in plan.input_bindings.bindings)
        combined_payloads: list[WorkerInputPayloadSet] = []
        for rank in range(len(bindings[0].payloads)):
            role_payloads = tuple(binding.payloads[rank] for binding in bindings)
            if any(not isinstance(item, WorkerInputPayload) for item in role_payloads):
                raise AlgorithmConfigurationError(
                    "input role adapter returned an already-composed payload"
                )
            combined_payloads.append(
                WorkerInputPayloadSet(
                    payloads=cast(tuple[WorkerInputPayload, ...], role_payloads),
                    primary_role=plan.input_bindings.primary_role,
                )
            )
        combined = tuple(combined_payloads)
        if any(
            tuple(payload.input_name for payload in payload_set.payloads) != roles
            for payload_set in combined
        ):
            raise AlgorithmConfigurationError(
                "combined Worker input payload roles drifted from the plan"
            )
        return RuntimeInputBinding(combined)

    @staticmethod
    def _execution_receipt(
        run_id: str,
        plan: ResolvedAlgorithmPlan,
        result: WorkerExecutionResult,
    ) -> ExecutionReceipt | None:
        """Build formal evidence only when the selected runtime supplies it."""
        if plan.distribution_spec is None or plan.runtime.execution_profile is None:
            return None
        metadata = result.worker_metadata
        workers_value = metadata.get("workers")
        state_value = metadata.get("state")
        if not isinstance(workers_value, (list, tuple)) or not isinstance(
            state_value, Mapping
        ):
            raise AlgorithmExecutionError(
                "formal distributed runtime did not return execution evidence"
            )
        try:
            workers = tuple(
                WorkerExecutionEvidence.from_dict(item)
                for item in workers_value
                if isinstance(item, Mapping)
            )
            if len(workers) != len(workers_value):
                raise AlgorithmConfigurationError(
                    "worker evidence entries must be mappings"
                )
            state = StateCoordinationEvidence.from_dict(state_value)
            input_complete = metadata.get("input_complete")
            driver_rows = metadata.get("driver_materialized_training_rows")
            if not isinstance(input_complete, bool):
                raise AlgorithmConfigurationError(
                    "input_complete evidence must be a boolean"
                )
            if not isinstance(driver_rows, int) or isinstance(driver_rows, bool):
                raise AlgorithmConfigurationError(
                    "driver materialization evidence must be an integer"
                )
        except (AlgorithmConfigurationError, KeyError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "formal distributed runtime returned malformed execution evidence"
            ) from exc
        artifact_ids = [artifact.sha256 for artifact in result.execution.artifacts]
        for output_name in ("bundle_id", "manifest_sha256"):
            output_value = result.execution.outputs.get(output_name)
            if isinstance(output_value, str) and output_value not in artifact_ids:
                artifact_ids.append(output_value)
        return ExecutionReceipt(
            run_id=run_id,
            plan_id=plan.plan_id,
            requested_algorithm=plan.resolution.requested_algorithm,
            canonical_algorithm=plan.resolution.algorithm,
            profile=plan.runtime.execution_profile,
            strategy=plan.distribution_spec.strategy,
            requested_worker_count=plan.runtime.worker_count,
            distributed_min_workers=plan.distribution_spec.distributed_min_workers,
            requested_resources_per_worker=WorkerResources(
                num_cpus=plan.runtime.num_cpus,
                num_gpus=plan.runtime.num_gpus,
                custom=plan.runtime.custom_resources,
            ),
            workers=workers,
            input_complete=input_complete,
            state=state,
            result_policy=plan.distribution_spec.result_policy,
            driver_materialized_training_rows=driver_rows,
            artifact_ids=tuple(artifact_ids),
            cluster_resources={},
        )


@DeveloperAPI
class AlgorithmDispatcher:
    """Public bounded façade that delegates planning and execution."""

    def __init__(
        self,
        planner: AlgorithmPlanner,
        coordinator: AlgorithmRunCoordinator,
        runtime_manager: RayRuntimeManager | None = None,
    ) -> None:
        self._planner = planner
        self._coordinator = coordinator
        self._runtime_manager = runtime_manager or RayRuntimeManager()

    def plan(
        self,
        request: AlgorithmRequest | ExecutionRequest,
        context: InputResolutionContext | None = None,
    ) -> ResolvedAlgorithmPlan:
        """Resolve a request without loading code or opening runtime input."""
        return self._planner.plan(request, context)

    def explain(
        self,
        request: AlgorithmRequest | ExecutionRequest,
        context: InputResolutionContext | None = None,
    ) -> dict[str, object]:
        """Return a credential-free deterministic plan projection."""
        return self._planner.explain(request, context)

    def execute(
        self,
        request: AlgorithmRequest | ExecutionRequest,
        context: InputExecutionContext,
        *,
        artifacts: tuple[ArtifactDraft, ...] = (),
        resolution_context: InputResolutionContext | None = None,
        cancelled: bool = False,
    ) -> AlgorithmRunResult:
        """Plan and execute one bounded request."""
        plan = self.plan(request, resolution_context)
        return self.execute_plan(
            plan,
            context,
            artifacts=artifacts,
            cancelled=cancelled,
        )

    def execute_plan(
        self,
        plan: ResolvedAlgorithmPlan,
        context: InputExecutionContext,
        *,
        artifacts: tuple[ArtifactDraft, ...] = (),
        cancelled: bool = False,
    ) -> AlgorithmRunResult:
        """Execute one already validated plan through the normal lifecycle."""
        plan.validate_integrity()
        if plan.runtime.execution_profile is None:
            return self._coordinator.execute(
                plan,
                context,
                artifacts,
                cancelled=cancelled,
            )
        if plan.distribution_spec is None:
            raise AlgorithmConfigurationError(
                "formal execution profile requires a DistributionSpec"
            )
        resources = WorkerResources(
            num_cpus=plan.runtime.num_cpus,
            num_gpus=plan.runtime.num_gpus,
            custom=plan.runtime.custom_resources,
        )
        with self._runtime_manager.open(
            plan.runtime.execution_profile,
            resources_per_worker=resources,
            worker_count=plan.runtime.worker_count,
        ) as runtime_session:
            result = self._coordinator.execute(
                plan,
                context,
                artifacts,
                cancelled=cancelled,
            )
            receipt = result.execution_receipt
            if receipt is None:
                return result
            updated_receipt = replace(
                cast(ExecutionReceipt, receipt),
                cluster_resources=dict(runtime_session.cluster_resources),
                runtime_owned=runtime_session.runtime_owned,
                resource_preflight=runtime_session.resource_preflight,
            )
            return replace(result, execution_receipt=updated_receipt)


__all__ = ["AlgorithmDispatcher", "AlgorithmRunCoordinator"]
