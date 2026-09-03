"""Side-effect-free portable algorithm request planner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from tributo._common.immutable import deep_thaw
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmRequest,
    AlgorithmResolution,
    DistributionStrategy,
    ExecutionRequest,
    InputBinding,
    InputBindingSet,
    InputDistribution,
    ResolvedAlgorithmPlan,
    ResolvedInputDescriptor,
    ResolvedInputDescriptorSet,
    RuntimeBinding,
    RuntimeTopology,
    TorchPolicy,
    WorkerResources,
    canonical_digest,
)
from tributo.algorithms.api.models import (
    FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS,
    AlgorithmRegistration,
)
from tributo.algorithms.core.contracts import validate_contract_value
from tributo.algorithms.core.registry import AlgorithmRegistrationRegistry
from tributo.algorithms.spi import InputResolutionContext, InputResolverPort
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class AlgorithmPlanner:
    """Resolve declarations and input metadata without loading executable code."""

    def __init__(
        self,
        registry: AlgorithmRegistrationRegistry,
        resolvers: Mapping[str, InputResolverPort],
    ) -> None:
        self._registry = registry
        self._resolvers = dict(resolvers)

    def resolver(self, resolver_id: str) -> InputResolverPort:
        """Return an injected resolver or fail before any input access."""
        try:
            return self._resolvers[resolver_id]
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                f"unknown input resolver: {resolver_id!r}"
            ) from exc

    def plan(
        self,
        request: AlgorithmRequest | ExecutionRequest,
        context: InputResolutionContext | None = None,
        *,
        available_resources: Mapping[str, float] | None = None,
    ) -> ResolvedAlgorithmPlan:
        """Build an immutable deterministic plan for one bounded request."""
        algorithm_request = (
            request.algorithm_request
            if isinstance(request, ExecutionRequest)
            else request
        )
        registration = self._registry.resolve(
            algorithm=algorithm_request.algorithm,
            operation=algorithm_request.operation,
            implementation_id=algorithm_request.implementation_id,
        )
        config = dict(deep_thaw(registration.spec.default_config))
        config.update(deep_thaw(algorithm_request.algorithm_config))
        unknown = sorted(
            set(config) - set(registration.implementation.allowed_config_keys)
        )
        if unknown:
            raise AlgorithmConfigurationError(
                f"configuration contains undeclared key(s): {unknown}"
            )
        if registration.contract_bindings is not None:
            config = validate_contract_value(
                registration.contract_bindings.config,
                config,
            )
            normalized_unknown = sorted(
                set(config) - set(registration.implementation.allowed_config_keys)
            )
            if normalized_unknown:
                raise AlgorithmConfigurationError(
                    "validated configuration contains undeclared key(s): "
                    f"{normalized_unknown}"
                )
        runtime = self._resolve_runtime(
            registration,
            request,
            available_resources=available_resources,
        )
        binding_set = algorithm_request.input_bindings
        if (
            registration.distribution_spec is not None
            and registration.distribution_spec.strategy
            is DistributionStrategy.RAY_TRAIN_TORCH
        ):
            policy = cast(TorchPolicy, registration.distribution_spec.policy)
            routes = {route.role: route for route in policy.dataset_routing}
            unknown_roles = sorted(
                {binding.name for binding in binding_set.bindings} - set(routes)
            )
            if unknown_roles:
                raise AlgorithmConfigurationError(
                    f"Torch input binding role(s) are not declared by TorchPolicy: {unknown_roles}"
                )
        resolution_context = context or InputResolutionContext()
        descriptors: list[ResolvedInputDescriptor] = []
        for binding in binding_set.bindings:
            resolver = self.resolver(binding.resolver_id)
            descriptor = resolver.describe(binding, resolution_context)
            self._validate_input_descriptor(
                registration,
                runtime,
                descriptor,
            )
            descriptors.append(descriptor)
        descriptor_set = ResolvedInputDescriptorSet(
            roles=tuple(binding.name for binding in binding_set.bindings),
            descriptors=tuple(descriptors),
        )
        selected_input_binding: InputBinding | InputBindingSet = (
            algorithm_request.input_binding
        )
        selected_input_descriptor: (
            ResolvedInputDescriptor | ResolvedInputDescriptorSet
        ) = (
            descriptor_set.descriptors[0]
            if isinstance(selected_input_binding, InputBinding)
            else descriptor_set
        )
        if registration.contract_bindings is not None:
            validate_contract_value(
                registration.contract_bindings.input,
                {
                    "primary_role": binding_set.primary_role,
                    "bindings": binding_set.descriptor_payload()["bindings"],
                    "descriptors": descriptor_set.to_dict(),
                },
            )
        resolution = AlgorithmResolution(
            algorithm=registration.spec.name,
            algorithm_version=registration.spec.version,
            implementation_id=registration.implementation.implementation_id,
            implementation_version=registration.implementation.version,
            execution_mode=registration.implementation.execution_mode,
            environment_id=registration.environment.environment_id,
            runtime_id=runtime.runtime_id,
            requested_algorithm=algorithm_request.algorithm,
        )
        provisional = ResolvedAlgorithmPlan(
            format_version=3
            if isinstance(selected_input_binding, InputBindingSet)
            else 2,
            plan_id="0" * 64,
            operation=algorithm_request.operation,
            resolution=resolution,
            implementation=registration.implementation,
            environment=registration.environment,
            runtime=runtime,
            input_binding=selected_input_binding,
            input_descriptor=selected_input_descriptor,
            algorithm_config=config,
            config_digest=canonical_digest(config),
            distribution_spec=registration.distribution_spec,
            contract_bindings=registration.contract_bindings,
        )
        plan_id = canonical_digest(provisional.to_dict(include_plan_id=False))
        return ResolvedAlgorithmPlan(
            format_version=provisional.format_version,
            plan_id=plan_id,
            operation=provisional.operation,
            resolution=provisional.resolution,
            implementation=provisional.implementation,
            environment=provisional.environment,
            runtime=provisional.runtime,
            input_binding=provisional.input_binding,
            input_descriptor=provisional.input_descriptor,
            algorithm_config=provisional.algorithm_config,
            config_digest=provisional.config_digest,
            distribution_spec=provisional.distribution_spec,
            contract_bindings=provisional.contract_bindings,
        )

    @staticmethod
    def _validate_input_descriptor(
        registration: AlgorithmRegistration,
        runtime: RuntimeBinding,
        descriptor: ResolvedInputDescriptor,
    ) -> None:
        """Validate one role without interpreting its domain semantics."""
        compatibility = registration.implementation.input_compatibility
        if descriptor.view_kind not in compatibility.accepted_input_views:
            raise AlgorithmConfigurationError(
                "input view is not accepted by the implementation Backend: "
                f"{descriptor.view_kind!r}"
            )
        if descriptor.engine_id not in compatibility.accepted_ingestion_engines:
            raise AlgorithmConfigurationError(
                "ingestion engine is not accepted by the implementation Backend: "
                f"{descriptor.engine_id!r}"
            )
        missing_capabilities = sorted(
            set(compatibility.required_input_capabilities)
            - set(descriptor.input_capabilities)
        )
        if missing_capabilities:
            raise AlgorithmConfigurationError(
                "input does not offer implementation-required capabilities: "
                f"{missing_capabilities}"
            )
        selected_adapter = str(runtime.worker_input_adapter_ref)
        supported_adapters = {
            str(reference) for reference in compatibility.supported_explicit_adapters
        }
        if selected_adapter not in supported_adapters:
            raise AlgorithmConfigurationError(
                "Worker input adapter is not declared by the implementation Backend: "
                f"{selected_adapter}"
            )
        if selected_adapter not in descriptor.compatible_worker_input_adapter_refs:
            raise AlgorithmConfigurationError(
                "input resolver and Worker input adapter are incompatible: "
                f"{selected_adapter}"
            )
        if (
            runtime.topology
            in {RuntimeTopology.DATA_PARALLEL, RuntimeTopology.RAY_MAP_REDUCE}
            and "shardable" not in descriptor.input_capabilities
        ):
            raise AlgorithmConfigurationError(
                "distributed input requires the explicit 'shardable' capability"
            )
        if (
            registration.distribution_spec is not None
            and registration.distribution_spec.input_distribution
            in {InputDistribution.SHARDED, InputDistribution.ROLE_ROUTED}
            and "shardable" not in descriptor.input_capabilities
        ):
            raise AlgorithmConfigurationError(
                "the declared distribution strategy requires shardable input"
            )

    @staticmethod
    def _resolve_runtime(
        registration: AlgorithmRegistration,
        request: AlgorithmRequest | ExecutionRequest,
        *,
        available_resources: Mapping[str, float] | None,
    ) -> RuntimeBinding:
        """Resolve a static declaration into one invocation-scoped binding."""
        distribution = registration.distribution_spec
        if distribution is None:
            if isinstance(request, ExecutionRequest):
                raise AlgorithmConfigurationError(
                    "legacy compatibility registrations do not accept "
                    "ExecutionRequest and cannot claim distributed training"
                )
            if registration.runtime is None:
                raise AlgorithmConfigurationError(
                    "legacy registration is missing its compatibility runtime"
                )
            return registration.runtime
        if not isinstance(request, ExecutionRequest):
            raise AlgorithmConfigurationError(
                "formal distributed registrations require an ExecutionRequest"
            )
        if not distribution.supports(request.profile, request.worker_count):
            raise AlgorithmConfigurationError(
                f"{distribution.strategy.value} does not support profile "
                f"{request.profile.value!r} with {request.worker_count} worker(s)"
            )
        declared = distribution.resources_per_worker
        resolved = request.resources_per_worker or declared
        AlgorithmPlanner._validate_resource_request(declared, resolved)
        if available_resources is not None:
            AlgorithmPlanner._validate_available_resources(
                resolved,
                request.worker_count,
                available_resources,
            )
        implementation = registration.implementation
        if (
            implementation.runtime_id is None
            or implementation.worker_input_adapter_ref is None
        ):
            raise AlgorithmConfigurationError(
                "formal implementation is missing runtime adapter declarations"
            )
        topology = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[distribution.strategy].topology
        policy_retries = getattr(distribution.policy, "max_retries", 0)
        max_retries = policy_retries if isinstance(policy_retries, int) else 0
        return RuntimeBinding(
            runtime_id=implementation.runtime_id,
            worker_input_adapter_ref=implementation.worker_input_adapter_ref,
            topology=topology,
            worker_count=request.worker_count,
            framework_parallelism=1,
            num_cpus=resolved.num_cpus,
            num_gpus=resolved.num_gpus,
            memory_bytes=resolved.memory_bytes,
            custom_resources=resolved.custom,
            max_retries=max_retries,
            execution_profile=request.profile,
            strategy=distribution.strategy,
            distribution_digest=distribution.digest,
            resume_from=request.resume_from,
            torch_recovery=(
                request.torch_recovery.to_dict()
                if request.torch_recovery is not None
                else None
            ),
        )

    @staticmethod
    def _validate_resource_request(
        declared: WorkerResources,
        requested: WorkerResources,
    ) -> None:
        if requested.num_cpus < declared.num_cpus:
            raise AlgorithmConfigurationError(
                "requested CPU per worker is below the DistributionSpec requirement"
            )
        if requested.num_gpus != declared.num_gpus:
            raise AlgorithmConfigurationError(
                "requested GPU per worker must exactly match the tested "
                "DistributionSpec capability; CPU fallback is forbidden"
            )
        if declared.memory_bytes is not None and (
            requested.memory_bytes is None
            or requested.memory_bytes < declared.memory_bytes
        ):
            raise AlgorithmConfigurationError(
                "requested memory per worker is below the DistributionSpec requirement"
            )
        for name, minimum in declared.custom.items():
            if requested.custom.get(name, 0.0) < minimum:
                raise AlgorithmConfigurationError(
                    f"requested custom resource {name!r} is below the declaration"
                )

    @staticmethod
    def _validate_available_resources(
        resources_per_worker: WorkerResources,
        worker_count: int,
        available_resources: Mapping[str, float],
    ) -> None:
        total = resources_per_worker.scaled(worker_count)
        required = {
            "CPU": total.num_cpus,
            "GPU": total.num_gpus,
            **dict(total.custom),
        }
        if total.memory_bytes is not None:
            required["memory"] = total.memory_bytes
        missing = {
            name: (amount, float(available_resources.get(name, 0.0)))
            for name, amount in required.items()
            if amount > float(available_resources.get(name, 0.0))
        }
        if missing:
            details = ", ".join(
                f"{name} required={required_amount:g} available={available:g}"
                for name, (required_amount, available) in sorted(missing.items())
            )
            raise AlgorithmConfigurationError(
                f"insufficient Ray cluster resources: {details}"
            )

    def explain(
        self,
        request: AlgorithmRequest | ExecutionRequest,
        context: InputResolutionContext | None = None,
        *,
        available_resources: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return the plan projection without opening input or loading code."""
        return self.plan(
            request,
            context,
            available_resources=available_resources,
        ).to_dict()


__all__ = ["AlgorithmPlanner"]
