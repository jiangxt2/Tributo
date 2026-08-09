"""Side-effect-free portable algorithm request planner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tributo._common.immutable import deep_thaw
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmRequest,
    AlgorithmResolution,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    canonical_digest,
)
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
        request: AlgorithmRequest,
        context: InputResolutionContext | None = None,
    ) -> ResolvedAlgorithmPlan:
        """Build an immutable deterministic plan for one bounded request."""
        registration = self._registry.resolve(
            algorithm=request.algorithm,
            operation=request.operation,
            implementation_id=request.implementation_id,
        )
        config = dict(deep_thaw(registration.spec.default_config))
        config.update(deep_thaw(request.algorithm_config))
        unknown = sorted(
            set(config) - set(registration.implementation.allowed_config_keys)
        )
        if unknown:
            raise AlgorithmConfigurationError(
                f"configuration contains undeclared key(s): {unknown}"
            )
        resolver = self.resolver(request.input_binding.resolver_id)
        descriptor = resolver.describe(
            request.input_binding,
            context or InputResolutionContext(),
        )
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
        selected_adapter = str(registration.runtime.worker_input_adapter_ref)
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
            registration.runtime.topology is RuntimeTopology.DATA_PARALLEL
            and "shardable" not in descriptor.input_capabilities
        ):
            raise AlgorithmConfigurationError(
                "data_parallel input requires the explicit 'shardable' capability"
            )
        resolution = AlgorithmResolution(
            algorithm=registration.spec.name,
            algorithm_version=registration.spec.version,
            implementation_id=registration.implementation.implementation_id,
            implementation_version=registration.implementation.version,
            execution_mode=registration.implementation.execution_mode,
            environment_id=registration.environment.environment_id,
            runtime_id=registration.runtime.runtime_id,
        )
        provisional = ResolvedAlgorithmPlan(
            format_version=2,
            plan_id="0" * 64,
            operation=request.operation,
            resolution=resolution,
            implementation=registration.implementation,
            environment=registration.environment,
            runtime=registration.runtime,
            input_binding=request.input_binding,
            input_descriptor=descriptor,
            algorithm_config=config,
            config_digest=canonical_digest(config),
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
        )

    def explain(
        self,
        request: AlgorithmRequest,
        context: InputResolutionContext | None = None,
    ) -> dict[str, Any]:
        """Return the plan projection without opening input or loading code."""
        return self.plan(request, context).to_dict()


__all__ = ["AlgorithmPlanner"]
