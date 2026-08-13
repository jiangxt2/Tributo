"""Capability planning for explainability operations."""

from __future__ import annotations

from dataclasses import dataclass, field

from tributo.explainability.contracts import ExplainabilityRequest
from tributo.explainability.protocols import (
    ExplainableModelContext,
    ExplainerAdapter,
    SupportDecision,
)
from tributo.explainability.registry import (
    ExplainerRegistry,
    get_default_explainer_registry,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExplainabilityPlan:
    """Pinned adapter/backend selection for one request."""

    request: ExplainabilityRequest
    adapter_id: str
    adapter: type[ExplainerAdapter]
    decision: SupportDecision
    resource_requirements: dict[str, int | float] = field(default_factory=dict)


@PublicAPI(stability="alpha")
class ExplainabilityPlanner:
    """Select an adapter from verified context and explicit request policy."""

    def __init__(self, registry: ExplainerRegistry | None = None) -> None:
        self._registry = registry or get_default_explainer_registry()

    @staticmethod
    def preflight_limits(
        request: ExplainabilityRequest, *, input_rows: int
    ) -> dict[str, int | float]:
        """Reject a request whose known upper bound exceeds its byte budget."""
        limit = request.limits.max_explanation_bytes
        feature_count = min(
            len(request.feature_columns) or request.limits.max_features or 1,
            request.limits.max_features or len(request.feature_columns) or 1,
        )
        output_count = 1 if request.output_target in {"raw", "raw_margin"} else 2
        background_rows = (
            request.limits.max_background_rows
            or (request.reference.rows if request.reference is not None else None)
            or 1
        )
        estimated_rows = input_rows * feature_count * output_count
        estimated_bytes = estimated_rows * 512
        if limit is not None and estimated_bytes > limit:
            raise ValueError(
                "estimated explanation output exceeds "
                f"limits.max_explanation_bytes={limit}"
            )
        if (
            request.limits.max_explanation_rows is not None
            and estimated_rows > request.limits.max_explanation_rows
        ):
            raise ValueError(
                "estimated explanation output rows exceed "
                f"limits.max_explanation_rows={request.limits.max_explanation_rows}"
            )
        return {
            "estimated_output_rows": estimated_rows,
            "estimated_output_bytes": estimated_bytes,
            "estimated_background_rows": background_rows,
            "batch_size": request.resource_policy.batch_size,
            "concurrency": request.resource_policy.concurrency,
        }

    def plan(
        self,
        context: ExplainableModelContext,
        request: ExplainabilityRequest,
    ) -> ExplainabilityPlan:
        adapter_id = f"{request.explainer}-v1"
        adapter = self._registry.get(adapter_id)
        decision = adapter.supports(context, request)
        if not decision.supported:
            missing = (
                f"; missing dependencies={decision.required_dependencies}"
                if decision.required_dependencies
                else ""
            )
            raise ValueError(
                f"Explainability adapter {adapter_id!r} does not support this "
                f"request: {decision.reason}{missing}"
            )
        return ExplainabilityPlan(
            request=request,
            adapter_id=adapter_id,
            adapter=adapter,
            decision=decision,
            resource_requirements=self.preflight_limits(
                request,
                input_rows=0,
            ),
        )


__all__ = ["ExplainabilityPlan", "ExplainabilityPlanner"]
