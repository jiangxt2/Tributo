"""Worker-only execution of a trusted module-qualified user function."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
    UserExecutionContext,
)
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class RayFunctionExecutable:
    """Adapt one trusted user function to the bounded fit capability."""

    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        function: Callable[[UserExecutionContext], object],
        artifacts: tuple[ArtifactDraft, ...],
    ) -> None:
        self._plan = plan
        self._function = function
        self._artifacts = artifacts

    def fit(
        self,
        context: AlgorithmExecutionContext,
    ) -> AlgorithmExecutionResult:
        """Invoke user code with a least-authority reporting context."""
        user_context = UserExecutionContext(
            inputs=context.inputs,
            configuration=self._plan.algorithm_config,
            worker_metadata=context.worker_metadata,
            artifacts=self._artifacts,
            cancelled=context.cancelled,
        )
        try:
            returned = self._function(user_context)
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"custom Ray function failed: {type(exc).__name__}: {exc}"
            ) from exc
        if returned is not None:
            raise AlgorithmExecutionError(
                "custom Ray function must report through UserExecutionContext "
                "and return None"
            )
        return user_context.build_result()


@DeveloperAPI
def create_executable(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[ArtifactDraft, ...],
) -> RayFunctionExecutable:
    """Create a user-function executable from a Worker-loaded callable."""
    if (
        not inspect.isfunction(implementation)
        or implementation.__name__ == "<lambda>"
        or implementation.__closure__ is not None
    ):
        raise AlgorithmExecutionError(
            "custom Ray function reference must resolve to a module-level function"
        )
    return RayFunctionExecutable(
        plan=plan,
        function=implementation,
        artifacts=artifacts,
    )


__all__ = ["RayFunctionExecutable", "create_executable"]
