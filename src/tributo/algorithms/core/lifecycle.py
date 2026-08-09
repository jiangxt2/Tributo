"""Bounded lifecycle dispatcher driven only by operation capability protocols."""

from __future__ import annotations

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmOperation,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    Evaluable,
    Fittable,
    Predictable,
    Transformable,
)
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class BatchLifecycle:
    """Invoke exactly one operation capability on a Worker executable."""

    def execute(
        self,
        operation: AlgorithmOperation,
        executable: object,
        context: AlgorithmExecutionContext,
    ) -> AlgorithmExecutionResult:
        """Execute the declared operation or reject a capability mismatch."""
        if operation is AlgorithmOperation.FIT and isinstance(executable, Fittable):
            result = executable.fit(context)
        elif operation is AlgorithmOperation.EVALUATE and isinstance(
            executable, Evaluable
        ):
            result = executable.evaluate(context)
        elif operation is AlgorithmOperation.PREDICT and isinstance(
            executable, Predictable
        ):
            result = executable.predict(context)
        elif operation is AlgorithmOperation.TRANSFORM and isinstance(
            executable, Transformable
        ):
            result = executable.transform(context)
        else:
            raise AlgorithmExecutionError(
                f"executable does not implement operation {operation.value!r}"
            )
        if not isinstance(result, AlgorithmExecutionResult):
            raise AlgorithmExecutionError(
                "executable returned an invalid AlgorithmExecutionResult"
            )
        return result


__all__ = ["BatchLifecycle"]
