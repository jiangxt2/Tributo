"""Framework-free Worker bootstrap with delayed implementation loading."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmDependencyError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    AlgorithmResolutionError,
    FailureCategory,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    WorkerExecutionResult,
)
from tributo.algorithms.core.lifecycle import BatchLifecycle
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    ExecutionEnvelope,
    PreparedInput,
)
from tributo.exceptions import TributoError
from tributo.util.annotations import DeveloperAPI

ExecutableFactory = Callable[..., object]
WorkerInputFactory = Callable[[object], PreparedInput]

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|token|api[_-]?key|client[_-]?secret|"
    r"secret[_-]?(?:access[_-]?)?key|authorization)\b"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URI_USERINFO_RE = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@")


def _sanitize_error_message(message: str) -> str:
    cleaned = message.replace("\r", " ").replace("\n", " ")
    cleaned = _URI_USERINFO_RE.sub(r"\1<redacted>@", cleaned)
    return _SENSITIVE_ASSIGNMENT_RE.sub(r"\1=<redacted>", cleaned)[:1000]


def _load_reference(reference: QualifiedReference) -> object:
    """Load one qualified reference. This function is called only on a Worker."""
    try:
        value: object = importlib.import_module(reference.module)
        for segment in reference.qualname.split("."):
            value = getattr(value, segment)
        return value
    except Exception as exc:
        raise AlgorithmDependencyError(
            f"cannot load Worker reference {reference}"
        ) from exc


def _actual_environment_versions(
    python_constraint: str,
    dependencies: tuple[str, ...],
) -> dict[str, str]:
    """Validate Worker dependencies and return their actual versions."""
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    if python_version not in SpecifierSet(python_constraint):
        raise AlgorithmDependencyError(
            f"Worker Python {python_version} does not satisfy {python_constraint}"
        )
    versions = {"python": python_version}
    for raw_requirement in dependencies:
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            actual = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AlgorithmDependencyError(
                f"Worker dependency is not installed: {requirement.name}"
            ) from exc
        if requirement.specifier and actual not in requirement.specifier:
            raise AlgorithmDependencyError(
                f"Worker dependency {requirement.name}=={actual} does not satisfy "
                f"{requirement.specifier}"
            )
        versions[requirement.name] = actual
    return versions


def _validate_module_digest(
    reference: QualifiedReference,
    expected_digest: str | None,
) -> None:
    """Validate an optional source-file digest before importing user code."""
    if expected_digest is None:
        return
    try:
        spec = importlib.util.find_spec(reference.module)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exc:
        raise AlgorithmDependencyError(
            f"cannot locate module for code digest: {reference.module!r}"
        ) from exc
    module_file = spec.origin if spec is not None else None
    if not isinstance(module_file, str) or module_file in {"built-in", "frozen"}:
        raise AlgorithmDependencyError(
            f"cannot verify code digest for module {reference.module!r}"
        )
    try:
        actual_digest = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
    except OSError as exc:
        raise AlgorithmDependencyError(
            f"cannot read module for code digest: {reference.module!r}"
        ) from exc
    if actual_digest != expected_digest:
        raise AlgorithmDependencyError(
            f"code digest mismatch for module {reference.module!r}"
        )


def _failure_category(exc: BaseException) -> FailureCategory:
    if isinstance(exc, AlgorithmDependencyError):
        return FailureCategory.DEPENDENCY
    if isinstance(exc, AlgorithmInputError):
        return FailureCategory.INPUT
    if isinstance(
        exc,
        (
            AlgorithmConfigurationError,
            AlgorithmResolutionError,
        ),
    ):
        return FailureCategory.VALIDATION
    if isinstance(exc, (AlgorithmExecutionError, TributoError)):
        return FailureCategory.EXECUTION
    return FailureCategory.INTERNAL


def _failed_result(exc: BaseException) -> AlgorithmExecutionResult:
    message = _sanitize_error_message(str(exc))
    return AlgorithmExecutionResult(
        status="failed",
        failure_category=_failure_category(exc),
        error_type=type(exc).__name__,
        error_message=message or "Worker execution failed",
    )


@DeveloperAPI
def worker_bootstrap(
    envelope: ExecutionEnvelope,
    worker_metadata: Mapping[str, Any] | None = None,
) -> WorkerExecutionResult:
    """Validate, load, execute, and clean up entirely inside a Worker process."""
    actual_versions: dict[str, str] = {}
    prepared: PreparedInput | None = None
    execution: AlgorithmExecutionResult | None = None
    try:
        envelope.plan.validate_integrity()
        actual_versions = _actual_environment_versions(
            envelope.plan.environment.python,
            envelope.plan.environment.dependencies,
        )
        _validate_module_digest(
            envelope.plan.implementation.implementation_ref,
            envelope.plan.implementation.code_digest,
        )
        implementation = _load_reference(
            envelope.plan.implementation.implementation_ref
        )
        executable_factory = _load_reference(
            envelope.plan.implementation.executable_factory_ref
        )
        if not callable(executable_factory):
            raise AlgorithmDependencyError(
                "resolved executable factory is not callable"
            )
        input_factory = _load_reference(envelope.plan.runtime.worker_input_adapter_ref)
        if not callable(input_factory):
            raise AlgorithmDependencyError(
                "resolved Worker input adapter is not callable"
            )
        prepared_value = input_factory(envelope.input_payload)
        if not isinstance(prepared_value, PreparedInput):
            raise AlgorithmInputError(
                "Worker input adapter did not return PreparedInput"
            )
        prepared = prepared_value
        executable = executable_factory(
            plan=envelope.plan,
            implementation=implementation,
            artifacts=envelope.artifacts,
        )
        context = AlgorithmExecutionContext(
            inputs=prepared.views,
            artifacts=envelope.artifacts,
            worker_metadata=dict(worker_metadata or {}),
            cancelled=envelope.cancelled,
        )
        execution = BatchLifecycle().execute(
            envelope.plan.operation,
            executable,
            context,
        )
    except Exception as exc:
        execution = _failed_result(exc)
    finally:
        if prepared is not None:
            try:
                prepared.close()
            except Exception as cleanup_exc:
                if execution is None or execution.status == "succeeded":
                    execution = _failed_result(cleanup_exc)
    if execution is None:
        execution = _failed_result(
            AlgorithmExecutionError("Worker produced no execution result")
        )
    return WorkerExecutionResult(
        execution=execution,
        actual_versions=actual_versions,
        worker_metadata=dict(worker_metadata or {}),
    )


@DeveloperAPI
def reduce_worker_group(
    plan: ResolvedAlgorithmPlan,
    results: tuple[WorkerExecutionResult, ...],
    reducer_worker_metadata: Mapping[str, Any] | None = None,
) -> WorkerExecutionResult:
    """Reduce rank-ordered results without loading reducer code on the Driver."""
    ordered: list[WorkerExecutionResult] = []
    aggregate_metadata: dict[str, Any] = {}
    try:
        if len(results) != plan.runtime.worker_count:
            raise AlgorithmExecutionError(
                "data-parallel Runtime returned an incomplete Worker group"
            )
        ranks: dict[int, WorkerExecutionResult] = {}
        worker_ids: set[str] = set()
        for result in results:
            rank = result.worker_metadata.get("world_rank")
            world_size = result.worker_metadata.get("world_size")
            worker_id = result.worker_metadata.get("worker_id")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or world_size != plan.runtime.worker_count
                or rank in ranks
                or not isinstance(worker_id, str)
                or not worker_id
            ):
                raise AlgorithmExecutionError(
                    "data-parallel Worker identity or rank metadata is invalid"
                )
            ranks[rank] = result
            worker_ids.add(worker_id)
        if set(ranks) != set(range(plan.runtime.worker_count)):
            raise AlgorithmExecutionError("data-parallel Worker ranks are incomplete")
        if len(worker_ids) != plan.runtime.worker_count:
            raise AlgorithmExecutionError(
                "data-parallel execution did not use distinct Ray Workers"
            )
        ordered = [ranks[rank] for rank in range(plan.runtime.worker_count)]
        aggregate_metadata = {
            "topology": "data_parallel",
            "worker_count": plan.runtime.worker_count,
            "workers": [dict(result.worker_metadata) for result in ordered],
            "reducer_worker": dict(reducer_worker_metadata or {}),
        }
        failed = [result for result in ordered if result.execution.status == "failed"]
        if failed:
            return WorkerExecutionResult(
                execution=failed[0].execution,
                actual_versions=failed[0].actual_versions,
                worker_metadata=aggregate_metadata,
            )
        expected_versions = dict(ordered[0].actual_versions)
        if any(dict(result.actual_versions) != expected_versions for result in ordered):
            raise AlgorithmDependencyError(
                "data-parallel Workers loaded inconsistent dependency versions"
            )
        reducer_ref = plan.runtime.result_reducer_ref
        if reducer_ref is None:
            raise AlgorithmConfigurationError(
                "data-parallel execution requires a result reducer"
            )
        _validate_module_digest(reducer_ref, plan.implementation.code_digest)
        reducer = _load_reference(reducer_ref)
        if not callable(reducer):
            raise AlgorithmDependencyError(
                "resolved data-parallel result reducer is not callable"
            )
        reduced = reducer(plan, tuple(ordered))
        if not isinstance(reduced, AlgorithmExecutionResult):
            raise AlgorithmExecutionError(
                "data-parallel result reducer returned an invalid result"
            )
        return WorkerExecutionResult(
            execution=reduced,
            actual_versions=expected_versions,
            worker_metadata=aggregate_metadata,
        )
    except Exception as exc:
        versions = dict(ordered[0].actual_versions) if ordered else {}
        return WorkerExecutionResult(
            execution=_failed_result(exc),
            actual_versions=versions,
            worker_metadata=aggregate_metadata,
        )


__all__ = ["reduce_worker_group", "worker_bootstrap"]
