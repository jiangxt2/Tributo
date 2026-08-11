"""Bounded-ingestion adapter owned by the inference domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from tributo.data import (
    IngestionDescriptor,
    IngestionGateway,
    IngestionOpenResult,
    IngestionPlanReceipt,
    IngestionRequest,
    IngestionRuntimeContext,
    RayDataHandle,
    ray_worker_distribution_probe,
)
from tributo.exceptions import DataSourceError, JobConfigurationError
from tributo.inference.contracts import ResolvedInputSelection
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data


class _IngestionGatewayLike(Protocol):
    def describe(self, request: IngestionRequest) -> IngestionDescriptor: ...

    def open(
        self,
        request: IngestionRequest,
        runtime_context: IngestionRuntimeContext | None = None,
    ) -> IngestionOpenResult: ...


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class OpenedInferenceInput:
    """Ray Dataset plus ingestion provenance and delegated lifecycle."""

    dataset: ray.data.Dataset
    receipt: IngestionPlanReceipt
    _result: IngestionOpenResult

    def close(self) -> None:
        """Release an owned input after successful terminal execution."""
        self._result.close()

    def cancel(self) -> None:
        """Cancel or release an owned input after failed terminal execution."""
        self._result.cancel()


@runtime_checkable
@PublicAPI(stability="alpha")
class InputResolverPort(Protocol):
    """Inference-owned boundary for describing and opening bounded input."""

    def describe(self, request: IngestionRequest) -> ResolvedInputSelection: ...

    def open(self, selection: ResolvedInputSelection) -> OpenedInferenceInput: ...


@PublicAPI(stability="alpha")
class IngestionGatewayInputResolver:
    """Pin and open explicit Ray ingestion through the public Data facade."""

    def __init__(
        self,
        gateway: _IngestionGatewayLike | None = None,
        *,
        runtime_context_factory: Callable[[], IngestionRuntimeContext] | None = None,
    ) -> None:
        self._gateway = gateway or IngestionGateway()
        self._runtime_context_factory = (
            runtime_context_factory or _default_runtime_context
        )

    def describe(self, request: IngestionRequest) -> ResolvedInputSelection:
        """Select one Binding and freeze a second, explicitly pinned descriptor."""
        _require_ray_request(request)
        selected = self._gateway.describe(request)
        _require_ray_descriptor(selected)

        pinned_request = request.model_copy(update={"binding_id": selected.binding_id})
        pinned = self._gateway.describe(pinned_request)
        _require_ray_descriptor(pinned)
        _compare_descriptor_routes(
            selected,
            pinned,
            context="binding selection changed while pinning the request",
        )
        return ResolvedInputSelection(request=pinned_request, descriptor=pinned)

    def open(self, selection: ResolvedInputSelection) -> OpenedInferenceInput:
        """Open and verify one Worker-local Ray handle without implicit conversion."""
        _require_ray_request(selection.request)
        execution_descriptor = self._gateway.describe(selection.request)
        _require_ray_descriptor(execution_descriptor)
        _compare_descriptor_routes(
            selection.descriptor,
            execution_descriptor,
            context="ingestion route differs between planning and execution",
            allow_tributo_source_overlay=True,
        )

        result = self._gateway.open(
            selection.request,
            self._runtime_context_factory(),
        )
        try:
            if not isinstance(result.handle, RayDataHandle):
                raise DataSourceError(
                    "Ray inference requires RayDataHandle; implicit engine "
                    "conversion is disabled"
                )
            _compare_descriptor_receipt(execution_descriptor, result.receipt)
        except Exception:
            result.cancel()
            raise
        return OpenedInferenceInput(
            dataset=result.handle.dataset,
            receipt=result.receipt,
            _result=result,
        )


def _default_runtime_context() -> IngestionRuntimeContext:
    """Request Worker version evidence when inference already runs under Ray."""
    try:
        import ray
    except ImportError:
        return IngestionRuntimeContext()
    if not ray.is_initialized():
        return IngestionRuntimeContext()
    return IngestionRuntimeContext(
        distribution_probe=ray_worker_distribution_probe,
        require_worker_validation=True,
    )


def _require_ray_request(request: IngestionRequest) -> None:
    if request.engine != "tributo.ray_data":
        raise JobConfigurationError(
            "RayMapBatchesExecutor requires ingestion engine 'ray'; automatic "
            "engine selection, fallback, and Daft-to-Ray conversion are disabled"
        )


def _require_ray_descriptor(descriptor: IngestionDescriptor) -> None:
    if (
        descriptor.engine_id != "tributo.ray_data"
        or descriptor.handle_kind != "ray_data"
    ):
        raise DataSourceError(
            "Ray inference planning produced a non-Ray ingestion descriptor"
        )


_ROUTE_FIELDS = (
    "engine_id",
    "provider_id",
    "connector_id",
    "binding_id",
    "scan_kind",
    "handle_kind",
    "schema_contract_fingerprint",
    "required_capabilities",
    "available_capabilities",
    "deferred_validations",
    "binding_distribution",
    "binding_distribution_version",
    "capability_version",
)


def _compare_descriptor_routes(
    expected: IngestionDescriptor,
    actual: IngestionDescriptor,
    *,
    context: str,
    allow_tributo_source_overlay: bool = False,
) -> None:
    fields: tuple[str, ...] = _ROUTE_FIELDS
    if (
        allow_tributo_source_overlay
        and expected.binding_distribution == "tributo"
        and actual.binding_distribution == "tributo"
    ):
        # Ray runtime_env.py_modules ships the current Tributo source but does
        # not replace the image's installed distribution metadata.  The
        # Worker-local describe/open pair still verifies one exact installed
        # version and records it in the receipt; comparing stale metadata to
        # the submitter would reject the source overlay itself.  Third-party
        # Binding versions remain cross-environment route invariants.
        fields = tuple(
            field for field in fields if field != "binding_distribution_version"
        )
    mismatches = tuple(
        field for field in fields if getattr(expected, field) != getattr(actual, field)
    )
    if mismatches:
        raise DataSourceError(f"{context}: mismatched fields {list(mismatches)}")


def _compare_descriptor_receipt(
    descriptor: IngestionDescriptor,
    receipt: IngestionPlanReceipt,
) -> None:
    pairs = {
        "request_digest": (descriptor.request_digest, receipt.request_digest),
        "source_ref": (descriptor.source_ref, receipt.source_ref),
        "dataset_ref": (descriptor.dataset_ref, receipt.dataset_ref),
        "logical_plan_digest": (
            descriptor.logical_plan_digest,
            receipt.logical_plan_digest,
        ),
        "engine_id": (descriptor.engine_id, receipt.engine_id),
        "provider_id": (descriptor.provider_id, receipt.provider_id),
        "connector_id": (descriptor.connector_id, receipt.connector_id),
        "binding_id": (descriptor.binding_id, receipt.binding_id),
        "scan_kind": (descriptor.scan_kind, receipt.scan_kind),
        "binding_distribution": (
            descriptor.binding_distribution,
            receipt.binding_distribution,
        ),
        "binding_distribution_version": (
            descriptor.binding_distribution_version,
            receipt.binding_distribution_version,
        ),
    }
    mismatches = tuple(
        field for field, (expected, actual) in pairs.items() if expected != actual
    )
    if mismatches:
        raise DataSourceError(
            "ingestion descriptor and open receipt differ for fields "
            f"{list(mismatches)}"
        )


__all__ = [
    "IngestionGatewayInputResolver",
    "InputResolverPort",
    "OpenedInferenceInput",
]
