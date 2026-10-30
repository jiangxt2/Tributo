"""Gateway for planning and executing native-engine writes."""

from __future__ import annotations

from tributo.data.contracts.handles import DaftDataFrameHandle, RayDataHandle
from tributo.data.writing.bindings import WriteBinding
from tributo.data.writing.contracts import (
    WriteBindingError,
    WriteCapabilityError,
    WriteDescriptor,
    WriteExecutionContext,
    WriteHandle,
    WriteReceipt,
    WriteRequest,
    _safe_exception_summary,
)
from tributo.data.writing.registry import (
    RegisteredWriteBinding,
    WriteBindingRegistry,
)
from tributo.data.writing.target_registry import WriteTargetRegistry
from tributo.data.writing.targets import LogicalWritePlan, WriteTargetProvider
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class WriteGateway:
    """Resolve a typed-handle binding and delegate the terminal write."""

    def __init__(
        self,
        registry: WriteBindingRegistry,
        target_registry: WriteTargetRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._target_registry = target_registry or WriteTargetRegistry()

    def plan(self, request: WriteRequest, input_handle: WriteHandle) -> WriteDescriptor:
        """Validate an input handle and return credential-free capabilities."""
        plan = self._target_plan(request)
        registered, binding = self._binding(request, input_handle)
        try:
            descriptor = binding.describe(plan, input_handle)
        except Exception as exc:
            raise WriteBindingError(
                f"Write binding failed during describe with {type(exc).__name__}: "
                f"{_safe_exception_summary(exc)}",
                source_error_type=type(exc).__name__,
            ) from None
        if not isinstance(descriptor, WriteDescriptor):
            raise WriteCapabilityError(
                "Write binding returned an invalid write descriptor"
            )
        if (
            descriptor != registered.descriptor
            or descriptor.engine_id != request.engine
            or descriptor.target_kind != request.target_kind
            or request.mode not in descriptor.capabilities.supported_modes
        ):
            raise WriteCapabilityError(
                "Write binding descriptor is inconsistent with the request"
            )
        return descriptor

    def execute(self, request: WriteRequest, input_handle: WriteHandle) -> WriteReceipt:
        """Run one terminal native write; the caller retains handle ownership."""
        plan = self._target_plan(request)
        registered, binding = self._binding(request, input_handle)
        context = WriteExecutionContext(
            request_digest=request.request_digest,
            runtime_options=request.runtime_options,
        )
        try:
            receipt = binding.execute(plan, input_handle, context)
        except Exception as exc:
            raise WriteBindingError(
                f"Write binding failed during execute with {type(exc).__name__}: "
                f"{_safe_exception_summary(exc)}",
                source_error_type=type(exc).__name__,
            ) from None
        if not isinstance(receipt, WriteReceipt):
            raise WriteCapabilityError(
                "Write binding returned an invalid write receipt"
            )
        if (
            receipt.request_digest != request.request_digest
            or receipt.engine_id != request.engine
            or receipt.binding_id != registered.descriptor.binding_id
            or receipt.target_kind != request.target_kind
            or receipt.target_ref != request.target
            or receipt.mode != request.mode
        ):
            raise WriteCapabilityError(
                "Write binding returned a receipt inconsistent with the request"
            )
        return receipt

    def _target_plan(self, request: WriteRequest) -> LogicalWritePlan:
        registered = self._target_registry.resolve(request)
        try:
            provider = registered.factory()
        except Exception as exc:
            raise WriteCapabilityError(
                f"Write target provider factory failed with {type(exc).__name__}"
            ) from None
        if not isinstance(provider, WriteTargetProvider):
            raise WriteCapabilityError(
                f"Write target provider factory returned {type(provider).__name__}, "
                "expected WriteTargetProvider"
            )
        try:
            plan = provider.plan(request)
        except Exception as exc:
            raise WriteCapabilityError(
                f"Write target provider failed with {type(exc).__name__}"
            ) from None
        if not isinstance(plan, LogicalWritePlan):
            raise WriteCapabilityError(
                "Write target provider returned an invalid logical write plan"
            )
        if (
            plan.request_digest != request.request_digest
            or plan.engine_id != request.engine
            or plan.target_kind != request.target_kind
            or plan.target != request.target
            or plan.mode != request.mode
            or dict(plan.options) != dict(request.options)
            or dict(plan.runtime_options)
            != dict(request.credential_free_runtime_options)
        ):
            raise WriteCapabilityError(
                "Write target provider returned a plan inconsistent with the request"
            )
        return plan

    def _binding(
        self, request: WriteRequest, input_handle: WriteHandle
    ) -> tuple[RegisteredWriteBinding, WriteBinding]:
        if request.engine == "tributo.ray_data" and not isinstance(
            input_handle, RayDataHandle
        ):
            raise WriteCapabilityError("Ray writes require a RayDataHandle")
        if request.engine == "tributo.daft" and not isinstance(
            input_handle, DaftDataFrameHandle
        ):
            raise WriteCapabilityError("Daft writes require a DaftDataFrameHandle")
        registered = self._registry.resolve(request)
        try:
            binding = registered.factory()
        except Exception as exc:
            raise WriteBindingError(
                f"Write binding factory failed with {type(exc).__name__}",
                source_error_type=type(exc).__name__,
            ) from None
        if not isinstance(binding, WriteBinding):
            raise WriteCapabilityError(
                f"Write binding factory returned {type(binding).__name__}, "
                "expected WriteBinding"
            )
        return registered, binding
