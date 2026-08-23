"""Public alpha contracts for native-engine data writing."""

from tributo.data.contracts.modes import WriteMode
from tributo.data.writing.bindings import WriteBinding
from tributo.data.writing.builtins import default_write_gateway
from tributo.data.writing.capabilities import WriteCapability
from tributo.data.writing.contracts import (
    DataWriteTargetRequest,
    WriteBindingError,
    WriteCapabilityError,
    WriteDescriptor,
    WriteError,
    WriteExecutionContext,
    WriteHandle,
    WriteReceipt,
    WriteRequest,
)
from tributo.data.writing.gateway import WriteGateway
from tributo.data.writing.registry import WriteBindingRegistry
from tributo.data.writing.target_registry import WriteTargetRegistry
from tributo.data.writing.targets import (
    GenericWriteTargetProvider,
    LogicalWritePlan,
    WriteTargetProvider,
)

__all__ = [
    "WriteBinding",
    "WriteBindingError",
    "DataWriteTargetRequest",
    "WriteCapability",
    "WriteBindingRegistry",
    "WriteCapabilityError",
    "WriteDescriptor",
    "WriteError",
    "WriteExecutionContext",
    "WriteGateway",
    "default_write_gateway",
    "WriteHandle",
    "WriteMode",
    "WriteReceipt",
    "WriteRequest",
    "GenericWriteTargetProvider",
    "LogicalWritePlan",
    "WriteTargetProvider",
    "WriteTargetRegistry",
]
