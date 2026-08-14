"""Protocol for thin native-engine writing adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tributo.data.writing.contracts import (
    WriteDescriptor,
    WriteExecutionContext,
    WriteHandle,
    WriteReceipt,
)
from tributo.data.writing.targets import LogicalWritePlan
from tributo.util.annotations import DeveloperAPI


@runtime_checkable
@DeveloperAPI
class WriteBinding(Protocol):
    """Adapt one typed handle to one engine-native terminal write API."""

    def describe(
        self, plan: LogicalWritePlan, input_handle: WriteHandle
    ) -> WriteDescriptor:
        """Validate the target plan and input without executing a write."""

    def execute(
        self,
        plan: LogicalWritePlan,
        input_handle: WriteHandle,
        context: WriteExecutionContext,
    ) -> WriteReceipt:
        """Execute without closing the caller-owned handle.

        ``context`` is execution metadata and reference configuration, not a
        security boundary: the binding also receives the full credential-free
        request.  The binding must not close or otherwise take ownership of
        ``input_handle``.
        """
