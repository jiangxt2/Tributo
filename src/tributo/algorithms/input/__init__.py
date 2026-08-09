"""Input implementations for portable algorithm execution."""

from tributo.algorithms.input.fake import FAKE_RESOLVER_ID as FAKE_RESOLVER_ID
from tributo.algorithms.input.fake import (
    FakeInputInvocation,
    FakeInputResolver,
    FakeInputRuntimeAdapter,
    FakeTabularPayload,
)
from tributo.algorithms.input.tabular import InMemoryTabularInputView

__all__ = [
    "FakeInputInvocation",
    "FakeInputResolver",
    "FakeInputRuntimeAdapter",
    "FakeTabularPayload",
    "InMemoryTabularInputView",
]
