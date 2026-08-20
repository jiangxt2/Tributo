"""Transport-neutral contracts for independently installed broker providers.

Core intentionally knows nothing about Redis, Kafka, RabbitMQ, or an external
operation protocol. Providers own transport semantics, request mapping, event
publication, and their production consume loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from tributo.util.annotations import PublicAPI

BROKER_API_VERSION = 1


@PublicAPI(stability="alpha")
class TaskDisposition(StrEnum):
    """Provider decision for one broker delivery."""

    ACK = "ack"
    RETRY = "retry"
    REJECT = "reject"


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class Message:
    """Opaque provider delivery passed across the Core Broker boundary.

    ``delivery_token`` identifies the transport delivery, not a business
    operation or Ray Job. Providers parse operation identity from ``payload``.
    Metadata is restricted to string keys and values so transport clients and
    credentials cannot be smuggled through this convenience surface.
    """

    payload: Any
    delivery_token: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.delivery_token, str) or not self.delivery_token.strip():
            raise ValueError("Message.delivery_token must not be empty")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise ValueError("Message.metadata requires string keys and values")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class BrokerError:
    """Minimal credential-safe provider error attached to a delivery outcome."""

    code: str
    sanitized_message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("BrokerError.code must not be empty")
        if not isinstance(self.sanitized_message, str):
            raise TypeError("BrokerError.sanitized_message must be a string")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TaskOutcome:
    """Transport-neutral disposition returned by a provider runtime."""

    disposition: TaskDisposition
    error: BrokerError | None = None


@PublicAPI(stability="alpha")
class TaskConsumer(ABC):
    """Consume opaque deliveries from a provider-owned transport."""

    @abstractmethod
    def poll(self, timeout_ms: int = 5000) -> Message | None:
        """Block until a delivery arrives or ``timeout_ms`` expires."""
        ...

    @abstractmethod
    def ack(self, message: Message) -> None:
        """Acknowledge a delivery according to provider semantics."""
        ...

    def retry(self, message: Message, error: BrokerError | None = None) -> None:
        """Apply provider-defined retry semantics.

        The default deliberately does nothing. Providers must override this
        hook unless leaving the delivery pending is their explicit retry
        policy.
        """
        del message, error

    def reject(self, message: Message, error: BrokerError | None = None) -> None:
        """Apply a provider-defined permanent rejection policy.

        The default deliberately does nothing. A provider must override this
        hook before returning ``REJECT`` for a delivery.
        """
        del message, error

    def recover_pending(self) -> int:
        """Best-effort provider hook; Core makes no recovery guarantee."""
        return 0

    def close(self) -> None:
        """Close transport resources; the default is a no-op."""
        return None


@PublicAPI(stability="alpha")
class BrokerRuntime(ABC):
    """Provider runtime for mapping one delivery into a delivery outcome."""

    @property
    @abstractmethod
    def consumer(self) -> TaskConsumer:
        """Return the provider-owned consumer."""
        ...

    @abstractmethod
    def handle(self, message: Message) -> TaskOutcome:
        """Handle one message without applying transport ACK side effects."""
        ...

    def close(self) -> None:
        """Close provider resources."""
        self.consumer.close()


@PublicAPI(stability="alpha")
class BrokerPlugin(ABC):
    """Structural base for an independently installed broker provider."""

    api_version: ClassVar[int] = BROKER_API_VERSION
    broker_id: ClassVar[str]
    capabilities: ClassVar[frozenset[str]] = frozenset()
    stability: ClassVar[str] = "alpha"

    @abstractmethod
    def validate_config(
        self, config: Mapping[str, Any], *, check_connectivity: bool = False
    ) -> None:
        """Validate provider config and optionally probe connectivity."""
        ...

    @abstractmethod
    def create_runtime(self, config: Mapping[str, Any]) -> BrokerRuntime:
        """Create a provider runtime; discovery itself must remain side-effect free."""
        ...


__all__ = [
    "BrokerError",
    "BrokerPlugin",
    "BrokerRuntime",
    "Message",
    "TaskConsumer",
    "TaskDisposition",
    "TaskOutcome",
]
