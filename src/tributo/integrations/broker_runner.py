"""Generic lifecycle runner for Broker API v1 providers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

from tributo.integrations.broker import (
    BrokerPlugin,
    BrokerRuntime,
    Message,
    TaskDisposition,
    TaskOutcome,
)
from tributo.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)


@DeveloperAPI
class BrokerRunnerState(str, Enum):
    """Observable consumer process state."""

    NEW = "new"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@DeveloperAPI
class BrokerRunner:
    """Drive poll, maintenance, outcome application, recovery, and reconnect."""

    def __init__(
        self,
        plugin: BrokerPlugin,
        config: Mapping[str, Any],
        *,
        poll_timeout_ms: int = 5000,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        pending_recovery_interval: float = 30.0,
        max_recovery_batches: int = 10,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_timeout_ms < 0:
            raise ValueError("poll_timeout_ms must be non-negative")
        if backoff_initial < 0 or backoff_max < backoff_initial:
            raise ValueError("invalid reconnect backoff")
        if pending_recovery_interval <= 0:
            raise ValueError("pending_recovery_interval must be positive")
        if max_recovery_batches <= 0:
            raise ValueError("max_recovery_batches must be positive")
        self.plugin = plugin
        self.config = config
        self.poll_timeout_ms = poll_timeout_ms
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._pending_recovery_interval = pending_recovery_interval
        self._max_recovery_batches = max_recovery_batches
        self._sleep = sleep
        self._clock = clock
        self._next_backoff = backoff_initial
        self._next_pending_recovery = 0.0
        self.runtime: BrokerRuntime | None = None
        self.state = BrokerRunnerState.NEW

    def start(self) -> bool:
        """Create the provider runtime and recover abandoned deliveries."""
        if self.state is BrokerRunnerState.CLOSED:
            raise RuntimeError("BrokerRunner is closed")
        if self.runtime is not None:
            return True
        try:
            self.runtime = self.plugin.create_runtime(self.config)
            self._recover_pending_batches()
        except Exception:
            logger.warning(
                "Broker provider unavailable; reconnecting: broker=%s",
                getattr(self.plugin, "broker_id", "unknown"),
                exc_info=True,
            )
            self._enter_reconnecting()
            return False
        self.state = BrokerRunnerState.RUNNING
        self._schedule_pending_recovery()
        return True

    def run_once(self) -> bool:
        """Run one bounded maintenance and poll iteration."""
        if self.state is BrokerRunnerState.CLOSED:
            raise RuntimeError("BrokerRunner is closed")
        if self.runtime is None and not self.start():
            return False
        assert self.runtime is not None
        try:
            if self._clock() >= self._next_pending_recovery:
                self._recover_pending_batches()
                self._schedule_pending_recovery()
            self.runtime.maintain()
            message = self.runtime.consumer.poll(self.poll_timeout_ms)
            if message is None:
                self._next_backoff = self._backoff_initial
                return False
            outcome = self.runtime.handle(message)
            self._apply_outcome(message, outcome)
            return True
        except Exception:
            logger.warning(
                "Broker processing failed; reconnecting: broker=%s",
                getattr(self.plugin, "broker_id", "unknown"),
                exc_info=True,
            )
            self._enter_reconnecting()
            return False

    def run_forever(self) -> None:
        """Run until interrupted or :meth:`close` is called."""
        while self.state is not BrokerRunnerState.CLOSED:
            self.run_once()

    def _apply_outcome(self, message: Message, outcome: TaskOutcome) -> None:
        if self.runtime is None:
            raise RuntimeError("BrokerRunner has not started")
        if outcome.disposition is TaskDisposition.ACK:
            self.runtime.consumer.ack(message)
        elif outcome.disposition is TaskDisposition.RETRY:
            self.runtime.consumer.retry(message, outcome.error)
        elif outcome.disposition is TaskDisposition.REJECT:
            self.runtime.consumer.reject(message, outcome.error)
        else:
            raise ValueError(f"Unsupported task disposition: {outcome.disposition}")
        self._next_backoff = self._backoff_initial

    def _recover_pending_batches(self) -> int:
        if self.runtime is None:
            raise RuntimeError("BrokerRunner has not started")
        total = 0
        for _ in range(self._max_recovery_batches):
            recovered = self.runtime.consumer.recover_pending()
            if not isinstance(recovered, int) or recovered <= 0:
                break
            total += recovered
        return total

    def _schedule_pending_recovery(self) -> None:
        self._next_pending_recovery = self._clock() + self._pending_recovery_interval

    def _enter_reconnecting(self) -> None:
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                logger.warning("Broker runtime close failed", exc_info=True)
        self.state = BrokerRunnerState.RECONNECTING
        delay = self._next_backoff
        self._next_backoff = min(
            max(self._next_backoff * 2, self._backoff_initial), self._backoff_max
        )
        if delay:
            self._sleep(delay)

    def close(self) -> None:
        """Stop the runner and release provider resources idempotently."""
        if self.state is BrokerRunnerState.CLOSED:
            return
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                logger.warning("Broker runtime close failed", exc_info=True)
        self.state = BrokerRunnerState.CLOSED


__all__ = ["BrokerRunner", "BrokerRunnerState"]
