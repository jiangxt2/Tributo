"""Transport-neutral broker consumer runner."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

from tributo.integrations.broker import (
    BrokerPlugin,
    BrokerRuntime,
    TaskDisposition,
    TaskOutcome,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class BrokerRunnerState(StrEnum):
    """Observable lifecycle state of a broker consumer process."""

    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


@PublicAPI(stability="beta")
class BrokerRunner:
    """Run provider-owned message handling with isolated broker failures."""

    def __init__(
        self,
        plugin: BrokerPlugin,
        config: Mapping[str, Any],
        *,
        poll_timeout_ms: int = 5000,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_timeout_ms < 0:
            raise ValueError("poll_timeout_ms must be non-negative")
        if backoff_initial <= 0 or backoff_max < backoff_initial:
            raise ValueError("invalid broker reconnect backoff")
        self.plugin = plugin
        self.config = dict(config)
        self.poll_timeout_ms = poll_timeout_ms
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self._sleep = sleep
        self._runtime: BrokerRuntime | None = None
        self._state = BrokerRunnerState.STOPPED
        self._stop_requested = False
        self._backoff = backoff_initial

    @property
    def state(self) -> BrokerRunnerState:
        """Return the current runner state."""
        return self._state

    @property
    def runtime(self) -> BrokerRuntime | None:
        """Expose the provider runtime for diagnostics and tests."""
        return self._runtime

    def start(self) -> None:
        """Create the provider runtime without assuming broker availability."""
        if self._runtime is not None and self._state != BrokerRunnerState.STOPPED:
            return
        self._state = BrokerRunnerState.STARTING
        self._stop_requested = False
        self._runtime = self.plugin.create_runtime(self.config)
        self._backoff = self.backoff_initial
        self._state = BrokerRunnerState.READY
        logger.info("Broker runner ready: broker=%s", self.plugin.broker_id)

    def request_stop(self) -> None:
        """Request graceful termination after the current poll/handle cycle."""
        self._stop_requested = True
        if self._state not in {BrokerRunnerState.STOPPED, BrokerRunnerState.STOPPING}:
            self._state = BrokerRunnerState.STOPPING

    def _handle_broker_failure(self, operation: str, exc: BaseException) -> None:
        self._state = BrokerRunnerState.DEGRADED
        logger.warning(
            "Broker unavailable during %s: broker=%s error=%s; retrying in %.1fs",
            operation,
            self.plugin.broker_id,
            type(exc).__name__,
            self._backoff,
            exc_info=True,
        )
        self._sleep(self._backoff)
        self._backoff = min(self.backoff_max, self._backoff * 2)
        self._state = BrokerRunnerState.RECONNECTING

    def _apply_outcome(self, message: Any, outcome: TaskOutcome) -> None:
        assert self._runtime is not None
        consumer = self._runtime.consumer
        try:
            if outcome.disposition == TaskDisposition.ACK:
                consumer.ack(message)
            elif outcome.disposition == TaskDisposition.RETRY:
                consumer.retry(message, outcome.error)
            elif outcome.disposition == TaskDisposition.REJECT:
                consumer.reject(message, outcome.error)
            else:  # pragma: no cover - StrEnum makes this defensive only.
                raise ValueError(f"Unknown task disposition: {outcome.disposition!r}")
        except Exception as exc:
            # A failed ACK must leave the message recoverable.  Do not invoke
            # another ACK from here: Redis/Kafka providers own their delivery
            # semantics and the next pending-recovery cycle will retry it.
            self._handle_broker_failure("acknowledgement", exc)
            return

        self._backoff = self.backoff_initial
        self._state = BrokerRunnerState.READY

    def run_once(self) -> bool:
        """Poll and process at most one message.

        Returns ``True`` when a message was received.  Connection failures are
        logged and delayed; they do not escape the runner boundary.
        """
        if self._stop_requested:
            return False
        if self._runtime is None:
            try:
                self.start()
            except Exception as exc:
                self._handle_broker_failure("startup", exc)
                return False
        assert self._runtime is not None
        try:
            recovered = self._runtime.consumer.recover_pending()
            if recovered:
                logger.info(
                    "Recovered pending broker messages: broker=%s count=%d",
                    self.plugin.broker_id,
                    recovered,
                )
            message = self._runtime.consumer.poll(self.poll_timeout_ms)
        except Exception as exc:
            self._handle_broker_failure("poll", exc)
            return False
        if message is None:
            self._state = BrokerRunnerState.READY
            return False

        try:
            outcome = self._runtime.handle(message)
            if not isinstance(outcome, TaskOutcome):
                raise TypeError("BrokerRuntime.handle must return TaskOutcome")
        except Exception as exc:
            # Provider exceptions are treated as temporary by default.  The
            # provider can return ACK for permanent validation failures after
            # best-effort FAILED reporting.
            logger.warning(
                "Broker task handling failed; retaining message for recovery: "
                "broker=%s job_id=%s error=%s",
                self.plugin.broker_id,
                getattr(message, "job_id", None),
                type(exc).__name__,
                exc_info=True,
            )
            outcome = TaskOutcome(
                disposition=TaskDisposition.RETRY,
                error=str(exc),
            )
        self._apply_outcome(message, outcome)
        return True

    def run(self) -> None:
        """Run until :meth:`request_stop` is called or interrupted."""
        try:
            while not self._stop_requested:
                self.run_once()
        except KeyboardInterrupt:
            logger.info("Broker runner interrupted")
            self.request_stop()
        finally:
            self.close()

    def close(self) -> None:
        """Close provider resources and mark the runner stopped."""
        if self._runtime is not None:
            self._state = BrokerRunnerState.STOPPING
            try:
                self._runtime.close()
            except Exception:
                logger.warning(
                    "Failed to close broker runtime: broker=%s",
                    self.plugin.broker_id,
                    exc_info=True,
                )
            finally:
                self._runtime = None
        self._state = BrokerRunnerState.STOPPED
