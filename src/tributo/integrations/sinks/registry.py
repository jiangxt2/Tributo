"""Extensible ResultSink registry for formal Ray inference."""

from __future__ import annotations

import threading
from typing import Callable, ClassVar, cast

from tributo.inference.contracts import (
    BoundResultSink,
    ResultSink,
    ResultSinkReceipt,
    ResultSinkRequest,
)
from tributo.util.annotations import DeveloperAPI

ResultSinkFactory = Callable[[], ResultSink]


@DeveloperAPI
class BoundResultSinkAdapter:
    """Bind one validated output request before inference core execution."""

    api_version: ClassVar[int] = 1

    def __init__(self, sink: ResultSink, request: ResultSinkRequest) -> None:
        if sink.sink_id != request.sink_id:
            raise ValueError(
                f"ResultSink {sink.sink_id!r} cannot bind {request.sink_id!r}"
            )
        self._sink = sink
        self._request = request
        self._sink_id = str(sink.sink_id)

    @property
    def sink_id(self) -> str:
        return self._sink_id

    def write(
        self,
        dataset: object,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt:
        return self._sink.write(
            dataset,
            self._request,
            run_id=run_id,
            plan_digest=plan_digest,
        )


@DeveloperAPI
class ResultSinkRegistry:
    """Resolve a sink by the credential-free ``sink_id`` contract."""

    def __init__(self) -> None:
        self._factories: dict[str, ResultSinkFactory] = {}

    def register(self, sink_id: str, factory: ResultSinkFactory) -> None:
        """Register one sink factory without replacing an existing sink."""
        if not sink_id or not callable(factory):
            raise ValueError("result sink registrations require an id and factory")
        if sink_id in self._factories:
            raise ValueError(f"ResultSink {sink_id!r} is already registered")
        self._factories[sink_id] = factory

    def create(self, request: ResultSinkRequest) -> ResultSink:
        """Create the sink selected by one validated request."""
        try:
            factory = self._factories[request.sink_id]
        except KeyError:
            raise ValueError(
                f"No ResultSink is registered for {request.sink_id!r}"
            ) from None
        sink = factory()
        if sink.sink_id != request.sink_id:
            raise ValueError(
                f"ResultSink factory returned {sink.sink_id!r}; "
                f"expected {request.sink_id!r}"
            )
        return sink

    def bind(self, request: ResultSinkRequest) -> BoundResultSink:
        """Resolve and bind target configuration outside inference core."""
        return BoundResultSinkAdapter(self.create(request), request)


_REGISTRY_LOCK = threading.RLock()
_DEFAULT_REGISTRY: ResultSinkRegistry | None = None


@DeveloperAPI
def default_result_sink_registry() -> ResultSinkRegistry:
    """Return built-ins without importing optional data connectors.

    Connector-specific outputs use ``data-write-v1`` and are resolved by the
    data module's WriteGateway only when the selected target is executed.
    """
    global _DEFAULT_REGISTRY
    with _REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            registry = ResultSinkRegistry()
            registry.register(
                "parquet-v1",
                lambda: _load_sink(
                    "tributo.integrations.sinks.parquet", "ParquetResultSink"
                ),
            )
            registry.register(
                "lance-v1",
                lambda: _load_sink(
                    "tributo.integrations.sinks.lance", "LanceResultSink"
                ),
            )
            registry.register(
                "data-write-v1",
                lambda: _load_sink(
                    "tributo.integrations.sinks.data_write", "DataWriteResultSink"
                ),
            )
            _DEFAULT_REGISTRY = registry
        return _DEFAULT_REGISTRY


def _load_sink(module_name: str, class_name: str) -> ResultSink:
    module = __import__(module_name, fromlist=[class_name])
    sink_type = cast(type[ResultSink], getattr(module, class_name))
    return sink_type()


__all__ = [
    "BoundResultSinkAdapter",
    "ResultSinkRegistry",
    "default_result_sink_registry",
]
