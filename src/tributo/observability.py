"""Explicit opt-in lifecycle log reporting. No optional SDK imports on import."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

from tributo._common.reporting.config import (
    CollectorConfig,
    DorisProfile,
    ReportingConfig,
    ReportingError,
    SenderConfig,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class ReportingHandle(Protocol):
    """A reporting context's public status and Ray propagation interface."""

    enabled: bool
    state: str

    def runtime_env(
        self, runtime_env: Mapping[str, Any] | None = None
    ) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...


@PublicAPI(stability="alpha")
@contextmanager
def reporting_session(
    config: ReportingConfig | None = None, *, service_lifetime: bool = False
) -> Iterator[ReportingHandle]:
    """Report one execution domain, explicitly enabling through config or env.

    Disabled is the default and imports no optional reporting dependencies.
    A caller-owned Ray session uses the yielded session's ``runtime_env()``
    when it first initializes Ray. Existing external workers are not changed.
    ``service_lifetime=True`` explicitly owns newly deployed managed Serve
    applications and deletes only those applications when this context exits.
    """
    from tributo._common.reporting.session import session_scope

    with session_scope(config, service_lifetime=service_lifetime) as session:
        yield session


__all__ = [
    "CollectorConfig",
    "DorisProfile",
    "ReportingConfig",
    "ReportingError",
    "ReportingHandle",
    "SenderConfig",
    "reporting_session",
]
