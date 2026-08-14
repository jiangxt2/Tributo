"""Errors raised by Tributo's Lance vector-index integration."""

from __future__ import annotations

from tributo.exceptions import JobConfigurationError, JobExecutionError


class VectorIndexConfigurationError(JobConfigurationError):
    """A vector-index request cannot be executed safely."""


class VectorIndexDependencyError(VectorIndexConfigurationError):
    """The selected Lance-Ray runtime contract is unavailable or incompatible."""


class VectorIndexExecutionError(JobExecutionError):
    """A Lance-Ray vector-index operation failed."""


class VectorResultDeliveryError(VectorIndexExecutionError):
    """A vector-search result could not be delivered under its contract."""


def safe_vector_error_diagnostic(exc: BaseException) -> tuple[str, tuple[str, ...]]:
    """Return a log-safe message and bounded exception-type chain.

    Only messages created by this module's domain errors are emitted. Arbitrary
    dependency exception messages are never copied into Ray Job logs.
    """
    if isinstance(exc, (VectorIndexConfigurationError, VectorIndexExecutionError)):
        message = str(exc)
    else:
        message = f"vector operation failed ({type(exc).__name__})"

    cause_types: list[str] = []
    current = exc.__cause__ or exc.__context__
    while current is not None and len(cause_types) < 4:
        cause_types.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return message, tuple(cause_types)
