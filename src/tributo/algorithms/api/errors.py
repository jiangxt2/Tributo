"""Portable algorithm execution errors."""

from __future__ import annotations

from tributo.exceptions import (
    DataSourceError,
    JobConfigurationError,
    JobExecutionError,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class AlgorithmConfigurationError(JobConfigurationError):
    """An algorithm declaration or request is invalid."""


@PublicAPI(stability="alpha")
class AlgorithmResolutionError(JobConfigurationError):
    """A request could not be resolved to one implementation."""


@PublicAPI(stability="alpha")
class AlgorithmInputError(DataSourceError):
    """An algorithm input binding or runtime input is invalid."""


@PublicAPI(stability="alpha")
class AlgorithmDependencyError(JobExecutionError):
    """A Worker environment does not satisfy declared dependencies."""


@PublicAPI(stability="alpha")
class AlgorithmExecutionError(JobExecutionError):
    """A portable algorithm implementation failed during execution."""


__all__ = [
    "AlgorithmConfigurationError",
    "AlgorithmDependencyError",
    "AlgorithmExecutionError",
    "AlgorithmInputError",
    "AlgorithmResolutionError",
]
