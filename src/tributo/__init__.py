"""Tributo: Unified framework for submitting Ray Jobs.

This package provides a standardized interface for submitting and managing
Ray jobs with consistent configuration and error handling.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version
from typing import Final

_version_str: str
try:
    _version_str = _get_version("tributo")
except PackageNotFoundError:
    _version_str = "0.1.0"
__version__: Final[str] = _version_str

from tributo.config import JobConfig  # noqa: E402
from tributo.exceptions import (  # noqa: E402
    DataSourceError,
    JobConfigurationError,
    JobExecutionError,
    JobSubmissionError,
    JobTimeoutError,
    ModelExportError,
    TributoError,
)
from tributo.job import RayJob, TributoClient  # noqa: E402

__all__: list[str] = [
    # Core
    "TributoClient",
    "RayJob",
    "JobConfig",
    # Exceptions
    "TributoError",
    "JobSubmissionError",
    "JobExecutionError",
    "JobConfigurationError",
    "JobTimeoutError",
    "ModelExportError",
    "DataSourceError",
]
