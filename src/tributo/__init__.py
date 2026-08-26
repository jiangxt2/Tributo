"""Ray-native machine learning control-plane APIs.

The root package exposes the stable Ray Jobs client, job configuration, and
shared exception hierarchy. Component APIs live in packages such as
``tributo.data``, ``tributo.algorithms``, ``tributo.exporting``,
``tributo.inference``, and ``tributo.vector_index``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version
from typing import TYPE_CHECKING, Any, Final

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
from tributo.runtime import (
    RuntimeExecutionMode,
    RuntimeLifecycle,
    RuntimeSubmissionMode,
    RuntimeTarget,
)

if TYPE_CHECKING:
    from tributo.job import RayJob, TributoClient
    from tributo.runtime_providers import (
        LocalRayJobsProvider,
        RuntimeLease,
        RuntimeProvider,
    )

_LAZY_EXPORTS = {
    "RayJob": ("tributo.job", "RayJob"),
    "TributoClient": ("tributo.job", "TributoClient"),
    "LocalRayJobsProvider": ("tributo.runtime_providers", "LocalRayJobsProvider"),
    "RuntimeLease": ("tributo.runtime_providers", "RuntimeLease"),
    "RuntimeProvider": ("tributo.runtime_providers", "RuntimeProvider"),
    "open_job_submission_client": (
        "tributo.runtime_providers",
        "open_job_submission_client",
    ),
    "open_ray_client": ("tributo.runtime_providers", "open_ray_client"),
    "register_runtime_provider": (
        "tributo.runtime_providers",
        "register_runtime_provider",
    ),
    "resolve_runtime_provider": (
        "tributo.runtime_providers",
        "resolve_runtime_provider",
    ),
    "run_local_entrypoint": ("tributo.runtime_providers", "run_local_entrypoint"),
}


def __getattr__(name: str) -> Any:
    """Load Ray Jobs integrations only when a caller requests them."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'tributo' has no attribute {name!r}") from exc
    import importlib

    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose stable lazy root symbols to interactive callers."""
    return sorted({*globals(), *__all__})


__all__: list[str] = [
    # Core
    "TributoClient",
    "RayJob",
    "JobConfig",
    "RuntimeTarget",
    "RuntimeExecutionMode",
    "RuntimeSubmissionMode",
    "RuntimeLifecycle",
    "RuntimeLease",
    "LocalRayJobsProvider",
    "RuntimeProvider",
    "open_job_submission_client",
    "open_ray_client",
    "register_runtime_provider",
    "resolve_runtime_provider",
    "run_local_entrypoint",
    # Exceptions
    "TributoError",
    "JobSubmissionError",
    "JobExecutionError",
    "JobConfigurationError",
    "JobTimeoutError",
    "ModelExportError",
    "DataSourceError",
]
