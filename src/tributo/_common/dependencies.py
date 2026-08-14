"""Unified dependency probing for core and optional dependencies.

Single source of truth for checking whether a dependency is importable
and meets the project's minimum version. ``probe_dependency`` is
side-effect free — it never imports the package; ``require_dependency``
adds the real import once the probe passes.

The top-level package name is the probe boundary: ``find_spec`` on a
dotted name imports the parent package first, which turns a missing
dependency into a ``ModuleNotFoundError`` instead of ``None`` (see PR
#17). Probing is always done on the top-level import name only; submodule
presence is therefore not checked here — it is surfaced at execution
time by the real import.
"""

from __future__ import annotations

import enum
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from types import ModuleType

from packaging.version import InvalidVersion, Version

from tributo.exceptions import TributoError
from tributo.util.annotations import PublicAPI

__all__ = [
    "DependencySpec",
    "DependencyState",
    "DependencyStatus",
    "DependencyUnavailableError",
    "MissingOptionalDependency",
    "probe_dependency",
    "require_dependency",
    # Convenience specs for dependencies probed across the codebase.
    "BOTO3",
    "LANCE",
    "ONNXMLTOOLS",
    "ONNXRUNTIME",
    "PYICEBERG",
    "SAFETENSORS",
    "TORCH",
    "TRANSFORMERS",
    "XGBOOST",
]


@PublicAPI(stability="beta")
class DependencyState(enum.Enum):
    """Availability state reported by a dependency probe."""

    AVAILABLE = "available"
    MISSING = "missing"
    TOO_OLD = "too_old"
    VERSION_UNKNOWN = "version_unknown"


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class DependencySpec:
    """Description of a dependency, keyed by its top-level import name.

    Args:
        import_name: Top-level module used in ``import`` statements.
        distribution_name: Distribution metadata name; may differ from
            the import name (e.g. ``import PIL`` / distribution
            ``Pillow``).
        minimum_version: Minimum supported version; ``None`` means no
            version floor.
        extra: Optional extra that provides the dependency; ``None``
            marks a core dependency.
    """

    import_name: str
    distribution_name: str
    minimum_version: str | None = None
    extra: str | None = None


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class DependencyStatus:
    """Result of a side-effect free dependency probe."""

    spec: DependencySpec
    state: DependencyState
    installed_version: str | None = None


@PublicAPI(stability="beta")
def probe_dependency(spec: DependencySpec) -> DependencyStatus:
    """Probe a dependency without importing it (no side effects).

    Steps:
      1. ``find_spec`` on the top-level import name only.
      2. ``ModuleNotFoundError`` / ``ValueError`` (namespace modules
         whose ``__spec__`` is ``None``) map to ``MISSING``.
      3. Distribution metadata decides ``AVAILABLE`` / ``TOO_OLD`` /
         ``VERSION_UNKNOWN``.
      4. Never falls back to a real import.

    A spec without ``minimum_version`` is ``AVAILABLE`` as soon as its
    top-level package is importable; metadata is not consulted.
    """
    try:
        module_spec = importlib.util.find_spec(spec.import_name)
    except (ModuleNotFoundError, ValueError):
        return DependencyStatus(spec, DependencyState.MISSING)
    if module_spec is None:
        return DependencyStatus(spec, DependencyState.MISSING)

    if spec.minimum_version is None:
        return DependencyStatus(spec, DependencyState.AVAILABLE)

    try:
        installed = importlib.metadata.version(spec.distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return DependencyStatus(spec, DependencyState.VERSION_UNKNOWN)

    meets = _version_meets_minimum(installed, spec.minimum_version)
    if meets is None:
        return DependencyStatus(spec, DependencyState.VERSION_UNKNOWN, installed)
    if meets:
        return DependencyStatus(spec, DependencyState.AVAILABLE, installed)
    return DependencyStatus(spec, DependencyState.TOO_OLD, installed)


@PublicAPI(stability="beta")
def require_dependency(spec: DependencySpec) -> ModuleType:
    """Probe, then import, the dependency's top-level package.

    Raises ``MissingOptionalDependency`` for optional dependencies and
    ``DependencyUnavailableError`` for core ones when the probe does not
    report ``AVAILABLE``. A ``ModuleNotFoundError`` whose ``name`` is the
    top-level import name (the package vanished between probe and
    import) is converted to the same error; internal import errors and
    missing transitive dependencies propagate unchanged.
    """
    status = probe_dependency(spec)
    if status.state is not DependencyState.AVAILABLE:
        raise _unavailable_error(spec, status)
    try:
        return importlib.import_module(spec.import_name)
    except ModuleNotFoundError as exc:
        if exc.name == spec.import_name:
            # The package vanished between probe and import; report the
            # actual condition instead of the probe's AVAILABLE state.
            raise _unavailable_error(
                spec, DependencyStatus(spec, DependencyState.MISSING)
            ) from exc
        raise


@PublicAPI(stability="beta")
class DependencyUnavailableError(TributoError):
    """A dependency is missing or below its minimum version.

    For optional dependencies (``spec.extra`` is not ``None``) the
    message includes an install hint; for core dependencies it reports a
    broken environment instead.
    """

    def __init__(self, spec: DependencySpec, status: DependencyStatus) -> None:
        self.spec = spec
        self.status = status
        super().__init__(_format_unavailable_message(spec, status))


@PublicAPI(stability="beta")
class MissingOptionalDependency(DependencyUnavailableError):
    """``DependencyUnavailableError`` for an optional dependency.

    Raised only for specs with a non-``None`` ``extra``.
    """


# Convenience specs for dependencies probed across the codebase. The
# minimum versions mirror pyproject.toml — pyproject.toml is the
# authoritative source.
ONNXRUNTIME = DependencySpec("onnxruntime", "onnxruntime", "1.20.0")
BOTO3 = DependencySpec("boto3", "boto3", "1.42.91", extra="s3")
TORCH = DependencySpec("torch", "torch", "2.5.0", extra="model-export-torch")
SAFETENSORS = DependencySpec(
    "safetensors", "safetensors", "0.4.3", extra="model-export-torch"
)
XGBOOST = DependencySpec("xgboost", "xgboost", "2.1.0", extra="training")
ONNXMLTOOLS = DependencySpec("onnxmltools", "onnxmltools", "1.13.0", extra="training")
TRANSFORMERS = DependencySpec("transformers", "transformers", "4.40.0", extra="hf")
# pylance is the PyPI distribution name of the Lance vector database
# (unrelated to the VS Code Pylance language server).
LANCE = DependencySpec("lance", "pylance", "4.0.0", extra="data")
PYICEBERG = DependencySpec("pyiceberg", "pyiceberg", "0.8.0", extra="data")


def _unavailable_error(
    spec: DependencySpec, status: DependencyStatus
) -> DependencyUnavailableError:
    if spec.extra is not None:
        return MissingOptionalDependency(spec, status)
    return DependencyUnavailableError(spec, status)


def _format_unavailable_message(spec: DependencySpec, status: DependencyStatus) -> str:
    if spec.extra is None:
        hint = (
            "This is a core Tributo dependency — the environment is broken "
            "or incomplete; reinstall tributo and its dependencies"
        )
    else:
        hint = f"Install it with: pip install tributo[{spec.extra}]"
    if status.state is DependencyState.MISSING:
        detail = "is not installed"
    elif status.state is DependencyState.TOO_OLD:
        detail = (
            f"is version {status.installed_version!r}, but "
            f"{spec.import_name}>={spec.minimum_version} is required"
        )
    else:
        detail = (
            f"is importable but its version could not be verified against "
            f"{spec.import_name}>={spec.minimum_version}"
        )
    return f"Dependency {spec.import_name!r} {detail}. {hint}"


def _version_meets_minimum(installed: str, minimum: str) -> bool | None:
    """Compare versions with standard PEP 440 semantics.

    ``None`` means that either value is not a valid PEP 440 version.  In
    particular, pre-release versions remain below the corresponding final
    release (for example, ``2.5.0rc1 < 2.5.0``), while post-release and
    local-version ordering follows the standard comparison rules.
    """
    try:
        return Version(installed) >= Version(minimum)
    except InvalidVersion:
        return None
