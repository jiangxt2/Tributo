"""Ray runtime environment builder.

Ships application code through Ray-managed ``working_dir`` and ``py_modules``.
Python package dependencies remain owned by the cluster image, trusted runtime
configuration, or an explicit algorithm artifact instead of being merged from
unrelated environments.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tributo.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tributo.algorithms.api.artifacts import (
        AlgorithmArtifact,
        ImageProfile,
    )


# Directories and files excluded by default from working_dir archive
DEFAULT_EXCLUDES = [
    "**/.git",
    "**/.venv",
    "**/.idea",
    "**/__pycache__",
    "**/*.pyc",
    "**/uv.lock",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "docs/**",
    "**/*.md",
    "!README.md",
    ".gitignore",
]


@DeveloperAPI
def find_project_root(start: Path | None = None) -> Path:
    """Walk up the directory tree to find the project root (anchored by pyproject.toml).

    Args:
        start: Starting path, defaults to the current file's directory.

    Returns:
        The directory containing ``pyproject.toml``.

    Raises:
        FileNotFoundError: No ``pyproject.toml`` found in any parent directory.
    """
    if start is None:
        start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found in any parent directory")


@DeveloperAPI
def build_runtime_env(
    *,
    project_root: Path | None = None,
    package_name: str = "tributo",
    extra_excludes: list[str] | None = None,
    extra_py_modules: list[str | Path] | None = None,
    runtime_pip_packages: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    pythonpath: str | None = None,
    algorithm_artifact: AlgorithmArtifact | None = None,
    image_profile: ImageProfile | None = None,
    declared_dependencies: tuple[str, ...] = (),
    execution_context: Mapping[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Build a runtime_env dict suitable for the Ray Jobs API.

    Core mechanism:
    1. ``py_modules`` uploads the latest package code, taking priority over the
       older version baked into the image;
    2. ``working_dir`` uploads the project root, providing entrypoint scripts;

    Package dependencies must be installed in the cluster image, supplied by
    trusted deployment configuration, or distributed through an explicit,
    preflighted algorithm artifact. Extension inputs must not be derived from
    an untrusted task payload. Core copies them without scanning the submitting
    process or resolving dependencies. When ``algorithm_artifact`` is present,
    the builder validates the Wheel or Bundle against ``image_profile`` and
    owns the artifact-related ``working_dir``, ``py_modules``, ``pip``, and
    ``excludes`` fields.

    Args:
        project_root: Project root directory. When ``None``, walks up to find
            ``pyproject.toml`` automatically.
        package_name: Package name to upload for priority override, defaults to
            ``tributo``.
        extra_excludes: Additional directories/files to exclude (glob patterns).
        extra_py_modules: Trusted extension modules appended after the Tributo
            Core package in caller-provided order. Paths are converted to
            strings. Do not populate this from a broker task payload.
        runtime_pip_packages: Trusted extension requirements copied to Ray's
            ``pip`` runtime environment. Core does not resolve them. This must
            not be combined with ``algorithm_artifact``.
        env_vars: Additional environment variables.
        pythonpath: Optional cluster-visible ``PYTHONPATH`` to append to an
            explicitly supplied value. ``None`` leaves ``PYTHONPATH`` untouched.
            Do not pass paths derived from the submitting host.
        algorithm_artifact: Optional user Wheel or offline Wheelhouse Bundle.
            Image mode requires a code-only Wheel; offline mode requires a
            complete local or attested remote Bundle.
        image_profile: Immutable image compatibility record required when an
            algorithm artifact is supplied.
        declared_dependencies: Additional PEP 508 constraints checked against
            the selected image or offline Wheelhouse.
        execution_context: Versioned broker-neutral worker factory context.

    Returns:
        A dict that can be passed directly to
        ``JobSubmissionClient.submit_job(runtime_env=...)``.

    Raises:
        ValueError: ``runtime_pip_packages`` and ``algorithm_artifact`` both
            request ownership of Ray's ``pip`` runtime environment.

    Example:
        >>> from tributo._common.runtime_env import build_runtime_env
        >>> runtime_env = build_runtime_env()
        >>> client.submit_job(
        ...     entrypoint="python train.py",
        ...     runtime_env=runtime_env,
        ... )
    """
    if runtime_pip_packages and algorithm_artifact is not None:
        raise ValueError(
            "runtime_pip_packages cannot be combined with algorithm_artifact"
        )

    if project_root is None:
        try:
            root = find_project_root()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "project_root is required when Tributo is not running from a "
                "source checkout containing pyproject.toml"
            ) from exc
    else:
        root = project_root
    src_pkg = root / "src" / package_name
    if not src_pkg.exists():
        # Compat with flat layout (package directly under root)
        src_pkg = root / package_name

    if not src_pkg.exists():
        raise FileNotFoundError(
            f"Package directory not found: tried {root / 'src' / package_name} "
            f"and {root / package_name}"
        )

    excludes = list(DEFAULT_EXCLUDES)
    if extra_excludes:
        excludes.extend(extra_excludes)

    merged_env_vars: dict[str, str] = {}
    if env_vars:
        merged_env_vars.update(env_vars)
    if execution_context is not None:
        merged_env_vars["TRIBUTO_EXECUTION_CONTEXT"] = _serialize_execution_context(
            execution_context
        )

    if pythonpath is not None:
        existing_paths = merged_env_vars.get("PYTHONPATH", "").split(":")
        requested_paths = pythonpath.split(":")
        paths = list(
            dict.fromkeys(path for path in (*existing_paths, *requested_paths) if path)
        )
        if paths:
            merged_env_vars["PYTHONPATH"] = ":".join(paths)

    runtime_env: dict[str, Any] = {
        "working_dir": str(root),
        "excludes": excludes,
        "py_modules": [
            str(src_pkg),
            *(str(module) for module in extra_py_modules or ()),
        ],
    }
    if merged_env_vars:
        runtime_env["env_vars"] = merged_env_vars
    if runtime_pip_packages:
        runtime_env["pip"] = list(runtime_pip_packages)

    if algorithm_artifact is None and (
        image_profile is not None or declared_dependencies
    ):
        raise ValueError(
            "image_profile and declared_dependencies require algorithm_artifact"
        )

    if algorithm_artifact is not None:
        if image_profile is None:
            raise ValueError(
                "image_profile is required when algorithm_artifact is configured"
            )
        from tributo._common.algorithm_distribution import (
            algorithm_runtime_env_patch,
            prepare_algorithm_distribution,
        )

        prepared = prepare_algorithm_distribution(
            algorithm_artifact,
            image_profile,
            declared_dependencies=declared_dependencies,
        )
        artifact_patch = algorithm_runtime_env_patch(
            prepared,
            existing_env_vars=merged_env_vars,
        )
        if "working_dir" in artifact_patch:
            runtime_env["working_dir"] = artifact_patch["working_dir"]
        if "excludes" in artifact_patch:
            runtime_env["excludes"] = artifact_patch["excludes"]
        if "py_modules" in artifact_patch:
            runtime_env["py_modules"] = [
                *runtime_env.get("py_modules", []),
                *artifact_patch["py_modules"],
            ]
        if "pip" in artifact_patch:
            runtime_env["pip"] = artifact_patch["pip"]
        merged_env_vars = artifact_patch["env_vars"]
        runtime_env["env_vars"] = merged_env_vars

    logger.debug(
        "Built runtime_env: py_module_count=%d, has_pip=%s, env_var_keys=%s",
        len(runtime_env["py_modules"]),
        "pip" in runtime_env,
        sorted(merged_env_vars),
    )
    return runtime_env


def _serialize_execution_context(context: Mapping[str, Any] | Any) -> str:
    from tributo.training.execution_context import ExecutionContext

    value = context.as_dict() if hasattr(context, "as_dict") else context
    if not isinstance(value, Mapping):
        raise TypeError("execution_context must be a mapping or expose as_dict()")
    plain = ExecutionContext.from_mapping(value).as_dict()
    return json.dumps(plain, sort_keys=True, separators=(",", ":"))
