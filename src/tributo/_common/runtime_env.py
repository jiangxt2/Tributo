"""Ray runtime environment builder.

Ships application code through Ray-managed ``working_dir`` and ``py_modules``.
Python package dependencies remain owned by the cluster image or an explicit
Ray runtime environment instead of being merged from unrelated environments.
"""

from __future__ import annotations

import logging
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
    env_vars: dict[str, str] | None = None,
    pythonpath: str | None = None,
    algorithm_artifact: AlgorithmArtifact | None = None,
    image_profile: ImageProfile | None = None,
    declared_dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a runtime_env dict suitable for the Ray Jobs API.

    Core mechanism:
    1. ``py_modules`` uploads the latest package code, taking priority over the
       older version baked into the image;
    2. ``working_dir`` uploads the project root, providing entrypoint scripts;

    Package dependencies must be installed in the cluster image or supplied by
    an explicit, preflighted offline Ray runtime environment. When
    ``algorithm_artifact`` is present, the builder validates the Wheel or
    Bundle against ``image_profile`` and owns the artifact-related
    ``working_dir``, ``py_modules``, ``pip``, and ``excludes`` fields. It never
    derives package paths from the submitting process or combines unrelated
    ``site-packages`` trees.

    Args:
        project_root: Project root directory. When ``None``, walks up to find
            ``pyproject.toml`` automatically.
        package_name: Package name to upload for priority override, defaults to
            ``tributo``.
        extra_excludes: Additional directories/files to exclude (glob patterns).
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

    Returns:
        A dict that can be passed directly to
        ``JobSubmissionClient.submit_job(runtime_env=...)``.

    Example:
        >>> from tributo._common.runtime_env import build_runtime_env
        >>> runtime_env = build_runtime_env()
        >>> client.submit_job(
        ...     entrypoint="python train.py",
        ...     runtime_env=runtime_env,
        ... )
    """
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
        "py_modules": [str(src_pkg)],
    }
    if merged_env_vars:
        runtime_env["env_vars"] = merged_env_vars

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
        "Built runtime_env: working_dir=%s, py_modules=%s, env_var_keys=%s",
        runtime_env["working_dir"],
        runtime_env["py_modules"],
        sorted(merged_env_vars),
    )
    return runtime_env
