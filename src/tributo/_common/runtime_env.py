"""Ray runtime_env builder.

Handles the `/venv` vs. anaconda dual-environment separation in official
Ray images by providing automated runtime_env configuration that ensures
dependencies are correctly loaded when submitting via the Jobs API.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from tributo.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)

# Standard anaconda site-packages path in official Ray images
_PY_VER = f"python{sys.version_info.major}.{sys.version_info.minor}"
DOCKER_PYTHONPATH = (
    f"/venv/lib/{_PY_VER}/site-packages:/home/ray/anaconda3/lib/{_PY_VER}/site-packages"
)

# Local dev environment uses the current Python's site-packages


def _default_pythonpath() -> str:
    """Auto-detect: anaconda path for Docker, current venv for local."""
    if os.path.isdir("/home/ray/anaconda3"):
        return DOCKER_PYTHONPATH
    sp = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    return str(sp)


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
) -> dict[str, Any]:
    """Build a runtime_env dict suitable for the Ray Jobs API.

    Core mechanism (addressing the Ray official image dual-environment pitfall):
    1. ``py_modules`` uploads the latest package code, taking priority over the
       older version baked into the image;
    2. ``working_dir`` uploads the project root, providing entrypoint scripts;
    3. ``env_vars.PYTHONPATH`` points to anaconda site-packages, allowing the
       ``/venv`` Python to find packages (e.g., ``ray``) only installed in anaconda.

    Args:
        project_root: Project root directory. When ``None``, walks up to find
            ``pyproject.toml`` automatically.
        package_name: Package name to upload for priority override, defaults to
            ``tributo``.
        extra_excludes: Additional directories/files to exclude (glob patterns).
        env_vars: Additional environment variables.
        pythonpath: PYTHONPATH default, includes both ``/venv`` and anaconda
            site-packages. Adjust if the cluster uses a different Python version.

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
    root = project_root or find_project_root()
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

    # Ensure PYTHONPATH includes both /venv and anaconda site-packages
    pythonpath = pythonpath or _default_pythonpath()
    existing_pythonpath = merged_env_vars.get("PYTHONPATH", "")
    if pythonpath not in existing_pythonpath:
        if existing_pythonpath:
            merged_env_vars["PYTHONPATH"] = f"{existing_pythonpath}:{pythonpath}"
        else:
            merged_env_vars["PYTHONPATH"] = pythonpath

    runtime_env: dict[str, Any] = {
        "working_dir": str(root),
        "excludes": excludes,
        "py_modules": [str(src_pkg)],
    }
    if merged_env_vars:
        runtime_env["env_vars"] = merged_env_vars

    logger.debug(
        "Built runtime_env: working_dir=%s, py_modules=%s, env_vars=%s",
        root,
        src_pkg,
        merged_env_vars,
    )
    return runtime_env
