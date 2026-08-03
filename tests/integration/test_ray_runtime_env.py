"""Explicit Ray uv runtime-environment propagation tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ray_runtime_env


def test_uv_run_runtime_environment_is_detected() -> None:
    """The feature remains tested separately from local Ray data contracts."""
    from ray._private import ray_constants
    from ray._private.worker import _maybe_modify_runtime_env

    assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is True
    modified = _maybe_modify_runtime_env({}, _skip_env_hook=False)

    assert modified.get("py_executable", "").startswith("uv run")
