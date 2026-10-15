"""Locked environment values for ordinary runtime fixtures, independent of SDK defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path

_PACKAGES = tomllib.loads(
    (Path(__file__).parents[2] / "uv.lock").read_text(encoding="utf-8")
)["package"]


def locked_version(name: str) -> str:
    versions = {item["version"] for item in _PACKAGES if item["name"] == name}
    if len(versions) != 1:
        raise ValueError(f"Expected one locked version for {name!r}, found {versions}")
    return versions.pop()


RAY_VERSION = locked_version("ray")
DAFT_VERSION = locked_version("daft")
