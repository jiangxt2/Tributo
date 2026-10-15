"""Audited runtime baselines shared by SDK defaults and compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path

_BASELINE = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))


def _version(name: str) -> str:
    value = _BASELINE[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid runtime baseline {name!r}")
    return value


SUPPORTED_RAY_VERSION = _version("ray")
SUPPORTED_DAFT_VERSION = _version("daft")
RAY_VERSION_SPEC = f"=={SUPPORTED_RAY_VERSION}"
DAFT_SQL_VERSION_SPEC = _version("daft_sql_spec")
