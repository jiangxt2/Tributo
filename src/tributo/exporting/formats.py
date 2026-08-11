"""Open format identifiers shared by export configuration and plugins.

Formats are extension points, not an enum.  The core only freezes their
portable spelling so a newly installed exporter can introduce a format
without changing Tributo's planner or executor.
"""

from __future__ import annotations

import re

FORMAT_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_FORMAT_ID_RE = re.compile(FORMAT_ID_PATTERN)


def validate_format_id(value: str) -> str:
    """Return *value* when it is a canonical open format identifier.

    Canonical identifiers are lowercase kebab-case ASCII strings such as
    ``onnx``, ``ubj``, ``xgboost-json``, and ``safetensors``.  This validates
    spelling only; it deliberately does not consult a central allowlist.
    """
    if not isinstance(value, str) or not _FORMAT_ID_RE.fullmatch(value):
        raise ValueError(
            f"format {value!r} must match {FORMAT_ID_PATTERN!r} (lowercase kebab-case)"
        )
    return value
