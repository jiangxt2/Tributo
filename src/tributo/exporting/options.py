"""Deprecated compatibility imports for first-party exporter options.

Concrete option schemas are owned by their integration plugins.  This module
remains for the beta compatibility window and resolves old imports lazily.
"""

from __future__ import annotations

import warnings
from typing import Any

_COMPAT_NAMES = frozenset(
    {
        "HFONNXOptions",
        "ONNXQuantizerOptions",
        "SafetensorsOptions",
        "TorchONNXOptions",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve a legacy option class and emit its migration warning."""
    if name not in _COMPAT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warnings.warn(
        f"tributo.exporting.options.{name} is deprecated; import it from "
        "tributo.integrations.exporters.options instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from tributo.integrations.exporters import options as plugin_options

    return getattr(plugin_options, name)


__all__ = sorted(_COMPAT_NAMES)
