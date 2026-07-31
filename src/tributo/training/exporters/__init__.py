"""Deprecated re-exports from ``tributo.exporting``.

All bundle export functionality has moved to ``tributo.exporting``.
These re-exports exist for backward compatibility and will be removed
in a future release.
"""

from __future__ import annotations

import importlib
import warnings

from tributo.training.exporters.torch_onnx_exporter import export_pytorch_to_onnx

__all__ = ["export_pytorch_to_onnx"]


def __getattr__(name: str):
    if name in ("export_pytorch_to_onnx",):
        return globals()[name]
    warnings.warn(
        f"tributo.training.exporters.{name} is deprecated; "
        f"use tributo.exporting.{name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(f"tributo.exporting.{name}"), name)
