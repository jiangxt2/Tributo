"""Model export module.

Provides PyTorch -> ONNX and other model export functionalities.
"""

from __future__ import annotations

from tributo.training.exporters.torch_onnx_exporter import export_pytorch_to_onnx

__all__ = [
    "export_pytorch_to_onnx",
]
