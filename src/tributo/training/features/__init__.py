"""Feature engineering module.

Provides feature column type definitions, preprocessing pipelines and PyTorch Dataset adapters.
"""

from __future__ import annotations

from tributo.training.features.column_types import DenseFeat, SparseFeat

__all__ = [
    "SparseFeat",
    "DenseFeat",
]
