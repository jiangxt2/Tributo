"""Feature preprocessing pipeline.

Provides Label Encoding, Hash Encoding, normalization and other preprocessing functions,
with parameter serialization for consistency across training, batch inference and online inference.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tributo.training.features.column_types import (
    DenseFeat,
    NormMethod,
    SparseFeat,
    features_from_dicts,
)

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """Feature preprocessor.

    Based on feature column configuration, preprocesses raw features:
    - Sparse: Label Encoding / Hash Encoding -> Embedding index
    - Dense: Missing value filling -> MinMax / Standard / Log transform

    Attributes:
        features: List of feature column configurations.
        label_encoders: Label Encoding mapping dictionary.
        norm_params: Dense feature normalization parameters.
        fitted: Whether the transformer has been fitted.
    """

    def __init__(self, features: list[SparseFeat | DenseFeat]) -> None:
        """Initialize the preprocessor.

        Args:
            features: List of feature column configurations.
        """
        self.features = features
        self.label_encoders: dict[str, dict[Any, int]] = {}
        self.norm_params: dict[str, dict[str, float]] = {}
        self.fitted = False

    def fit(self, data: dict[str, np.ndarray]) -> FeatureTransformer:
        """Fit preprocessing parameters.

        Args:
            data: Raw feature data dictionary, key is feature name, value is feature value array.

        Returns:
            self, supporting chaining.
        """
        for feat in self.features:
            if isinstance(feat, SparseFeat):
                self._fit_sparse(feat, data[feat.name])
            elif isinstance(feat, DenseFeat):
                self._fit_dense(feat, data[feat.name])

        self.fitted = True
        return self

    def transform(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform feature data.

        Args:
            data: Raw feature data dictionary.

        Returns:
            Preprocessed feature data dictionary.

        Raises:
            RuntimeError: If the preprocessor has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("FeatureTransformer not fitted. Call fit() first.")

        result: dict[str, np.ndarray] = {}
        for feat in self.features:
            if isinstance(feat, SparseFeat):
                result[feat.name] = self._transform_sparse(feat, data[feat.name])
            elif isinstance(feat, DenseFeat):
                result[feat.name] = self._transform_dense(feat, data[feat.name])

        return result

    def fit_transform(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Fit and transform feature data.

        Args:
            data: Raw feature data dictionary.

        Returns:
            Preprocessed feature data dictionary.
        """
        return self.fit(data).transform(data)

    def _fit_sparse(self, feat: SparseFeat, values: np.ndarray) -> None:
        """Fit Sparse feature parameters."""
        if feat.use_hash:
            # Hash Encoding does not require fitting
            return

        # Label Encoding: build value -> index mapping
        # Use pd.isna for string types, np.isnan for numeric types
        if values.dtype.kind in ("U", "S", "O"):  # string types
            mask = pd.isna(values)
        else:
            mask = np.isnan(values.astype(float))
        unique_values = np.unique(values[~mask])
        encoder = {v: i for i, v in enumerate(unique_values)}
        self.label_encoders[feat.name] = encoder
        logger.debug(
            "SparseFeat '%s': fitted with %d unique values", feat.name, len(encoder)
        )

    def _fit_dense(self, feat: DenseFeat, values: np.ndarray) -> None:
        """Fit Dense feature normalization parameters."""
        # Use np.isnan for numeric types
        valid_values = values[~np.isnan(values.astype(float))].astype(np.float64)

        if feat.norm == NormMethod.MINMAX:
            min_val = float(np.min(valid_values))
            max_val = float(np.max(valid_values))
            self.norm_params[feat.name] = {"min": min_val, "max": max_val}
        elif feat.norm == NormMethod.STANDARD:
            mean_val = float(np.mean(valid_values))
            std_val = float(np.std(valid_values))
            self.norm_params[feat.name] = {"mean": mean_val, "std": std_val}
        elif feat.norm == NormMethod.LOG:
            # Log transform only needs the minimum value for offset
            min_val = float(np.min(valid_values))
            self.norm_params[feat.name] = {"min": min_val}
        # NormMethod.NONE does not require parameters

        logger.debug("DenseFeat '%s': fitted with norm=%s", feat.name, feat.norm)

    def _transform_sparse(self, feat: SparseFeat, values: np.ndarray) -> np.ndarray:
        """Transform Sparse features."""
        if feat.use_hash:
            # Hash Encoding: use deterministic hashing (consistent across processes)
            bucket_size = feat.hash_bucket_size
            return np.array(
                [
                    int(hashlib.md5(str(v).encode()).hexdigest(), 16) % bucket_size
                    for v in values
                ],
                dtype=np.int64,
            )

        # Label Encoding: use the fitted mapping
        encoder = self.label_encoders.get(feat.name, {})
        default_idx = len(encoder)  # unknown values map to the end
        return np.array([encoder.get(v, default_idx) for v in values], dtype=np.int64)

    def _transform_dense(self, feat: DenseFeat, values: np.ndarray) -> np.ndarray:
        """Transform Dense features."""
        result = values.astype(np.float64)

        # Fill missing values (using 0 or mean)
        nan_mask = np.isnan(result)
        if nan_mask.any():
            result[nan_mask] = 0.0

        params = self.norm_params.get(feat.name, {})

        if feat.norm == NormMethod.MINMAX:
            min_val = params["min"]
            max_val = params["max"]
            range_val = max_val - min_val
            if range_val > 0:
                result = (result - min_val) / range_val
            else:
                result = np.zeros_like(result)
        elif feat.norm == NormMethod.STANDARD:
            mean_val = params["mean"]
            std_val = params["std"]
            if std_val > 0:
                result = (result - mean_val) / std_val
            else:
                result = result - mean_val
        elif feat.norm == NormMethod.LOG:
            min_val = params["min"]
            # Log(1 + x - min) ensures non-negative
            result = np.log1p(np.maximum(result - min_val, 0))
        # NormMethod.NONE: no transformation

        return result.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save preprocessing parameters to a JSON file.

        Args:
            path: Save path.
        """
        path = Path(path)
        state = {
            "features": [
                {
                    "type": "sparse" if isinstance(f, SparseFeat) else "dense",
                    "name": f.name,
                    **(
                        {
                            "vocab_size": f.vocab_size,
                            "embedding_dim": f.embedding_dim,
                            "use_hash": f.use_hash,
                            "hash_bucket_size": f.hash_bucket_size,
                        }
                        if isinstance(f, SparseFeat)
                        else {
                            "dimension": f.dimension,
                            "norm": f.norm.value,
                        }
                    ),
                }
                for f in self.features
            ],
            "label_encoders": {
                k: {
                    str(kk): int(vv) if isinstance(vv, (np.integer, np.int64)) else vv
                    for kk, vv in v.items()
                }
                for k, v in self.label_encoders.items()
            },
            "norm_params": self.norm_params,
        }
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        logger.info("Saved FeatureTransformer to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FeatureTransformer:
        """Load a preprocessor from a JSON file.

        Args:
            path: File path.

        Returns:
            Loaded FeatureTransformer instance.
        """
        path = Path(path)
        state = json.loads(path.read_text())

        # Rebuild feature column configuration
        features = features_from_dicts(state["features"])

        transformer = cls(features)

        # Load label_encoders, try to convert string keys back to original types
        def _try_convert_key(key: str) -> Any:
            """Try to convert a string key back to its numeric type."""
            try:
                # Try converting to int
                return int(key)
            except ValueError:
                try:
                    # Try converting to float
                    return float(key)
                except ValueError:
                    # Keep as string
                    return key

        transformer.label_encoders = {
            k: {_try_convert_key(kk): vv for kk, vv in v.items()}
            for k, v in state["label_encoders"].items()
        }
        transformer.norm_params = state["norm_params"]
        transformer.fitted = True

        logger.info("Loaded FeatureTransformer from %s", path)
        return transformer
