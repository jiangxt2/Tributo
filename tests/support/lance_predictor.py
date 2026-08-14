"""Importable user-owned Predictor fixture for Ray cluster integration tests."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


class HFLikePredictor:
    """Small Predictor with explicit task and output semantics."""

    def __init__(self, model_uri: str, predictor_config: dict[str, object]) -> None:
        self.model_uri = model_uri
        self.predictor_config = predictor_config
        bias = predictor_config.get("bias", 0.0)
        self.model_bias = float(bias) if isinstance(bias, (int, float)) else 0.0

    def __call__(self, batch: pa.Table) -> pa.Table:
        values: list[Any] = batch.column("value").to_pylist()
        vectors = [[value + self.model_bias, value * 2] for value in values]
        return pa.table(
            {
                "id": batch.column("id"),
                "vector": pa.array(vectors, type=pa.list_(pa.float32(), 2)),
            }
        )
