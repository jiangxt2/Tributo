"""Ray 2.55.1 DataConfig compatibility for exact row coverage.

Ray Train's stable ``DataConfig`` is subclassable, but its DeveloperAPI
``configure`` implementation hard-codes ``streaming_split(equal=True)``. This
version-locked adapter preserves Ray's execution options, resource exclusion,
and locality behavior while selecting the public ``equal=False`` Dataset API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ray.data.aggregate import AggregateFnV2
from ray.data.block import Block, BlockAccessor
from ray.train import DataConfig

from tributo.algorithms.api import AlgorithmConfigurationError
from tributo.training.features.column_types import DenseFeat, NormMethod, SparseFeat
from tributo.training.features.transformer import FeatureTransformer
from tributo.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.data import DataIterator, Dataset, NodeIdStr


@DeveloperAPI
class ExactCoverageDataConfig(DataConfig):
    """Keep every input row when Ray Train assigns Dataset iterators."""

    @DeveloperAPI
    def configure(
        self,
        datasets: dict[str, Dataset],
        world_size: int,
        worker_handles: list[ActorHandle] | None,
        worker_node_ids: list[NodeIdStr] | None,
        **kwargs: Any,
    ) -> list[dict[str, DataIterator]]:
        """Mirror Ray 2.55.1 ``DataConfig`` except for ``equal=False``."""
        del worker_handles, kwargs
        from ray.data import ExecutionResources

        output: list[dict[str, DataIterator]] = [{} for _ in range(world_size)]
        for dataset_name, dataset in datasets.items():
            if dataset.name is None:
                dataset.set_name(dataset_name)

        datasets_to_split = (
            set(datasets)
            if self._datasets_to_split == "all"
            else set(self._datasets_to_split)
        )
        locality_hints = worker_node_ids if self._enable_shard_locality else None
        for name, dataset in datasets.items():
            execution_options = self._get_execution_options(name)
            if (
                execution_options.is_resource_limits_default()
                and not self._is_v2_autoscaler()
            ):
                execution_options.exclude_resources = (
                    execution_options.exclude_resources.add(
                        ExecutionResources(
                            cpu=self._num_train_cpus,
                            gpu=self._num_train_gpus,
                        )
                    )
                )

            configured = dataset.copy(dataset)
            configured.context.execution_options = execution_options
            if name in datasets_to_split:
                splits = configured.streaming_split(
                    world_size,
                    equal=False,
                    locality_hints=locality_hints,
                )
                for rank, split in enumerate(splits):
                    output[rank][name] = split
            else:
                for rank in range(world_size):
                    output[rank][name] = configured.iterator()
        return output


_FeatureState = dict[str, dict[str, Any]]


def _category_identity(value: object) -> tuple[str, str]:
    return type(value).__qualname__, repr(value)


def _merge_feature_states(
    left: _FeatureState,
    right: _FeatureState,
    *,
    sparse_limits: Mapping[str, int],
) -> _FeatureState:
    merged: _FeatureState = {name: dict(state) for name, state in left.items()}
    for name, state in right.items():
        if name in sparse_limits:
            values = {
                _category_identity(value): value
                for value in (*merged.get(name, {}).get("values", ()), *state["values"])
            }
            if len(values) >= sparse_limits[name]:
                raise AlgorithmConfigurationError(
                    f"Sparse feature {name!r} has at least {len(values)} categories "
                    f"but vocab_size={sparse_limits[name]} must reserve one "
                    "unknown-category index"
                )
            merged[name] = {"values": tuple(values[key] for key in sorted(values))}
            continue
        current = merged.get(name)
        if current is None:
            merged[name] = dict(state)
            continue
        merged[name] = {
            "count": int(current["count"]) + int(state["count"]),
            "sum": float(current["sum"]) + float(state["sum"]),
            "square_sum": float(current["square_sum"]) + float(state["square_sum"]),
            "minimum": min(float(current["minimum"]), float(state["minimum"])),
            "maximum": max(float(current["maximum"]), float(state["maximum"])),
        }
    return merged


class _DNNFeatureSummary(AggregateFnV2[_FeatureState, _FeatureState]):
    def __init__(self, features: list[SparseFeat | DenseFeat]) -> None:
        self._features = tuple(features)
        self._sparse_limits = {
            feature.name: feature.vocab_size
            for feature in features
            if isinstance(feature, SparseFeat) and not feature.use_hash
        }
        super().__init__(
            "dnn_feature_summary",
            zero_factory=dict,
            on=None,
            ignore_nulls=True,
        )

    def aggregate_block(self, block: Block) -> _FeatureState:
        raw = BlockAccessor.for_block(block).to_numpy(
            [feature.name for feature in self._features]
        )
        if not isinstance(raw, Mapping):
            raise AlgorithmConfigurationError(
                "DNN feature summary requires named Ray Data columns"
            )
        state: _FeatureState = {}
        for feature in self._features:
            values = np.asarray(raw[feature.name]).reshape(-1)
            if isinstance(feature, SparseFeat):
                if feature.use_hash:
                    continue
                valid = values[~pd.isna(values)]
                unique = {
                    _category_identity(
                        value.item() if hasattr(value, "item") else value
                    ): (value.item() if hasattr(value, "item") else value)
                    for value in valid
                }
                if len(unique) >= feature.vocab_size:
                    raise AlgorithmConfigurationError(
                        f"Sparse feature {feature.name!r} has at least {len(unique)} "
                        f"categories but vocab_size={feature.vocab_size} must reserve "
                        "one unknown-category index"
                    )
                state[feature.name] = {
                    "values": tuple(unique[key] for key in sorted(unique))
                }
                continue
            if feature.norm is NormMethod.NONE:
                continue
            numeric = values.astype(np.float64)
            valid_numeric = numeric[np.isfinite(numeric)]
            if not valid_numeric.size:
                continue
            state[feature.name] = {
                "count": int(valid_numeric.size),
                "sum": float(valid_numeric.sum(dtype=np.float64)),
                "square_sum": float(np.square(valid_numeric).sum(dtype=np.float64)),
                "minimum": float(valid_numeric.min()),
                "maximum": float(valid_numeric.max()),
            }
        return state

    def combine(
        self, current_accumulator: _FeatureState, new: _FeatureState
    ) -> _FeatureState:
        return _merge_feature_states(
            current_accumulator,
            new,
            sparse_limits=self._sparse_limits,
        )


@DeveloperAPI
def fit_dnn_feature_transformer(
    dataset: Dataset,
    features: list[SparseFeat | DenseFeat],
) -> FeatureTransformer:
    """Fit all bounded DNN preprocessing state in one Ray Data aggregate."""
    summary = dataset.aggregate(_DNNFeatureSummary(features))
    if not isinstance(summary, Mapping):
        raise AlgorithmConfigurationError("DNN feature summary is malformed")
    state = summary.get("dnn_feature_summary")
    if not isinstance(state, Mapping):
        raise AlgorithmConfigurationError("DNN feature summary is missing")
    transformer = FeatureTransformer(features)
    for feature in features:
        feature_state = state.get(feature.name)
        if isinstance(feature, SparseFeat):
            if feature.use_hash:
                continue
            if not isinstance(feature_state, Mapping):
                transformer.label_encoders[feature.name] = {}
                continue
            values = tuple(feature_state.get("values", ()))
            transformer.label_encoders[feature.name] = {
                value: index for index, value in enumerate(values)
            }
            continue
        if feature.norm is NormMethod.NONE:
            continue
        if not isinstance(feature_state, Mapping) or int(feature_state["count"]) <= 0:
            raise AlgorithmConfigurationError(
                f"Dense feature {feature.name!r} has no finite values"
            )
        if feature.norm is NormMethod.MINMAX:
            transformer.norm_params[feature.name] = {
                "min": float(feature_state["minimum"]),
                "max": float(feature_state["maximum"]),
            }
        elif feature.norm is NormMethod.STANDARD:
            count = float(feature_state["count"])
            mean = float(feature_state["sum"]) / count
            variance = max(
                0.0,
                float(feature_state["square_sum"]) / count - mean * mean,
            )
            transformer.norm_params[feature.name] = {
                "mean": mean,
                "std": math.sqrt(variance),
            }
        else:
            transformer.norm_params[feature.name] = {
                "min": float(feature_state["minimum"])
            }
    transformer.fitted = True
    return transformer


__all__ = ["ExactCoverageDataConfig", "fit_dnn_feature_transformer"]
