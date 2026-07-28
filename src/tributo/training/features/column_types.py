"""Feature column type definitions.

References DeepCTR's SparseFeat / DenseFeat abstraction to define feature column configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tributo.util.annotations import PublicAPI


class NormMethod(str, Enum):
    """Dense feature normalization method."""

    MINMAX = "minmax"
    STANDARD = "standard"
    LOG = "log"
    NONE = "none"


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class SparseFeat:
    """Discrete categorical feature column configuration.

    Discrete features require an Embedding layer to map category IDs to dense vectors.

    Attributes:
        name: Feature column name, must match input data column name.
        vocab_size: Discrete value space size (number of categories).
        embedding_dim: Embedding dimension.
        use_hash: Whether to use Hash Encoding (high cardinality scenario).
        hash_bucket_size: Hash bucket size, only effective when use_hash=True.
        dtype: Feature data type, default int64.
    """

    name: str
    vocab_size: int
    embedding_dim: int = 8
    use_hash: bool = False
    hash_bucket_size: int = 100000
    dtype: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration legality."""
        if self.vocab_size <= 0:
            raise ValueError(
                f"SparseFeat '{self.name}': vocab_size must be positive, "
                f"got {self.vocab_size}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"SparseFeat '{self.name}': embedding_dim must be positive, "
                f"got {self.embedding_dim}"
            )
        if self.use_hash and self.hash_bucket_size <= 0:
            raise ValueError(
                f"SparseFeat '{self.name}': hash_bucket_size must be positive "
                f"when use_hash=True, got {self.hash_bucket_size}"
            )


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class DenseFeat:
    """Continuous numerical feature column configuration.

    Continuous features are fed directly into the model, with optional normalization.

    Attributes:
        name: Feature column name, must match input data column name.
        dimension: Feature dimension (typically 1; multi-dimensional features like embeddings can be > 1).
        norm: Normalization method: minmax / standard / log / none.
        dtype: Feature data type, default float32.
    """

    name: str
    dimension: int = 1
    norm: NormMethod = NormMethod.NONE
    dtype: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration legality."""
        if self.dimension <= 0:
            raise ValueError(
                f"DenseFeat '{self.name}': dimension must be positive, "
                f"got {self.dimension}"
            )


def get_feature_names(features: list[SparseFeat | DenseFeat]) -> list[str]:
    """Get the list of feature column names.

    Args:
        features: List of feature column configurations.

    Returns:
        List of feature column names.
    """
    return [feat.name for feat in features]


def get_sparse_features(
    features: list[SparseFeat | DenseFeat],
) -> list[SparseFeat]:
    """Filter Sparse feature columns.

    Args:
        features: List of feature column configurations.

    Returns:
        List of SparseFeat columns.
    """
    return [f for f in features if isinstance(f, SparseFeat)]


def get_dense_features(
    features: list[SparseFeat | DenseFeat],
) -> list[DenseFeat]:
    """Filter Dense feature columns.

    Args:
        features: List of feature column configurations.

    Returns:
        List of DenseFeat columns.
    """
    return [f for f in features if isinstance(f, DenseFeat)]


# ── Default value constants ──

DEFAULT_VOCAB_SIZE: int = 1000
DEFAULT_HASH_BUCKET_SIZE: int = 100_000
DEFAULT_EMBEDDING_DIM: int = 8


def features_from_dicts(
    dicts: list[dict[str, Any]],
) -> list[SparseFeat | DenseFeat]:
    """Build a list of feature columns from a list of dictionaries.

    Unified feature parsing factory, eliminating duplicate code across
    build_features_from_config, _parse_features, and dnn_train_loop_per_worker.

    Args:
        dicts: List of feature config dictionaries. Each dict must contain 'name'.
            'type' is optional; when missing it is inferred from 'vocab_size':
            - has vocab_size -> sparse
            - no vocab_size -> dense

    Returns:
        List of feature columns.

    Raises:
        ValueError: If type is not 'sparse' or 'dense'.

    Example::

        features = features_from_dicts([
            {"type": "sparse", "name": "dept", "vocab_size": 10},
            {"type": "dense", "name": "age", "norm": "standard"},
        ])
    """
    features: list[SparseFeat | DenseFeat] = []
    for d in dicts:
        feat_type = d.get("type", "")
        name = d.get("name", "")

        # Auto-infer type: has vocab_size -> sparse, otherwise -> dense
        if not feat_type:
            if "vocab_size" in d:
                feat_type = "sparse"
            else:
                feat_type = "dense"

        if feat_type == "sparse":
            features.append(
                SparseFeat(
                    name=name,
                    vocab_size=d.get("vocab_size", DEFAULT_VOCAB_SIZE),
                    embedding_dim=d.get("embedding_dim", DEFAULT_EMBEDDING_DIM),
                    use_hash=d.get("use_hash", False),
                    hash_bucket_size=d.get(
                        "hash_bucket_size", DEFAULT_HASH_BUCKET_SIZE
                    ),
                )
            )
        elif feat_type == "dense":
            features.append(
                DenseFeat(
                    name=name,
                    dimension=d.get("dimension", 1),
                    norm=NormMethod(d.get("norm", "none")),
                )
            )
        else:
            raise ValueError(
                f"Unknown feature type: '{feat_type}'. Expected 'sparse' or 'dense'."
            )
    return features
