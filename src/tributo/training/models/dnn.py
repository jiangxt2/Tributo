"""Fully connected neural network model.

Supports Sparse/Dense feature inputs for identity mining binary classification tasks.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from tributo.training.features.column_types import (
    DenseFeat,
    SparseFeat,
)

logger = logging.getLogger(__name__)

from tributo.util.annotations import PublicAPI  # noqa: E402

if HAS_TORCH:

    @PublicAPI(stability="beta")
    class DNNModel(nn.Module):
        """Fully connected neural network model.

        Architecture: Embedding (Sparse) + Concat (Dense) -> DNN -> Output

        Attributes:
            features: List of feature column configurations.
            dnn_hidden_units: List of DNN hidden layer dimensions.
            dnn_dropout: Dropout rate.
            use_batch_norm: Whether to use BatchNorm.
        """

        def __init__(
            self,
            features: list[SparseFeat | DenseFeat],
            dnn_hidden_units: list[int] | None = None,
            dnn_dropout: float = 0.0,
            use_batch_norm: bool = False,
        ) -> None:
            """Initialize the DNN model.

            Args:
                features: List of feature column configurations.
                dnn_hidden_units: DNN hidden layer dimensions, default [256, 128, 64].
                dnn_dropout: Dropout rate.
                use_batch_norm: Whether to use BatchNorm.
            """
            super().__init__()

            if dnn_hidden_units is None:
                dnn_hidden_units = [256, 128, 64]

            self.features = features
            self.dnn_hidden_units = dnn_hidden_units
            self.dnn_dropout = dnn_dropout

            # Separate Sparse and Dense features
            self.sparse_features = [f for f in features if isinstance(f, SparseFeat)]
            self.dense_features = [f for f in features if isinstance(f, DenseFeat)]

            # Embedding layers
            self.embeddings = nn.ModuleDict()
            for feat in self.sparse_features:
                vocab_size = feat.hash_bucket_size if feat.use_hash else feat.vocab_size
                self.embeddings[feat.name] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=feat.embedding_dim,
                    padding_idx=0,
                )

            # Compute DNN input dimension
            dnn_input_dim = self._compute_dnn_input_dim()

            # DNN layers
            self.dnn = self._build_dnn(
                dnn_input_dim, dnn_hidden_units, dnn_dropout, use_batch_norm
            )

            # Output layer
            self.output_layer = nn.Linear(dnn_hidden_units[-1], 1)

            # Initialize weights
            self._init_weights()

        def _compute_dnn_input_dim(self) -> int:
            """Compute DNN input dimension."""
            dim = 0
            # Embedding dimensions of Sparse features
            for feat in self.sparse_features:
                dim += feat.embedding_dim
            # Dimensions of Dense features
            for feat in self.dense_features:
                dim += feat.dimension
            return dim

        def _build_dnn(
            self,
            input_dim: int,
            hidden_units: list[int],
            dropout: float,
            use_batch_norm: bool,
        ) -> nn.Sequential:
            """Build the DNN layers."""
            layers: list[nn.Module] = []
            prev_dim = input_dim

            for hidden_dim in hidden_units:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            return nn.Sequential(*layers)

        def _init_weights(self) -> None:
            """Initialize model weights."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0, std=0.01)

        def forward(
            self,
            inputs: dict[str, torch.Tensor],
            **kwargs: Any,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                inputs: Feature dictionary, key is feature name, value is feature tensor.

            Returns:
                Model output logits.
            """
            # Process Sparse features
            sparse_outputs = []
            for feat in self.sparse_features:
                feat_input = inputs[feat.name]
                # Ensure input is long type
                if feat_input.dtype != torch.long:
                    feat_input = feat_input.long()
                # Embedding lookup
                embed = self.embeddings[feat.name](feat_input)
                # Pooling for variable-length sequences (mean)
                if embed.dim() > 2:
                    embed = embed.mean(dim=1)
                sparse_outputs.append(embed)

            # Process Dense features
            dense_outputs = []
            for feat in self.dense_features:
                feat_input = inputs[feat.name]
                # Ensure input is float type
                if feat_input.dtype != torch.float32:
                    feat_input = feat_input.float()
                # Handle multi-dimensional features
                if feat_input.dim() == 1:
                    feat_input = feat_input.unsqueeze(-1)
                dense_outputs.append(feat_input)

            # Concatenate all features
            all_features = sparse_outputs + dense_outputs
            if len(all_features) == 1:
                dnn_input = all_features[0]
            else:
                dnn_input = torch.cat(all_features, dim=-1)

            # DNN forward pass
            dnn_output = self.dnn(dnn_input)

            # Output layer
            logits = self.output_layer(dnn_output).squeeze(-1)

            return logits

        def predict_proba(
            self,
            inputs: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            """Predict probabilities.

            Args:
                inputs: Feature dictionary.

            Returns:
                Predicted probabilities.
            """
            with torch.no_grad():
                logits = self.forward(inputs)
                return torch.sigmoid(logits)

else:
    # Placeholder when PyTorch is not installed
    class DNNModel:  # type: ignore[no-redef]
        """Placeholder class used when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required for DNNModel. Install with: pip install torch"
            )
