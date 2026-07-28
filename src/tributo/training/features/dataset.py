"""PyTorch Dataset adapter.

Converts preprocessed feature data into PyTorch tensor format.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from tributo.training.features.column_types import (
    DenseFeat,
    SparseFeat,
)


class IdentityDataset:
    """Identity mining dataset.

    Converts preprocessed feature data and labels into dictionary format,
    compatible with the PyTorch Dataset interface.

    Attributes:
        sparse_features: List of Sparse feature names.
        dense_features: List of Dense feature names.
        labels: Label array.
        data: Preprocessed feature data dictionary.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        labels: np.ndarray,
        features: list[SparseFeat | DenseFeat],
    ) -> None:
        """Initialize the dataset.

        Args:
            data: Preprocessed feature data dictionary, key is feature name, value is feature value array.
            labels: Label array.
            features: List of feature column configurations.
        """
        self.data = data
        self.labels = labels
        self.features = features
        self.sparse_features = [f.name for f in features if isinstance(f, SparseFeat)]
        self.dense_features = [f.name for f in features if isinstance(f, DenseFeat)]
        self._length = len(labels)

    def __len__(self) -> int:
        """Return the dataset size."""
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing features and label.
        """
        sample: dict[str, Any] = {}

        # Sparse features
        for name in self.sparse_features:
            sample[name] = self.data[name][idx]

        # Dense features
        for name in self.dense_features:
            sample[name] = self.data[name][idx]

        # Label
        sample["label"] = self.labels[idx]

        return sample

    def to_torch_dataset(self) -> TorchIdentityDataset:
        """Convert to a PyTorch Dataset.

        Returns:
            TorchIdentityDataset instance.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for to_torch_dataset(). "
                "Install with: pip install torch"
            )
        return TorchIdentityDataset(self)


if HAS_TORCH:

    class TorchIdentityDataset(Dataset):  # type: ignore[type-arg]
        """PyTorch Dataset wrapper.

        Wraps IdentityDataset as a standard PyTorch Dataset,
        returning features and labels as PyTorch tensors.
        """

        def __init__(self, dataset: IdentityDataset) -> None:
            """Initialize the PyTorch Dataset.

            Args:
                dataset: IdentityDataset instance.
            """
            self.dataset = dataset
            self._length = len(dataset)

            # Pre-convert to tensors for performance
            self._sparse_tensors = {
                name: torch.tensor(dataset.data[name], dtype=torch.long)
                for name in dataset.sparse_features
            }
            self._dense_tensors = {
                name: torch.tensor(dataset.data[name], dtype=torch.float32)
                for name in dataset.dense_features
            }
            self._labels = torch.tensor(dataset.labels, dtype=torch.float32)

        def __len__(self) -> int:
            """Return the dataset size."""
            return self._length

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            """Get a single sample as tensors.

            Args:
                idx: Sample index.

            Returns:
                Dictionary containing feature tensors and label tensor.
            """
            sample: dict[str, torch.Tensor] = {}

            for name, tensor in self._sparse_tensors.items():
                sample[name] = tensor[idx]

            for name, tensor in self._dense_tensors.items():
                sample[name] = tensor[idx]

            sample["label"] = self._labels[idx]

            return sample
