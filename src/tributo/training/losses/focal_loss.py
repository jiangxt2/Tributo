"""Focal Loss implementation.

Reference: "Focal Loss for Dense Object Detection"
https://arxiv.org/abs/1708.02002

Used for handling class imbalance by down-weighting easily classified samples.
"""

from __future__ import annotations

import logging
from typing import Any

from tributo.util.annotations import PublicAPI

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # used by binary_cross_entropy_with_logits

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

if HAS_TORCH:

    @PublicAPI(stability="beta")
    class FocalLoss(nn.Module):
        """Focal Loss function.

        Down-weights easily classified samples to focus on hard examples.

        Attributes:
            alpha: Positive class weight for balancing positive and negative samples.
            gamma: Focusing parameter, larger values focus more on hard samples.
            reduction: Loss reduction method: 'mean' / 'sum' / 'none'.
        """

        def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = "mean",
        ) -> None:
            """Initialize Focal Loss.

            Args:
                alpha: Positive class weight, range [0, 1].
                gamma: Focusing parameter, typically 2.0.
                reduction: Loss reduction method.
            """
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Compute Focal Loss.

            Args:
                logits: Model output logits (without sigmoid).
                labels: Labels, 0 or 1.

            Returns:
                Loss value.
            """
            # Compute BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )

            # Compute probabilities
            probs = torch.sigmoid(logits)

            # Compute p_t
            p_t = probs * labels + (1 - probs) * (1 - labels)

            # Compute alpha_t
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)

            # Compute focal weight
            focal_weight = alpha_t * (1 - p_t) ** self.gamma

            # Compute focal loss
            focal_loss = focal_weight * bce_loss

            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            else:
                return focal_loss

    @PublicAPI(stability="beta")
    def focal_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Functional interface for computing Focal Loss.

        Args:
            logits: Model output logits.
            labels: Labels.
            alpha: Positive class weight.
            gamma: Focusing parameter.

        Returns:
            Loss value.
        """
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        loss: torch.Tensor = criterion(logits, labels)
        return loss

else:
    # Placeholder when PyTorch is not installed
    class FocalLoss:  # type: ignore[no-redef]
        """Placeholder class used when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required for FocalLoss. Install with: pip install torch"
            )

    def focal_loss(*args: Any, **kwargs: Any) -> Any:
        """Placeholder function used when PyTorch is not installed."""
        raise ImportError(
            "PyTorch is required for focal_loss. Install with: pip install torch"
        )
