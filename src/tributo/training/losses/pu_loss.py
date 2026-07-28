"""nnPU (non-negative Positive-Unlabeled) loss function implementation.

Reference: "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
https://arxiv.org/abs/1703.00593

Implementation based on:
- https://github.com/cimeister/pu-learning
- https://github.com/kiryor/nnPUlearning
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

logger = logging.getLogger(__name__)

from tributo.util.annotations import PublicAPI  # noqa: E402

if HAS_TORCH:

    @PublicAPI(stability="beta")
    class PULoss(nn.Module):
        """nnPU loss function.

        Non-negative risk estimator for Positive-Unlabeled learning scenarios.

        Attributes:
            class_prior: Proportion of positive examples in unlabeled data (π_p).
            beta: Non-negative constraint threshold to prevent excessive negative risk.
            gamma: Negative risk scaling factor.
            loss_type: Loss type, 'nnpu' or 'upu'.
        """

        def __init__(
            self,
            class_prior: float,
            beta: float = 0.0,
            gamma: float = 1.0,
            loss_type: str = "nnpu",
        ) -> None:
            """Initialize nnPU loss.

            Args:
                class_prior: Positive class proportion, range (0, 1).
                beta: Non-negative constraint threshold.
                gamma: Negative risk scaling factor.
                loss_type: 'nnpu' (non-negative) or 'upu' (unbiased).

            Raises:
                ValueError: If class_prior is not in (0, 1).
            """
            super().__init__()
            if not 0 < class_prior < 1:
                raise ValueError(f"class_prior must be in (0, 1), got {class_prior}")
            if loss_type not in ("nnpu", "upu"):
                raise ValueError(f"loss_type must be 'nnpu' or 'upu', got {loss_type}")

            self.class_prior = class_prior
            self.beta = beta
            self.gamma = gamma
            self.loss_type = loss_type

        def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Compute nnPU loss.

            Args:
                logits: Model output logits (without sigmoid).
                labels: Labels, 1 for positive, 0 for unlabeled.

            Returns:
                Loss value.
            """
            # Compute losses for positive and unlabeled samples
            positive_mask = labels == 1
            unlabeled_mask = labels == 0

            # Compute probabilities via sigmoid
            pos_probs = torch.sigmoid(logits)

            # Positive loss: -log(σ(x))
            positive_loss = (
                -torch.log(pos_probs[positive_mask] + 1e-10).mean()
                if positive_mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

            # Unlabeled loss decomposed into positive and negative parts
            # Positive part: π_p * (-log(1 - σ(x)))
            # Negative part: -log(1 - σ(x))
            unlabeled_pos_loss = (
                -torch.log(1 - pos_probs[unlabeled_mask] + 1e-10).mean()
                if unlabeled_mask.any()
                else torch.tensor(0.0, device=logits.device)
            )

            # Risk estimation
            positive_risk = self.class_prior * positive_loss
            negative_risk = unlabeled_pos_loss - self.class_prior * unlabeled_pos_loss

            if self.loss_type == "nnpu":
                # nnPU: non-negative constraint
                if negative_risk < -self.beta:
                    loss = -self.gamma * negative_risk
                else:
                    loss = positive_risk + negative_risk
            else:
                # uPU: unbiased estimation
                loss = positive_risk + negative_risk

            return loss

    @PublicAPI(stability="beta")
    def nnpu_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_prior: float,
        beta: float = 0.0,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Functional interface for computing nnPU loss.

        Args:
            logits: Model output logits.
            labels: Labels, 1 for positive, 0 for unlabeled.
            class_prior: Positive class proportion.
            beta: Non-negative constraint threshold.
            gamma: Negative risk scaling factor.

        Returns:
            Loss value.
        """
        criterion = PULoss(
            class_prior=class_prior,
            beta=beta,
            gamma=gamma,
            loss_type="nnpu",
        )
        return criterion(logits, labels)

    def compute_class_prior(
        positive_count: int,
        total_count: int,
        method: str = "simple",
    ) -> float:
        """Estimate class prior.

        Args:
            positive_count: Number of positive examples.
            total_count: Total number of samples.
            method: Estimation method, 'simple' means positive/total.

        Returns:
            Estimated class prior value.

        Raises:
            ValueError: If parameters are invalid.
        """
        if positive_count < 0:
            raise ValueError(
                f"positive_count must be non-negative, got {positive_count}"
            )
        if total_count <= 0:
            raise ValueError(f"total_count must be positive, got {total_count}")
        if positive_count > total_count:
            raise ValueError(
                f"positive_count ({positive_count}) cannot exceed "
                f"total_count ({total_count})"
            )

        if method == "simple":
            return positive_count / total_count
        else:
            raise ValueError(f"Unknown method: {method}")

else:
    # Placeholder when PyTorch is not installed
    class PULoss:  # type: ignore[no-redef]
        """Placeholder class used when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required for PULoss. Install with: pip install torch"
            )

    def nnpu_loss(*args: Any, **kwargs: Any) -> Any:
        """Placeholder function used when PyTorch is not installed."""
        raise ImportError(
            "PyTorch is required for nnpu_loss. Install with: pip install torch"
        )

    def compute_class_prior(
        positive_count: int,
        total_count: int,
        method: str = "simple",
    ) -> float:
        """Estimate class prior (pure NumPy implementation)."""
        if positive_count < 0:
            raise ValueError(
                f"positive_count must be non-negative, got {positive_count}"
            )
        if total_count <= 0:
            raise ValueError(f"total_count must be positive, got {total_count}")
        if positive_count > total_count:
            raise ValueError(
                f"positive_count ({positive_count}) cannot exceed "
                f"total_count ({total_count})"
            )
        return positive_count / total_count
