"""nnPU (non-negative Positive-Unlabeled) loss function implementation.

Reference: "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
https://arxiv.org/abs/1703.00593

Implementation based on:
- https://github.com/cimeister/pu-learning
- https://github.com/kiryor/nnPUlearning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

from tributo.util.annotations import PublicAPI  # noqa: E402

if HAS_TORCH:

    def _validate_pu_labels(
        positive_mask: torch.Tensor,
        unlabeled_mask: torch.Tensor,
        *,
        require_both: bool,
    ) -> None:
        """Validate the binary PU label contract for one tensor batch."""
        if not bool(torch.all(positive_mask | unlabeled_mask)):
            raise ValueError(
                "PU labels must contain only 1 (positive) or 0 (unlabeled)"
            )
        if require_both and (
            not bool(positive_mask.any()) or not bool(unlabeled_mask.any())
        ):
            raise ValueError(
                "PU optimization requires every batch to contain both positive "
                "and unlabeled examples"
            )

    @dataclass
    class PURiskAccumulator:
        """Constant-memory accumulator for a split-level PU empirical risk."""

        class_prior: float
        loss_type: str
        positive_loss_sum: float = 0.0
        positive_as_negative_loss_sum: float = 0.0
        unlabeled_negative_loss_sum: float = 0.0
        positive_count: int = 0
        unlabeled_count: int = 0

        def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
            """Accumulate loss sums without requiring both groups in each batch."""
            positive_mask = labels == 1
            unlabeled_mask = labels == 0
            _validate_pu_labels(
                positive_mask,
                unlabeled_mask,
                require_both=False,
            )

            positive_count = int(positive_mask.sum().item())
            unlabeled_count = int(unlabeled_mask.sum().item())
            if positive_count:
                self.positive_loss_sum += float(
                    F.softplus(-logits[positive_mask]).sum().item()
                )
                self.positive_as_negative_loss_sum += float(
                    F.softplus(logits[positive_mask]).sum().item()
                )
                self.positive_count += positive_count
            if unlabeled_count:
                self.unlabeled_negative_loss_sum += float(
                    F.softplus(logits[unlabeled_mask]).sum().item()
                )
                self.unlabeled_count += unlabeled_count

        def value(self) -> float:
            """Return the uPU or Eq. 6 nnPU risk over all accumulated rows."""
            if self.positive_count == 0 or self.unlabeled_count == 0:
                raise ValueError(
                    "PU empirical risk requires both positive and unlabeled examples"
                )
            positive_risk = self.class_prior * (
                self.positive_loss_sum / self.positive_count
            )
            negative_risk = (
                self.unlabeled_negative_loss_sum / self.unlabeled_count
                - self.class_prior
                * self.positive_as_negative_loss_sum
                / self.positive_count
            )
            if self.loss_type == "nnpu":
                negative_risk = max(0.0, negative_risk)
            return positive_risk + negative_risk

    @PublicAPI(stability="beta")
    class PULoss(nn.Module):
        """nnPU loss function.

        Non-negative risk estimator for Positive-Unlabeled learning scenarios.

        Attributes:
            class_prior: Positive-class prior P(Y=1) in the population
                distribution (π_p).
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
                class_prior: Positive-class prior P(Y=1) in the population
                    distribution, range (0, 1).
                beta: Non-negative constraint threshold.
                gamma: Negative risk scaling factor.
                loss_type: 'nnpu' (non-negative) or 'upu' (unbiased).

            Raises:
                ValueError: If an argument is outside the nnPU contract.
            """
            super().__init__()
            if not 0 < class_prior < 1:
                raise ValueError(f"class_prior must be in (0, 1), got {class_prior}")
            if beta < 0:
                raise ValueError(f"beta must be non-negative, got {beta}")
            if not 0 <= gamma <= 1:
                raise ValueError(f"gamma must be in [0, 1], got {gamma}")
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
                Optimization surrogate whose backward pass follows Algorithm 1.

            Raises:
                ValueError: If labels are not binary PU labels or the batch does
                    not contain both positive and unlabeled examples.
            """
            positive_mask = labels == 1
            unlabeled_mask = labels == 0
            _validate_pu_labels(
                positive_mask,
                unlabeled_mask,
                require_both=True,
            )

            # Logistic losses in logit space avoid log(sigmoid(x)) overflow
            # for large positive or negative logits.
            positive_losses = F.softplus(-logits)
            negative_losses = F.softplus(logits)
            positive_loss = positive_losses[positive_mask].mean()
            positive_as_negative_loss = negative_losses[positive_mask].mean()
            unlabeled_negative_loss = negative_losses[unlabeled_mask].mean()

            positive_risk = self.class_prior * positive_loss
            negative_risk = (
                unlabeled_negative_loss - self.class_prior * positive_as_negative_loss
            )

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

        def empirical_risk(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            """Compute the reportable uPU or Eq. 6 nnPU empirical risk."""
            positive_mask = labels == 1
            unlabeled_mask = labels == 0
            _validate_pu_labels(
                positive_mask,
                unlabeled_mask,
                require_both=True,
            )
            positive_risk = self.class_prior * F.softplus(-logits[positive_mask]).mean()
            negative_risk = F.softplus(logits[unlabeled_mask]).mean() - (
                self.class_prior * F.softplus(logits[positive_mask]).mean()
            )
            if self.loss_type == "nnpu":
                negative_risk = torch.clamp(negative_risk, min=0.0)
            return positive_risk + negative_risk

        def new_risk_accumulator(self) -> PURiskAccumulator:
            """Create a split-level accumulator using this loss configuration."""
            return PURiskAccumulator(
                class_prior=self.class_prior,
                loss_type=self.loss_type,
            )

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
            class_prior: Positive-class prior P(Y=1) in the population distribution.
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
        loss: torch.Tensor = criterion(logits, labels)
        return loss

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
