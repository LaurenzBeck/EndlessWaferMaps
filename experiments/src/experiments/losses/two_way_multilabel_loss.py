"""Two-way multi-label loss."""

import torch
from torch import nn

NEGATIVE_INFINITY = -100


class TwoWayLoss(nn.Module):
    """Two-way multi-label loss adapted from the reference implementation."""

    def __init__(self, Tp: float = 4.0, Tn: float = 1.0) -> None:
        super().__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss from logits and multi-label targets."""
        class_mask = (targets > 0).any(dim=0)
        sample_mask = (targets > 0).any(dim=1)

        positive_mask = targets.masked_fill(targets <= 0, NEGATIVE_INFINITY).masked_fill(targets > 0, 0.0)
        positive_logits_by_class = torch.logsumexp(-predictions / self.Tp + positive_mask, dim=0).mul(self.Tp)[class_mask]
        positive_logits_by_sample = torch.logsumexp(-predictions / self.Tp + positive_mask, dim=1).mul(self.Tp)[sample_mask]

        negative_mask = targets.masked_fill(targets != 0, NEGATIVE_INFINITY).masked_fill(targets == 0, 0.0)
        negative_logits_by_class = torch.logsumexp(predictions / self.Tn + negative_mask, dim=0).mul(self.Tn)[class_mask]
        negative_logits_by_sample = torch.logsumexp(predictions / self.Tn + negative_mask, dim=1).mul(self.Tn)[sample_mask]

        return (
            torch.nn.functional.softplus(negative_logits_by_class + positive_logits_by_class).mean()
            + torch.nn.functional.softplus(negative_logits_by_sample + positive_logits_by_sample).mean()
        )
