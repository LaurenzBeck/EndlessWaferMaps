"""Asymmetric multi-label loss."""

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    """Asymmetric multi-label loss adapted from the ASL reference implementation."""

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ) -> None:
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss from logits and multi-label targets."""
        positive_probabilities = torch.sigmoid(predictions)
        negative_probabilities = 1 - positive_probabilities

        if self.clip is not None and self.clip > 0:
            negative_probabilities = (negative_probabilities + self.clip).clamp(max=1)

        positive_loss = targets * torch.log(positive_probabilities.clamp(min=self.eps))
        negative_loss = (1 - targets) * torch.log(negative_probabilities.clamp(min=self.eps))
        loss = positive_loss + negative_loss

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            positive_pt = positive_probabilities * targets
            negative_pt = negative_probabilities * (1 - targets)
            pt = positive_pt + negative_pt
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_weight = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_weight

        return -loss.mean()
