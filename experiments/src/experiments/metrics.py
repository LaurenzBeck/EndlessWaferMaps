"""Extra metrics used by the experiment loops."""

from __future__ import annotations

from typing import Literal

import torch
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix


class MultilabelConfusionMatrix(torchmetrics.Metric):
    """Multi-label confusion matrix used for experiment visualizations."""

    is_differentiable = False
    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        """Constructor method.

        Args:
            num_classes: Number of labels in the problem.
        """
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "cmat",
            default=torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int32),
            dist_reduce_fx="sum",
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the confusion matrix from one batch."""
        if predictions.dtype == torch.float32:
            predictions = (predictions > 0).int()

        targets = targets.int()
        predictions = predictions.int()

        num_classes = predictions.shape[1]
        assert num_classes == self.num_classes
        ntl = npl = num_classes

        T_i1 = torch.logical_and(targets, predictions)
        T_i2 = torch.logical_and(targets, torch.logical_not(predictions))
        T_empty = targets.sum(1) == 0
        P_i2 = torch.logical_and(torch.logical_not(targets), predictions)
        P_empty = predictions.sum(1) == 0

        self.cmat[:num_classes, :num_classes] += torch.diag(T_i1.sum(0))
        self.cmat[ntl, npl] += torch.logical_and(T_empty, P_empty).sum()

        self.cmat[:num_classes, npl] += T_i2.sum(0)

        self.cmat[:num_classes, :num_classes] += torch.logical_and(targets.unsqueeze(1), P_i2.unsqueeze(2).int()).sum(0)
        self.cmat[ntl, :num_classes] += P_i2[T_empty].sum(0)

        self.cmat[:num_classes, :num_classes] += torch.logical_and(T_i2.unsqueeze(1), P_i2.unsqueeze(2)).sum(0)

    def compute(self) -> torch.Tensor:
        """Computes the metric at the end of the epoch.

        Returns:
            torch.Tensor: Multi-label confusion matrix.
        """
        return self.cmat


ConfusionMatrixNormalization = Literal["none", "true", "pred", "all"]


class CardinalityConfusionMatrix(MulticlassConfusionMatrix):
    """Confusion matrix over the number of positive labels in each sample."""

    def __init__(
        self,
        max_cardinality: int,
        normalize: ConfusionMatrixNormalization = "none",
        threshold: float = 0.5,
    ) -> None:
        """A cardinality confusion matrix.

        The `update` method expects multi-label preds and targets.

        Args:
            max_cardinality: Largest allowed positive-label count.
            normalize: Normalization mode passed to torchmetrics.
            threshold: Threshold used to turn probabilities into binary predictions.
        """
        super().__init__(num_classes=max_cardinality + 1, normalize=normalize)
        self.threshold = threshold

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the cardinality confusion matrix from one batch."""
        predicted_cardinality = torch.sum(predictions.sigmoid() > self.threshold, dim=1)
        target_cardinality = targets.sum(dim=1).int()
        return super().update(predicted_cardinality, target_cardinality)


class MeanMagnitude(torchmetrics.MeanMetric):
    """Mean vector magnitude across a batch of features."""

    is_differentiable = False
    higher_is_better = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, features: torch.Tensor) -> None:
        """Update the running mean from one feature batch."""
        magnitudes = torch.linalg.vector_norm(features, dim=1)
        super().update(magnitudes)


__all__ = [
    "CardinalityConfusionMatrix",
    "ConfusionMatrixNormalization",
    "MeanMagnitude",
    "MultilabelConfusionMatrix",
]
