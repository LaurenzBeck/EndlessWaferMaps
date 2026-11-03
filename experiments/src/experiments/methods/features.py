"""Access to timm spatial features and classifiers."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TimmFeatureAdapter:
    """Expose the timm feature operations used by continual-learning methods."""

    model: nn.Module

    def __post_init__(self) -> None:
        required = ("forward_features", "forward_head", "get_classifier")
        missing = [name for name in required if not callable(getattr(self.model, name, None))]
        if missing:
            raise TypeError(f"Model does not provide the required timm feature API: {', '.join(missing)}")
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(f"Expected a linear timm classifier, got {type(self.classifier).__name__}.")

    @property
    def classifier(self) -> nn.Linear:
        return self.model.get_classifier()

    def spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Return NCHW spatial features."""
        return self.model.forward_features(images)

    def pooled(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Pool spatial features without applying the classifier."""
        return self.model.forward_head(spatial_features, pre_logits=True)

    def logits(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Forward spatial features through the timm head."""
        return self.model.forward_head(spatial_features)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Return pre-classifier feature vectors."""
        return self.pooled(self.spatial(images))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify pre-logit feature vectors."""
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Freeze every parameter except the final linear classifier."""
        for parameter in self.model.parameters():
            parameter.requires_grad_(requires_grad=False)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(requires_grad=True)

    def keep_backbone_eval(self) -> None:
        """Keep frozen feature-extractor normalization state fixed."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                module.eval()
