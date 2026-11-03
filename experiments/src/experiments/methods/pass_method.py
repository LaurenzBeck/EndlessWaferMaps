"""Prototype augmentation and feature distillation for multi-label streams."""

from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch.nn.functional import mse_loss

from . import Method, TaskContext
from .features import TimmFeatureAdapter


@dataclass
class PASS(Method):
    """Combine powerset prototypes with prior-model feature distillation."""

    name: str = "pass"
    prototype_augmentation_strength: float = 10.0
    distillation_strength: float = 10.0
    prototypes: dict[tuple[int, ...], torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    radius: float = field(default=0.0, init=False)
    adapter: TimmFeatureAdapter | None = field(default=None, init=False, repr=False)
    teacher: torch.nn.Module | None = field(default=None, init=False, repr=False)
    teacher_adapter: TimmFeatureAdapter | None = field(default=None, init=False, repr=False)

    def before_training(self, context: TaskContext) -> None:
        self.adapter = TimmFeatureAdapter(context.model)

    def forward(self, task_context: TaskContext, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return classification, feature-distillation, and prototype losses."""
        assert self.adapter is not None
        features = self.adapter.features(images)
        loss = task_context.loss_function(self.adapter.classify(features), targets)
        if self.teacher_adapter is None or not self.prototypes:
            return loss
        with torch.no_grad():
            old_features = self.teacher_adapter.features(images)
        distillation = mse_loss(features, old_features)

        keys = list(self.prototypes)
        selected = [keys[index % len(keys)] for index in range(targets.shape[0])]
        prototype_features = torch.stack([self.prototypes[key] for key in selected]).to(task_context.device)
        if self.radius > 0:
            prototype_features = prototype_features + torch.randn_like(prototype_features) * self.radius
        prototype_targets = torch.tensor(selected, dtype=targets.dtype, device=task_context.device)
        prototype_loss = task_context.loss_function(self.adapter.classify(prototype_features), prototype_targets)
        return loss + float(self.distillation_strength) * distillation + float(self.prototype_augmentation_strength) * prototype_loss

    def after_task(self, context: TaskContext) -> None:
        """Extract task prototypes and freeze a teacher for the next task."""
        assert self.adapter is not None
        loader = context.build_train_loader(context.stream.train[context.task])
        grouped: dict[tuple[int, ...], list[torch.Tensor]] = {}
        was_training = context.model.training
        context.model.eval()
        with torch.no_grad():
            for batch_images, batch_targets in loader:
                images = context.augmentation(batch_images.to(context.device))
                features = self.adapter.features(images).cpu()
                for feature, target in zip(features, batch_targets, strict=True):
                    key = tuple(int(value) for value in target.tolist())
                    grouped.setdefault(key, []).append(feature)
        context.model.train(was_training)
        task_prototypes = {key: torch.stack(values).mean(dim=0) for key, values in grouped.items()}
        if context.task == 0:
            variances = [torch.stack(values).var(dim=0, unbiased=False).mean() for values in grouped.values()]
            self.radius = float(torch.stack(variances).mean().clamp_min(0).sqrt()) if variances else 0.0
        self.prototypes.update(task_prototypes)
        self.teacher = deepcopy(context.model).to(context.device).eval().requires_grad_(requires_grad=False)
        self.teacher_adapter = TimmFeatureAdapter(self.teacher)
