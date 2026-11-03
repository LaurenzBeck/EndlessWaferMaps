"""SIESTA wake/sleep training on bounded spatial-feature memory."""

from dataclasses import dataclass, field

import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from . import TaskContext
from .remind import REMIND


@dataclass
class SIESTA(REMIND):
    """Update memory and class means while awake, then train from memory."""

    name: str = "siesta"
    num_updates_during_sleep: int = 100
    class_sums: torch.Tensor | None = field(default=None, init=False, repr=False)
    class_counts: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sleep_updates_per_batch: int = field(default=1, init=False, repr=False)

    def before_task(self, context: TaskContext) -> None:
        """Run the wake phase before later-task sleep training."""
        if context.task == 0:
            return
        assert self.adapter is not None
        features, targets = self._extract_task_features(context)
        self.memory.update(features, targets)
        self._update_class_means(self.adapter.pooled(features.to(context.device)), targets.to(context.device))
        self._copy_means_to_classifier()
        batches = max(1, len(context.build_train_loader(context.stream.train[context.task])))
        self._sleep_updates_per_batch = max(1, int(self.num_updates_during_sleep) // batches)

    def training_updates_per_epoch(self, context: TaskContext, train_loader: DataLoader) -> int:
        """Count base-task batches or later sleep updates."""
        return len(train_loader) if context.task == 0 else len(train_loader) * self._sleep_updates_per_batch

    def training_step(
        self,
        task_context: TaskContext,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> list[torch.Tensor]:
        """Train task zero on images and later tasks using memory-only sleep batches."""
        if task_context.task == 0:
            return super().training_step(task_context, images, targets, optimizer)
        assert self.adapter is not None
        self.adapter.keep_backbone_eval()
        losses: list[torch.Tensor] = []
        for _ in range(self._sleep_updates_per_batch):
            features, replay_targets = self.memory.sample(self.replay_batch_size, task_context.device)
            loss = task_context.loss_function(self.adapter.logits(features), replay_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        return losses

    def after_task(self, context: TaskContext) -> None:
        """Initialize task-zero memory and measured class means."""
        if context.task != 0:
            return
        super().after_task(context)
        assert self.adapter is not None
        features, targets = self._extract_task_features(context)
        self._update_class_means(self.adapter.pooled(features.to(context.device)), targets.to(context.device))
        self._copy_means_to_classifier()

    def _update_class_means(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        num_classes = targets.shape[1]
        if self.class_sums is None or self.class_counts is None:
            self.class_sums = torch.zeros(num_classes, features.shape[1], device=features.device)
            self.class_counts = torch.zeros(num_classes, device=features.device)
        for class_index in range(num_classes):
            selected = targets[:, class_index] > 0
            if selected.any():
                self.class_sums[class_index] += features[selected].sum(dim=0)
                self.class_counts[class_index] += selected.sum()

    def _copy_means_to_classifier(self) -> None:
        assert self.adapter is not None
        assert self.class_sums is not None and self.class_counts is not None
        valid = self.class_counts > 0
        means = self.class_sums[valid] / self.class_counts[valid, None]
        with torch.no_grad():
            self.adapter.classifier.weight[valid] = normalize(means, dim=1)
            if self.adapter.classifier.bias is not None:
                self.adapter.classifier.bias.zero_()
