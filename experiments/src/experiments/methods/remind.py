"""REMIND online latent replay for timm image models."""

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from . import Method, TaskContext
from .features import TimmFeatureAdapter
from .latent_memory import LatentMemory


@dataclass
class REMIND(Method):
    """Train the first task on images, then replay bounded spatial features."""

    name: str = "remind"
    num_exemplars_per_class: int = 100
    replay_batch_size: int = 64
    seed: int = 42
    use_quantization: bool = False
    num_groups: int = 4
    nbits: int = 8
    mixup_alpha: float | None = 0.1
    adapter: TimmFeatureAdapter | None = field(default=None, init=False, repr=False)
    memory: LatentMemory = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.memory = LatentMemory(
            num_exemplars_per_class=self.num_exemplars_per_class,
            seed=self.seed,
            use_quantization=self.use_quantization,
            num_groups=self.num_groups,
            nbits=self.nbits,
        )

    def before_training(self, context: TaskContext) -> None:
        self.adapter = TimmFeatureAdapter(context.model)

    def training_updates_per_epoch(self, context: TaskContext, train_loader: DataLoader) -> int:
        """Count one update per image after the base task."""
        return len(train_loader) if context.task == 0 else len(train_loader.dataset)

    def training_step(
        self,
        task_context: TaskContext,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> list[torch.Tensor]:
        """Use image training for task zero and online latent replay afterward."""
        if task_context.task == 0:
            return super().training_step(task_context, images, targets, optimizer)
        assert self.adapter is not None
        self.adapter.keep_backbone_eval()
        with torch.no_grad():
            current_features = self.adapter.spatial(images)
        losses: list[torch.Tensor] = []
        for feature, target in zip(current_features, targets, strict=True):
            replay_count = max(0, int(self.replay_batch_size) - 1)
            if replay_count:
                replay_features, replay_targets = self.memory.sample(replay_count, task_context.device)
                latent_batch = torch.cat([feature.unsqueeze(0), replay_features])
                target_batch = torch.cat([target.unsqueeze(0), replay_targets])
            else:
                latent_batch = feature.unsqueeze(0)
                target_batch = target.unsqueeze(0)
            if self.mixup_alpha is not None and len(latent_batch) > 1:
                beta = torch.distributions.Beta(float(self.mixup_alpha), float(self.mixup_alpha))
                ratio = beta.sample().to(task_context.device)
                permutation = torch.randperm(len(latent_batch), device=task_context.device)
                mixed = ratio * latent_batch + (1 - ratio) * latent_batch[permutation]
                logits = self.adapter.logits(mixed)
                loss = ratio * task_context.loss_function(logits, target_batch) + (1 - ratio) * task_context.loss_function(
                    logits,
                    target_batch[permutation],
                )
            else:
                loss = task_context.loss_function(self.adapter.logits(latent_batch), target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            self.memory.update(feature.unsqueeze(0), target.unsqueeze(0))
        return losses

    def after_task(self, context: TaskContext) -> None:
        """Initialize memory after task zero and keep updating it online later."""
        if context.task != 0:
            return
        assert self.adapter is not None
        features, targets = self._extract_task_features(context)
        self.memory.fit_quantizer(features)
        self.memory.update(features, targets)
        self.adapter.freeze_backbone()

    def _extract_task_features(self, context: TaskContext) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.adapter is not None
        loader = context.build_train_loader(context.stream.train[context.task])
        spatial_batches: list[torch.Tensor] = []
        target_batches: list[torch.Tensor] = []
        was_training = context.model.training
        context.model.eval()
        with torch.no_grad():
            for batch_images, batch_targets in loader:
                images = context.augmentation(batch_images.to(context.device))
                spatial_batches.append(self.adapter.spatial(images).cpu())
                target_batches.append(batch_targets.cpu())
        context.model.train(was_training)
        return torch.cat(spatial_batches), torch.cat(target_batches)
