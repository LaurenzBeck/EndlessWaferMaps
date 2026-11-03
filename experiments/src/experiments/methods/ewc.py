"""Elastic weight consolidation for a shared multi-label model."""

from dataclasses import dataclass, field

import torch

from . import Method, TaskContext


@dataclass
class EWC(Method):
    """Penalize changes to parameters with high diagonal Fisher information."""

    name: str = "ewc"
    regularization_strength: float = 5000.0
    fisher_merge_alpha: float = 0.5
    fisher_sampling: str = "true"
    fisher_num_samples: int = -1
    fisher: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    previous_parameters: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def forward(self, task_context: TaskContext, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return classification loss with the consolidated Fisher penalty."""
        logits = task_context.model(images)
        loss = task_context.loss_function(logits, targets)
        if task_context.task == 0 or not self.fisher:
            return loss
        penalty = torch.zeros((), device=task_context.device)
        for name, parameter in task_context.model.named_parameters():
            if name in self.fisher:
                penalty = penalty + torch.sum(self.fisher[name] * (parameter - self.previous_parameters[name]).pow(2))
        return loss + 0.5 * float(self.regularization_strength) * penalty

    def after_task(self, context: TaskContext) -> None:
        """Estimate and merge Fisher information, then snapshot parameters."""
        current_fisher = self._compute_fisher(context)
        if not self.fisher:
            self.fisher = current_fisher
        else:
            alpha = float(self.fisher_merge_alpha)
            self.fisher = {name: alpha * self.fisher[name] + (1.0 - alpha) * current_fisher[name] for name in current_fisher}
        self.previous_parameters = {
            name: parameter.detach().clone() for name, parameter in context.model.named_parameters() if parameter.requires_grad
        }

    def _compute_fisher(self, context: TaskContext) -> dict[str, torch.Tensor]:
        parameters = {name: parameter for name, parameter in context.model.named_parameters() if parameter.requires_grad}
        fisher = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
        loader = context.build_train_loader(context.stream.train[context.task])
        limit = int(self.fisher_num_samples)
        observed = 0
        was_training = context.model.training
        context.model.eval()
        for batch_images, batch_targets in loader:
            for batch_image, batch_target in zip(batch_images, batch_targets, strict=True):
                if limit > 0 and observed >= limit:
                    break
                image = context.augmentation(batch_image.unsqueeze(0).to(context.device))
                target = batch_target.unsqueeze(0).to(context.device, dtype=torch.float32)
                logits = context.model(image)
                if self.fisher_sampling == "true":
                    fisher_target = target
                elif self.fisher_sampling == "bernoulli":
                    fisher_target = torch.bernoulli(torch.sigmoid(logits.detach()))
                else:
                    msg = "fisher_sampling must be 'true' or 'bernoulli'."
                    raise ValueError(msg)
                loss = context.loss_function(logits, fisher_target)
                gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
                for (name, _), gradient in zip(parameters.items(), gradients, strict=True):
                    if gradient is not None:
                        fisher[name] += gradient.detach().pow(2)
                observed += 1
            if limit > 0 and observed >= limit:
                break
        context.model.train(was_training)
        if observed == 0:
            msg = "Cannot estimate Fisher information from an empty task dataset."
            raise ValueError(msg)
        return {name: value / observed for name, value in fisher.items()}
