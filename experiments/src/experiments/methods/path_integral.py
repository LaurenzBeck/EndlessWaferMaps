"""Path-integral synaptic intelligence, with optional exemplar replay."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import Method, TaskContext
from .replay import ExemplarDataset


@dataclass
class PathIntegral(Method):
    """Consolidate parameters according to their contribution to loss reduction."""

    name: str = "path_integral"
    regularization_strength: float = 0.1
    damping: float = 0.1
    importance: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    path_accumulator: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    previous_parameters: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def before_training(self, context: TaskContext) -> None:
        """Initialize parameter snapshots and path state."""
        self.previous_parameters = {
            name: parameter.detach().clone() for name, parameter in context.model.named_parameters() if parameter.requires_grad
        }
        self.importance = {name: torch.zeros_like(value) for name, value in self.previous_parameters.items()}
        self.path_accumulator = {name: torch.zeros_like(value) for name, value in self.previous_parameters.items()}

    def forward(self, task_context: TaskContext, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return classification loss with the consolidated SI penalty."""
        logits = task_context.model(images)
        loss = task_context.loss_function(logits, targets)
        if task_context.task == 0:
            return loss
        penalty = torch.zeros((), device=task_context.device)
        for name, parameter in task_context.model.named_parameters():
            if name in self.importance:
                penalty = penalty + torch.sum(self.importance[name] * (parameter - self.previous_parameters[name]).pow(2))
        return loss + float(self.regularization_strength) * penalty

    def training_step(
        self,
        task_context: TaskContext,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> list[torch.Tensor]:
        """Update parameters and accumulate their loss-reduction contribution."""
        parameters = {name: parameter for name, parameter in task_context.model.named_parameters() if parameter.requires_grad}
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        logits = task_context.model(images)
        task_loss = task_context.loss_function(logits, targets)
        unregularized_gradients = torch.autograd.grad(task_loss, tuple(parameters.values()), retain_graph=True, allow_unused=True)
        total_loss = task_loss
        if task_context.task > 0:
            penalty = torch.zeros((), device=task_context.device)
            for name, parameter in parameters.items():
                penalty = penalty + torch.sum(self.importance[name] * (parameter - self.previous_parameters[name]).pow(2))
            total_loss = total_loss + float(self.regularization_strength) * penalty
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            for (name, parameter), gradient in zip(parameters.items(), unregularized_gradients, strict=True):
                if gradient is not None:
                    self.path_accumulator[name] -= gradient.detach() * (parameter.detach() - before[name])
        return [total_loss.detach()]

    def after_task(self, context: TaskContext) -> None:
        """Consolidate non-negative importance and reset the path state."""
        current = {name: parameter.detach().clone() for name, parameter in context.model.named_parameters() if parameter.requires_grad}
        with torch.no_grad():
            for name in self.importance:
                contribution = self.path_accumulator[name] / ((current[name] - self.previous_parameters[name]).pow(2) + float(self.damping))
                self.importance[name].add_(contribution).clamp_(min=0.0)
                self.path_accumulator[name].zero_()
        self.previous_parameters = current


@dataclass
class PathIntegralReplay(PathIntegral):
    """Path-integral consolidation trained with class-balanced image replay."""

    name: str = "path_integral_replay"
    num_exemplars_per_class: int = 100
    seed: int = 42
    exemplars: ExemplarDataset = field(init=False)
    _build_train_loader: Callable[[Dataset], DataLoader] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.exemplars = ExemplarDataset(self.num_exemplars_per_class, self.seed)

    def before_training(self, context: TaskContext) -> None:
        """Initialize SI and wrap task loaders with exemplar replay."""
        super().before_training(context)
        build_original = context.build_train_loader
        self._build_train_loader = build_original

        def build_train_loader(dataset: Dataset) -> DataLoader:
            combined = dataset if len(self.exemplars) == 0 else ConcatDataset([dataset, self.exemplars])
            return build_original(combined)

        context.build_train_loader = build_train_loader

    def after_task(self, context: TaskContext) -> None:
        """Consolidate SI state and update exemplar memory."""
        super().after_task(context)
        self.exemplars.update(context.stream.train[context.task])
