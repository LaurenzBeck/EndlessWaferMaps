"""Learning without forgetting for shared multi-label logits."""

from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from . import Method, TaskContext


@dataclass
class LwF(Method):
    """Distill prior-task Bernoulli predictions while learning new data."""

    name: str = "lwf"
    distillation_strength: float = 1.0
    temperature: float = 2.0
    teacher: torch.nn.Module | None = field(default=None, init=False, repr=False)

    def forward(self, task_context: TaskContext, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return classification loss with temperature-scaled distillation."""
        logits = task_context.model(images)
        loss = task_context.loss_function(logits, targets)
        if self.teacher is None:
            return loss
        temperature = float(self.temperature)
        with torch.no_grad():
            teacher_targets = torch.sigmoid(self.teacher(images) / temperature)
        distillation = binary_cross_entropy_with_logits(logits / temperature, teacher_targets)
        return loss + float(self.distillation_strength) * temperature**2 * distillation

    def after_task(self, context: TaskContext) -> None:
        """Freeze a copy of the current model as the next-task teacher."""
        self.teacher = deepcopy(context.model).to(context.device).eval()
        self.teacher.requires_grad_(requires_grad=False)
