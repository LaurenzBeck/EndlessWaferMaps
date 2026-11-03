"""Training methods for stream-based experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from experiments.data import DataStream


@dataclass
class TaskContext:
    """Task-level inputs shared while one stream task is prepared, trained, and evaluated."""

    model: torch.nn.Module
    stream: DataStream
    device: torch.device
    build_train_loader: Callable[[], DataLoader]
    build_val_loader: Callable[[], DataLoader]
    augmentation: torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]
    loss_function: torch.nn.Module
    task: int = 0
    step: int = 0


@dataclass
class Method:
    """Base class for training methods with optional state across tasks."""

    name: str

    def before_training(self, context: TaskContext) -> None:
        pass

    def before_task(self, context: TaskContext) -> None:
        pass

    def training_updates_per_epoch(self, context: TaskContext, train_loader: DataLoader) -> int:
        """Return the optimizer updates expected in one training epoch."""
        return len(train_loader)

    def forward(
        self,
        task_context: TaskContext,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return the loss tensor."""
        preds = task_context.model(images)
        return task_context.loss_function(preds, targets)

    def training_step(
        self,
        task_context: TaskContext,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> list[torch.Tensor]:
        """Run one or more optimizer updates for an input batch."""
        optimizer.zero_grad(set_to_none=True)
        loss = self.forward(task_context, images, targets)
        loss.backward()
        optimizer.step()
        return [loss.detach()]

    def after_task(self, context: TaskContext) -> None:
        """Save any method state after the task finishes."""
        pass


from .ewc import EWC as EWC  # noqa: E402
from .finetuning import Finetuning as Finetuning  # noqa: E402
from .joint import Joint as Joint  # noqa: E402
from .lwf import LwF as LwF  # noqa: E402
from .pass_method import PASS as PASS  # noqa: E402
from .path_integral import PathIntegral as PathIntegral  # noqa: E402
from .path_integral import PathIntegralReplay as PathIntegralReplay  # noqa: E402
from .remind import REMIND as REMIND  # noqa: E402
from .replay import Replay as Replay  # noqa: E402
from .siesta import SIESTA as SIESTA  # noqa: E402

__all__ = [
    "EWC",
    "PASS",
    "REMIND",
    "SIESTA",
    "Finetuning",
    "Joint",
    "LwF",
    "Method",
    "PathIntegral",
    "PathIntegralReplay",
    "Replay",
    "TaskContext",
]
