"""Joint training method."""

from dataclasses import dataclass, field

from torch.utils.data import ConcatDataset, Dataset

from . import Method, TaskContext


@dataclass
class Joint(Method):
    """Train each task on all task datasets seen so far."""

    name: str = "joint"
    _original_train_stream: list[Dataset] = field(default_factory=list, init=False, repr=False)

    def before_training(self, context: TaskContext) -> None:
        """Accumulate datasets from all previous tasks."""
        if not self._original_train_stream:
            self._original_train_stream = list(context.stream.train)

        for task in range(len(self._original_train_stream)):
            datasets = self._original_train_stream[: task + 1]
            context.stream.train[task] = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
