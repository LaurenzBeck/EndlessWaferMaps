"""Exemplar replay method."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from loguru import logger as log
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import Method, TaskContext

Exemplar = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ExemplarDataset(Dataset):
    """Class-balanced exemplar memory with one reservoir per active label."""

    num_exemplars_per_class: int = 100
    seed: int = 42
    seen_counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    reservoirs: dict[int, list[Exemplar]] = field(default_factory=dict, init=False, repr=False)
    _exemplars: list[Exemplar] = field(default_factory=list, init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the deterministic reservoir sampler."""
        self._generator = torch.Generator().manual_seed(int(self.seed))

    def __len__(self) -> int:
        """Return the number of stored exemplars."""
        return len(self._exemplars)

    def __getitem__(self, index: int) -> Exemplar:
        """Return one stored exemplar."""
        return self._exemplars[index]

    def update(self, dataset: Dataset) -> None:
        """Update exemplar reservoirs from one task dataset."""
        k = int(self.num_exemplars_per_class)
        if k <= 0:
            self._exemplars = []
            return

        for index in range(len(dataset)):
            sample, target = dataset[index]
            image = sample.detach().clone()
            labels = target.detach().clone()
            for class_index in torch.nonzero(labels > 0, as_tuple=False).flatten().tolist():
                self._update_class(int(class_index), image, labels)

        reservoir_sizes = {class_index: len(self.reservoirs[class_index]) for class_index in sorted(self.reservoirs)}
        log.info(f"exemplar reservoirs updated: {reservoir_sizes}")

        self._exemplars = [exemplar for class_index in sorted(self.reservoirs) for exemplar in self.reservoirs[class_index]]

    def _update_class(self, class_index: int, image: torch.Tensor, labels: torch.Tensor) -> None:
        k = int(self.num_exemplars_per_class)
        seen_count = self.seen_counts.get(class_index, 0) + 1
        self.seen_counts[class_index] = seen_count

        bucket = self.reservoirs.setdefault(class_index, [])
        exemplar = (image.clone(), labels.clone())
        if len(bucket) < k:
            bucket.append(exemplar)
            return

        replacement_index = int(torch.randint(seen_count, (1,), generator=self._generator).item())
        if replacement_index < k:
            bucket[replacement_index] = exemplar


@dataclass
class Replay(Method):
    """Finetune on the current task plus class-balanced exemplar memory."""

    name: str = "replay"
    num_exemplars_per_class: int = 100
    seed: int = 42
    exemplars: ExemplarDataset = field(init=False)
    _build_train_loader: Callable[[Dataset], DataLoader] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the exemplar memory."""
        self.exemplars = ExemplarDataset(
            num_exemplars_per_class=int(self.num_exemplars_per_class),
            seed=int(self.seed),
        )

    def before_training(self, context: TaskContext) -> None:
        """Wrap train-loader construction to include exemplar memory."""
        if self._build_train_loader is None:
            self._build_train_loader = context.build_train_loader
        build_original_train_loader = self._build_train_loader

        def build_train_loader(dataset: Dataset) -> DataLoader:
            if len(self.exemplars) == 0:
                return build_original_train_loader(dataset)
            return build_original_train_loader(ConcatDataset([dataset, self.exemplars]))

        context.build_train_loader = build_train_loader

    def after_task(self, context: TaskContext) -> None:
        """Update exemplar memory from the just-finished task dataset."""
        self.exemplars.update(context.stream.train[context.task])
