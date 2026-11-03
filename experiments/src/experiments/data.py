"""🌊 Dataset and stream builders for offline and continual-learning experiments."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset
import torch
from torchvision.transforms import v2

from endlesswm.io.datasets import H5WM


@dataclass(slots=True)
class DataStream:
    train: list[Dataset]
    test: dict[str, list[Dataset | None]]
    class_names: list[str]


def evaluation_output_path(dataset_path: Path) -> Path:
    """Return the paired evaluation dataset path for a generated dataset."""
    return dataset_path.with_name(f"{dataset_path.stem}-t{dataset_path.suffix}")


def split_dataset(dataset: Dataset, train_fraction: float = 0.8, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Split one dataset into train and validation subsets."""
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, train_size=train_fraction, random_state=seed)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def take_dataset_fraction(dataset: Dataset, fraction: float = 1.0, seed: int = 42) -> Dataset:
    """Keep only a fraction of a dataset, mainly for quick development runs."""
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("Dataset fraction must be in the interval (0, 1].")

    indices = list(range(len(dataset)))
    chosen_indices, _ = train_test_split(indices, train_size=float(fraction), random_state=seed)
    return Subset(dataset, chosen_indices)


def build_temporal_test_streams(present_datasets: list[Dataset]) -> dict[str, list[Dataset | None]]:
    task_count = len(present_datasets)
    test_streams: dict[str, list[Dataset | None]] = {
        "present": present_datasets,
        "past": [None] * task_count,
        "future": [None] * task_count,
    }
    for task_index in range(task_count):
        if task_index > 0:
            test_streams["past"][task_index] = ConcatDataset(present_datasets[:task_index])
        if task_index + 1 < task_count:
            test_streams["future"][task_index] = ConcatDataset(present_datasets[task_index + 1 :])
    return test_streams


def build_single_task_stream(
    train_dataset: Dataset,
    class_names: list[str],
    val_dataset: Dataset | None = None,
    test_datasets: Mapping[str, Dataset] | None = None,
    train_val_split: float | None = None,
    data_fraction: float | None = None,
    seed: int = 42,
) -> DataStream:
    if val_dataset:
        val_stream = [val_dataset]
    elif train_val_split:
        train_dataset, val_dataset = split_dataset(train_dataset, train_fraction=train_val_split, seed=seed)
        val_stream = [val_dataset]
    else:
        val_stream = [train_dataset]

    if data_fraction is not None:
        train_dataset = take_dataset_fraction(train_dataset, fraction=data_fraction, seed=seed)

    test_streams = {str(name): [dataset] for name, dataset in (test_datasets or {}).items()} | {
        "present": val_stream,
        "train": [train_dataset],
    }

    return DataStream(
        train=[train_dataset],
        test=test_streams,
        class_names=list(class_names),
    )


def build_stream(
    train_datasets: list[Dataset],
    class_names: list[str],
    val_datasets: list[Dataset] | None = None,
    test_datasets: Mapping[str, Dataset | list[Dataset]] | None = None,
    train_val_split: float | None = None,
    data_fraction: float | None = None,
    seed: int = 42,
    add_temporal_streams: bool = True,
) -> DataStream:
    train_stream = train_datasets
    if val_datasets:
        val_stream = val_datasets
    elif train_val_split:
        train_stream = []
        val_stream = []
        for task_index, dataset in enumerate(train_datasets):
            train_split, val_split = split_dataset(dataset, train_fraction=train_val_split, seed=seed + task_index)
            train_stream.append(train_split)
            val_stream.append(val_split)
    else:
        val_stream = train_stream

    test_streams = build_temporal_test_streams(val_stream) if add_temporal_streams else {"present": val_stream}
    for name, dataset in (test_datasets or {}).items():
        if isinstance(dataset, list):
            test_streams[str(name)] = dataset
        else:
            test_streams[str(name)] = [dataset for _ in train_stream]

    if data_fraction is not None:
        train_stream = [
            take_dataset_fraction(dataset, fraction=data_fraction, seed=seed + task_index)
            for task_index, dataset in enumerate(train_stream)
        ]

    test_streams["train"] = list(train_stream)

    return DataStream(
        train=train_stream,
        test=test_streams,
        class_names=list(class_names),
    )


class FillNaN(v2.Transform):
    """Replace NaNs in wafer maps before later tensor transforms."""

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = float(value)

    def _transform(self, x: object, _params: object) -> object:
        if isinstance(x, torch.Tensor):
            return torch.nan_to_num(x, nan=self.value)
        return x


def build_endlesswm_single_task_stream(
    train_dataset: str,
    class_names: list[str],
    use_val_dataset: bool = False,
    test_datasets: Mapping[str, str] | None = None,
    task_index: int | None = None,
    image_size: int = 64,
    train_val_split: float | None = None,
    data_fraction: float | None = None,
    seed: int = 42,
) -> DataStream:
    train_path = Path("../../../../" + train_dataset)
    transforms = {"sample": v2.Compose([FillNaN(), v2.Resize((image_size, image_size))])}
    train_dataset = H5WM(str(train_path), task=task_index, transform=transforms)

    if use_val_dataset:
        val_path = evaluation_output_path(train_path)
        val_dataset = H5WM(str(val_path), task=task_index, transform=transforms)

    test_datasets = {
        name: H5WM("../../../../" + path, task=task_index, transform=transforms) for name, path in (test_datasets or {}).items()
    }

    return build_single_task_stream(
        train_dataset=train_dataset,
        class_names=class_names,
        val_dataset=val_dataset if use_val_dataset else None,
        test_datasets=test_datasets,
        train_val_split=train_val_split,
        data_fraction=data_fraction,
        seed=seed,
    )


def build_endlesswm_segmentation_stream(
    train_dataset: str,
    class_names: list[str],
    test_dataset: str,
    task_index: int = 0,
    data_fraction: float | None = None,
    seed: int = 42,
) -> DataStream:
    """Build a single-task stream with pixel-wise segmentation masks."""
    transforms = {"sample": FillNaN()}
    train_stream = H5WM(
        "../../../../" + train_dataset,
        task=task_index,
        transform=transforms,
        fields=("sample", "masks"),
    )
    test_stream = H5WM(
        "../../../../" + test_dataset,
        task=task_index,
        transform=transforms,
        fields=("sample", "masks"),
    )
    return build_single_task_stream(
        train_dataset=train_stream,
        class_names=class_names,
        test_datasets={"present": test_stream},
        data_fraction=data_fraction,
        seed=seed,
    )


def build_endlesswm_stream(
    train_dataset: str,
    class_names: list[str],
    use_val_dataset: bool = False,
    test_datasets: Mapping[str, str] | None = None,
    image_size: int = 64,
    train_val_split: float | None = None,
    data_fraction: float | None = None,
    seed: int = 42,
    add_temporal_streams: bool = True,
    task_repetitions: int = 1,
) -> DataStream:
    train_path = Path("../../../../" + train_dataset)
    transforms = {"sample": v2.Compose([FillNaN(), v2.Resize((image_size, image_size))])}
    train_datasets = list(H5WM(str(train_path), transform=transforms).tasks)
    if int(task_repetitions) < 1:
        msg = "task_repetitions must be at least 1."
        raise ValueError(msg)
    train_datasets *= int(task_repetitions)

    if use_val_dataset:
        val_path = evaluation_output_path(train_path)
        val_datasets = list(H5WM(str(val_path), transform=transforms).tasks)
        val_datasets *= int(task_repetitions)

    test_datasets = {name: H5WM("../../../../" + path, transform=transforms) for name, path in (test_datasets or {}).items()}

    return build_stream(
        train_datasets=train_datasets,
        class_names=class_names,
        val_datasets=val_datasets if use_val_dataset else None,
        test_datasets=test_datasets,
        train_val_split=train_val_split,
        data_fraction=data_fraction,
        seed=seed,
        add_temporal_streams=add_temporal_streams,
    )


class MixedWM38(Dataset):
    """In-memory MixedWM38 wafer-map dataset loaded from the original NPZ.

    The NPZ shipped by Wang et al. contains two arrays: ``arr_0`` with samples
    of shape ``(N, 52, 52)`` (integer 0/1/2 wafer maps) and ``arr_1`` with
    multi-label targets of shape ``(N, 8)``. Samples are returned with an
    explicit channel dimension to match the wafer-map convention used by the
    EndlessWM training pipeline.
    """

    def __init__(self, file_path: str, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        data = np.load(file_path)
        self.samples = torch.from_numpy(data["arr_0"]).to(torch.float32).unsqueeze(1)
        self.labels = torch.from_numpy(data["arr_1"]).to(torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[idx]


def build_mixedwm38_single_task_stream(
    class_names: list[str],
    dataset_path: str = "experiments/data/downloaded/mixedwm38/Wafer_Map_Datasets.npz",
    train_val_split: float | None = 0.8,
    data_fraction: float | None = None,
    seed: int = 42,
) -> DataStream:
    """Single-task MixedWM38 stream with a held-out validation split.

    The validation split (controlled by ``train_val_split``) becomes the
    ``present`` test stream, mirroring the random-scenario variant in the
    Cardinality-Incremental-Learning reference repo. Samples are kept at their
    native 52x52 resolution.
    """
    npz_path = Path("../../../../" + dataset_path)
    dataset = MixedWM38(str(npz_path))

    return build_single_task_stream(
        train_dataset=dataset,
        class_names=class_names,
        train_val_split=train_val_split,
        data_fraction=data_fraction,
        seed=seed,
    )
