"""HDF5 export routines for EndlessWM datasets."""

import json
import time
import warnings
from collections import defaultdict
from typing import Any

import h5py
import numpy as np
from streamgen.samplers.tree import SamplingTree
from tqdm.auto import tqdm

from ..geometry import full_ellipse
from ..patterns import classes as _default_classes
from ..utils import flatten_dict

CURRENT_VERSION = "2.1"
DEFAULT_AUTHOR = "Laurenz A. Farthofer"

COMPRESSION_SETTINGS = {
    "float": {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "fletcher32": False,
    },
    "uint8": {
        "compression": "lzf",
        "shuffle": False,
        "fletcher32": False,
    },
    "params": {
        "compression": "gzip",
        "compression_opts": 6,
        "shuffle": True,
        "fletcher32": False,
    },
}

POLARITY_TO_FLOAT = {
    "+": 1.0,
    "-": -1.0,
    "NaN": 0.0,
}


def encode_parameter_value(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return POLARITY_TO_FLOAT.get(value, np.nan)
    return float(value)


def get_chunk_shape(num_samples: int, num_channels: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Return chunk shape tuned for random single-sample task access."""
    return (1, num_channels, height, width)


def get_compression_settings(data_type: str) -> dict:
    return COMPRESSION_SETTINGS.get(data_type, COMPRESSION_SETTINGS["float"])


def filter_generation_parameters(params_dict: dict) -> dict:
    flattened = flatten_dict(params_dict)
    return {key: value for key, value in flattened.items() if not (key.endswith(".size") or key.endswith(".shape"))}


def compute_multilabel_from_masks(masks: np.ndarray) -> np.ndarray:
    return (masks != 0).any(axis=(1, 2)).astype(np.uint8)


def create_wafer_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(full_ellipse(shape), dtype=np.uint8)


def create_hdf5_metadata(
    file_handle: h5py.File,
    num_tasks: int,
    num_classes_count: int,
    class_names: list[str],
    seed: int | None = None,
    parallel: bool = False,
    generation_metadata: dict[str, Any] | None = None,
) -> None:
    file_handle.attrs["version"] = CURRENT_VERSION
    file_handle.attrs["schema_version"] = CURRENT_VERSION
    file_handle.attrs["created_at"] = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime())
    file_handle.attrs["author"] = DEFAULT_AUTHOR

    format_notes = "Improved format: (N,H,W) layout, DataFrame params, optimized chunks"
    if parallel:
        format_notes += ", parallel generation"
    file_handle.attrs["format_notes"] = format_notes

    if seed is not None:
        file_handle.attrs["seed"] = seed

    if parallel:
        file_handle.attrs["generation_mode"] = "parallel"

    meta = file_handle.create_group("meta")
    meta.attrs["schema_version"] = CURRENT_VERSION
    meta.attrs["num_classes"] = int(num_classes_count)
    meta.attrs["num_tasks"] = int(num_tasks)

    vlen_str = h5py.string_dtype(encoding="utf-8")
    classes_dataset = meta.create_dataset("classes", shape=(num_classes_count,), dtype=vlen_str)
    classes_dataset[...] = np.array(class_names, dtype=object)

    if generation_metadata:
        for attribute_name in (
            "preset_identifier",
            "preset_alias",
            "command_invocation",
            "resolved_output_path",
            "dataset_name",
        ):
            attribute_value = generation_metadata.get(attribute_name)
            if attribute_value is not None:
                meta.attrs[attribute_name] = str(attribute_value)

        generation_context_dataset = meta.create_dataset("generation_context_json", shape=(1,), dtype=vlen_str)
        generation_context_dataset[0] = json.dumps(generation_metadata, sort_keys=True)


def create_global_index(
    meta_group: h5py.Group,
    task_sample_counts: list[int],
) -> None:
    num_tasks = len(task_sample_counts)
    offsets = np.zeros(num_tasks + 1, dtype=np.uint64)
    for i, count in enumerate(task_sample_counts):
        offsets[i + 1] = offsets[i] + count

    total_samples = int(offsets[-1])
    compression = get_compression_settings("params")

    global_index = meta_group.create_dataset(
        "global_index",
        shape=(total_samples, 2),
        dtype=np.uint32,
        **compression,
    )

    position = 0
    for task_idx, sample_count in enumerate(task_sample_counts):
        rows = np.empty((sample_count, 2), dtype=np.uint32)
        rows[:, 0] = np.uint32(task_idx)
        rows[:, 1] = np.arange(sample_count, dtype=np.uint32)
        global_index[position : position + sample_count] = rows
        position += sample_count

    meta_group.create_dataset("task_offsets", data=offsets, dtype=np.uint64)


def create_image_dataset(
    group: h5py.Group,
    name: str,
    shape: tuple[int, int, int],
    data: np.ndarray | None = None,
) -> h5py.Dataset:
    num_samples, height, width = shape
    chunk_shape = get_chunk_shape(num_samples, 1, height, width)
    compression = get_compression_settings("float")

    dataset = group.create_dataset(
        name,
        shape=shape,
        data=data,
        dtype=np.float32,
        chunks=(chunk_shape[0], height, width),
        **compression,
    )
    dataset.attrs["layout"] = "NHW"
    return dataset


def create_masks_dataset(
    group: h5py.Group,
    shape: tuple[int, int, int, int],
    data: np.ndarray | None = None,
) -> h5py.Dataset:
    num_samples, num_channels, height, width = shape
    chunk_shape = get_chunk_shape(num_samples, num_channels, height, width)
    compression = get_compression_settings("uint8")

    return group.create_dataset(
        "masks",
        shape=shape,
        data=data,
        dtype=np.uint8,
        chunks=chunk_shape,
        **compression,
    )


def create_multilabel_dataset(
    group: h5py.Group,
    num_samples: int,
    num_classes_count: int,
    data: np.ndarray | None = None,
) -> h5py.Dataset:
    compression = get_compression_settings("uint8")
    chunk_size = min(512, num_samples)

    return group.create_dataset(
        "multilabel",
        shape=(num_samples, num_classes_count),
        data=data,
        dtype=np.uint8,
        chunks=(chunk_size, num_classes_count),
        **compression,
    )


def create_parameters_group(
    group: h5py.Group,
    parameters_dict: dict[str, list],
    num_samples: int,
) -> None:
    params_group = group.create_group("parameters")
    compression = get_compression_settings("params")

    for param_name, values in parameters_dict.items():
        arr = np.full(num_samples, np.nan, dtype=np.float32)
        for idx, val in enumerate(values):
            arr[idx] = encode_parameter_value(val)

        chunk_size = min(2048, num_samples)
        params_group.create_dataset(
            param_name,
            data=arr,
            dtype=np.float32,
            chunks=(chunk_size,),
            **compression,
        )


def write_task_to_hdf5(
    tasks_group: h5py.Group,
    task_idx: int,
    samples: np.ndarray,
    backgrounds: np.ndarray,
    patterns: np.ndarray,
    masks: np.ndarray,
    multilabels: np.ndarray,
    parameters: dict,
    wafer_mask: np.ndarray,
) -> None:
    task_group = tasks_group.create_group(str(task_idx))

    num_samples, height, width = samples.shape
    num_channels = masks.shape[1]

    compression_uint8 = get_compression_settings("uint8")
    task_group.create_dataset(
        "wafer_mask",
        data=wafer_mask,
        **compression_uint8,
    )

    create_image_dataset(task_group, "sample", (num_samples, height, width), samples)
    create_image_dataset(task_group, "background", (num_samples, height, width), backgrounds)
    create_image_dataset(task_group, "pattern", (num_samples, height, width), patterns)

    create_masks_dataset(task_group, (num_samples, num_channels, height, width), masks)
    create_multilabel_dataset(task_group, num_samples, num_channels, multilabels)

    create_parameters_group(task_group, parameters, num_samples)


def generate_and_export_dataset(
    path: str,
    tree: SamplingTree,
    num_updates: int,
    num_samples_per_update: int,
    num_samples_in_first_task: int | None = None,
    seed: int | None = None,
    generation_metadata: dict[str, Any] | None = None,
    class_names: list[str] | None = None,
) -> None:
    if num_samples_in_first_task is None:
        num_samples_in_first_task = num_samples_per_update

    resolved_class_names = list(class_names) if class_names is not None else list(_default_classes)
    resolved_num_classes = len(resolved_class_names)

    num_tasks = num_updates
    task_sample_counts = [num_samples_in_first_task] + [num_samples_per_update] * (num_tasks - 1)

    with h5py.File(path, "w", libver="latest") as file_handle:
        create_hdf5_metadata(
            file_handle,
            num_tasks,
            resolved_num_classes,
            resolved_class_names,
            seed,
            parallel=False,
            generation_metadata=generation_metadata,
        )

        meta_group = file_handle["meta"]
        create_global_index(meta_group, task_sample_counts)

        tasks_group = file_handle.create_group("tasks")

        for task_idx in tqdm(range(num_tasks), desc="Generating tasks"):
            num_samples = task_sample_counts[task_idx]
            shape = tree.get_params()["init_output_dictionary.size"].value

            samples = []
            backgrounds = []
            patterns = []
            masks_list = []
            multilabels = []
            parameters = defaultdict(list)

            all_samples_data = []
            all_param_keys = set()

            for _ in range(num_samples):
                sample_dict = tree.sample()
                all_samples_data.append(sample_dict)

                params = filter_generation_parameters(sample_dict["parameters"])
                all_param_keys.update(params.keys())

            for key in all_param_keys:
                parameters[key] = []

            for sample_dict in all_samples_data:
                samples.append(np.asarray(sample_dict["sample"], dtype=np.float32))
                backgrounds.append(np.asarray(sample_dict["background"], dtype=np.float32))
                patterns.append(np.asarray(sample_dict["pattern"], dtype=np.float32))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    masks = np.asarray(sample_dict["masks"], dtype=np.uint8)
                masks_list.append(masks)
                multilabels.append(compute_multilabel_from_masks(masks))

                params = filter_generation_parameters(sample_dict["parameters"])
                for key in all_param_keys:
                    parameters[key].append(params.get(key, None))

            write_task_to_hdf5(
                tasks_group,
                task_idx,
                np.stack(samples),
                np.stack(backgrounds),
                np.stack(patterns),
                np.stack(masks_list),
                np.stack(multilabels),
                dict(parameters),
                create_wafer_mask(shape),
            )

            # Advance each scheduled parameter exactly once per task.
            tree.update()


__all__ = ["generate_and_export_dataset"]
