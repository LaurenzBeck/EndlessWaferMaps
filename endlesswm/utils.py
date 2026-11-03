"""# 🛠️🪛 Utility Functions

This module contains utility functions unrelated to any other module.
"""

from __future__ import annotations

import inspect
from functools import partial, reduce, update_wrapper  # noqa: F401
from typing import TYPE_CHECKING, Any, Callable

import itables
import matplotlib as mpl
import numpy as np
import pandas as pd
import rich.traceback
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from endlesswm import CombinationStrategy, CombinationStrategyLit

if TYPE_CHECKING:
    from torch import Tensor


def _import_torch():
    import torch

    return torch


def collect(func: Callable, *args, num_samples: int = 16, **kwargs) -> list:  # noqa: ANN002, ANN003
    """🪺 collects `num_samples` results from calling `func(*args, **kwargs)`.

    Args:
        func (Callable): function to collect samples from
        num_samples (int, optional): how many samples to collect. Defaults to 16
        *args: variable length argument list are passed to `func`
        **kwargs: arbitrary keyword arguments are passed to `func`

    Returns:
        list: list of samples
    """
    return [func(*args, **kwargs) for _ in range(num_samples)]


def compose(*funcs: Callable) -> Callable:
    """🪢 sequential function composition.

    `compose(add, square)(1,3) == square(add(1,3)) == 16`.

    Args:
        *funcs: variable length argument list of functions

    Returns:
        Callable: one function that sequentially calls each function
            with the output of the previous function.
    """
    return lambda x: reduce(lambda acc, f: f(acc), funcs, x)


def generate_and_compose(*funcs: Callable) -> Callable:
    """🎲 sequential function composition, where the first function generates the input.

    Args:
        *funcs: variable length argument list of functions

    Returns:
        Callable: one function that sequentially calls each function
            with the output of the previous function.
    """

    def pipeline():  # noqa: ANN202
        output = funcs[0]()
        for func in funcs[1:]:
            output = func(output)
        return output

    return pipeline


def generate_and_combine(
    generator: Callable,
    combine: CombinationStrategy | CombinationStrategyLit | Callable[[Any, Any], Any] = "+",
) -> Callable:
    """🎲➕ a transform that generates using the `generator` and combines the output with the `sample` input using `combine`.

    Args:
        generator (Callable): generator function
        combine (CombinationStrategy | CombinationStrategyLit | Callable[[Any, Any], Any]): strategy that defines
            how to combine the generated output with the sample. Defaults to addition.

    Returns:
        Callable: wrapped generator function that combines its output with `sample`
    """  # noqa: RUF002

    def wrapped_func(sample: np.array, *args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
        output = generator(*args, **kwargs)

        match combine:
            case "+":
                return sample + output
            case "*":
                return sample * output
            case _:
                return combine(sample, output)

    return update_wrapper(wrapped_func, generator)


def ignore_first_arg(func: Callable) -> Callable:
    """🫣 creates a wrapped version of `func`, that ignores the first argument."""

    def wrapped_func(_, *args, **kwargs) -> Any:  # noqa: ANN001, ANN002, ANN003, ANN401
        return func(*args, **kwargs)

    return update_wrapper(wrapped_func, func)


def operate_on_key(func: Callable, key: str = "input") -> Callable:
    """🗝️ function wrapper, that converts `func(input: dict, ...)` to `func(input[key], ...)`.

    The result is put back into `input`, the first argument to `func`.

    This facilitates working with transforms that take dictionaries as first arguments.
    """

    def wrapped_func(input: dict, *args, **kwargs) -> Any:  # noqa: A002, ANN002, ANN003, ANN401
        x = input[key]
        out = func(x, *args, **kwargs)
        input[key] = out
        return input

    return update_wrapper(wrapped_func, func)


def sample_kwargs(func: Callable, rng: np.random.Generator | int = 42, **kwargs_ranges: dict[str, tuple[float, float]]) -> Callable:
    """🎲📖 wraps a function to sample keyword arguments uniformly in a specified range upon calling.

    This behaviour is similar to `sample_params`, but comparable to `functools.partial` in the API.

    Args:
        func (Callable): function to wrap
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42
        **kwargs_ranges (tuple[float, float]): ranges for the keywords arguments to sample from.

    Returns:
        Callable: wrapped `func`
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    def wrapped_func(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
        random = kwargs.get("rng", rng)

        # remove sampled keyword arguments from **kwargs
        kwargs = {k: v for k, v in kwargs.items() if k not in kwargs_ranges}

        # sample them
        sampled_kwargs = {k: random.uniform(low=v[0], high=v[1]) for k, v in kwargs_ranges.items()}

        return func(*args, **sampled_kwargs, **kwargs)

    return update_wrapper(wrapped_func, func)


def has_arg(func: Callable, name: str) -> bool:
    """❓ checks if `func` has an argument with a given name."""
    return name in inspect.getfullargspec(func).args


def sample_params(func: Callable, rng: np.random.Generator | int = 42) -> Callable:
    """🎲📖 wraps a function to sample keyword arguments uniformly in a specified range upon invocation.

    This function is similar to `sample_kwargs`, but instead of configuring the kwarg ranges
    upon wrapping, which does not play well with the streamgen parameter passing, the ranges
    are configured in the parameters themselves, by:

    1. suffixing a sampled param with `"_r"`
    2. using a tuple of a lower and an upper bound as the parameter

    You can also suffix a sampled param with `"_c"` and define multiple possible options through a list or tuple.
    For this option, `np.random.choice` will be called on the list.

    Args:
        func (Callable): function to wrap
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        Callable: wrapped `func`
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    def wrapped_func(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
        random = kwargs.get("rng", rng)

        # remove sampled keyword arguments from **kwargs
        params = {k: v for k, v in kwargs.items() if not k.endswith(("_r", "_c"))}

        # sample them
        sampled_params = {k[:-2]: random.uniform(low=v[0], high=v[1]) for k, v in kwargs.items() if k.endswith("_r")} | {
            k[:-2]: random.choice(v) for k, v in kwargs.items() if k.endswith("_c")
        }

        return func(*args, **sampled_params, **params)

    return update_wrapper(wrapped_func, func)


def sample_and_store_params(func: Callable, name: str | None = None) -> Callable:
    """🎲🛢️ a variation of `sample_params` that stores the sampled params and static params.

    Stores:
    - Sampled parameters (with _r or _c suffix)
    - Static parameters (no suffix, excluding technical params)

    Excludes from storage:
    - Technical params: rng, shape, size, num_classes

    Args:
        func (Callable): function to wrap
        name (str | None, optional): name under which the parameters should be stored.
            This functionality is important if the wrapper is wrapped in a `TransformNode` with a different name.

    Returns:
        Callable: wrapped `func`
    """
    # Technical parameters to exclude from storage
    EXCLUDED_PARAMS = {"rng", "shape", "size", "num_classes"}

    orig_sig = inspect.signature(func)
    orig_params = list(orig_sig.parameters.values())
    accepts_rng_param = "rng" in orig_sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in orig_params)

    def wrapped_func(input: dict, *args, **kwargs) -> Any:
        rng = kwargs.pop("rng")

        # Separate sampled and non-sampled kwargs
        non_sampled_kwargs = {k: v for k, v in kwargs.items() if not k.endswith(("_r", "_c"))}

        # Sample range parameters (_r suffix)
        sampled_params = {k[:-2]: rng.uniform(low=v[0], high=v[1]) for k, v in kwargs.items() if k.endswith("_r")}
        # Sample choice parameters (_c suffix)
        sampled_params |= {k[:-2]: rng.choice(v).item() for k, v in kwargs.items() if k.endswith("_c")}

        # Store sampled parameters
        key = func.__name__ if name is None else name
        input["parameters"][key] |= sampled_params

        # Store static parameters (excluding technical params)
        static_params = {k: v for k, v in non_sampled_kwargs.items() if k not in EXCLUDED_PARAMS}
        input["parameters"][key] |= static_params

        forward_rng = accepts_rng_param or has_var_kw
        if forward_rng:
            non_sampled_kwargs["rng"] = rng

        return func(input, *args, **sampled_params, **non_sampled_kwargs)

    if not accepts_rng_param:
        rng_param = inspect.Parameter(
            "rng",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="np.random.Generator | None",
        )
        try:
            var_kw_index = next(i for i, p in enumerate(orig_params) if p.kind == inspect.Parameter.VAR_KEYWORD)
            new_params = [*orig_params[:var_kw_index], rng_param, *orig_params[var_kw_index:]]
        except StopIteration:
            new_params = [*orig_params, rng_param]
        wrapped_func.__signature__ = orig_sig.replace(parameters=new_params)
    else:
        wrapped_func.__signature__ = orig_sig

    return update_wrapper(wrapped_func, func)


def okss(func: Callable, key: str = "background", name: str | None = None, rng: np.random.Generator | int = 42) -> Callable:
    """🗝️🎲🛢️ a combination of `operate_on_key` and `sample_params` that stores the sampled params.

    Args:
        func (Callable): function to wrap
        key (str): name of the field of the input dictionary to take as first arguments
        name (str | None, optional): name under which the parameters should be stored.
            This functionality is important if the okss wrapper is wrapped in a `TransformNode` with a different name.
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        Callable: wrapped `func`
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    def wrapped_func(input: dict, *args, **kwargs) -> Any:  # noqa: A002, ANN002, ANN003, ANN401
        random = kwargs.get("rng", rng)

        # remove sampled keyword arguments from **kwargs
        params = {k: v for k, v in kwargs.items() if not k.endswith(("_r", "_c"))}

        # sample them
        sampled_params = {k[:-2]: random.uniform(low=v[0], high=v[1]).item() for k, v in kwargs.items() if k.endswith("_r")} | {
            k[:-2]: random.choice(v).item() for k, v in kwargs.items() if k.endswith("_c")
        }

        # store parameters excluding `rng`
        input["parameters"][func.__name__ if name is None else name] = sampled_params | {k: v for k, v in params.items() if k != "rng"}

        # operate on key
        x = input[key]
        out = func(x, *args, **sampled_params, **params)
        input[key] = out

        return input

    return update_wrapper(wrapped_func, func)


def maybe(func: Callable, rng: np.random.Generator | None = None, prob_apply: float = 0.9) -> Callable:
    """❓ wraps a function to apply its logic randomly with a probability.

    Each invocation will sample from a Bernoulli distribution and if the sampled value is above
    `prob_apply`, the first argument of `func` will be used as the output of the new function.

    Args:
        func (Callable): parameterized transformation, that does sth to its first argument
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.
        prob_apply (float, optional): probability of applying `func`. Defaults to 0.9.

    Returns:
        Callable: wrapped `func`
    """

    def wrapped_func(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
        random = kwargs.get("rng", rng)
        if random.random() < prob_apply:
            return func(*args, **kwargs)
        return args[0]

    return update_wrapper(wrapped_func, func)


def repeat(func: Callable, amount_range: tuple[int, int] = (1, 1), rng: np.random.Generator | int = 42) -> Callable:
    """🔁 wraps a function to be repeatedly called for an amount of times sampled uniformly from `amount_range`.

    Args:
        func (Callable): parameterized transformation, that does sth to its first argument
        amount_range (tuple[int, int]): upper and lower bound of times to repeat `func`
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        Callable: wrapped `func`
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    def wrapped_func(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
        random = kwargs.get("rng", rng)
        repetitions = random.integers(amount_range[0], amount_range[1] + 1)
        sample, *args = args
        for _ in range(repetitions):
            sample = func(sample, *args, **kwargs)

        return sample

    return update_wrapper(wrapped_func, func)


def visualize(func: Callable, plotting_func: Callable[[Any, plt.Axes], plt.Axes]) -> Callable[[Any], mpl.figure.Figure]:
    """🖼️ a function wrapper, that automatically plots the output of `func` using `plotting_func`.

    Args:
        func (Callable): function, whose output we want to plot
        plotting_func (Callable[[Any, plt.Axes], plt.Axes]): plotting function, that takes the output of `func` as first argument

    Returns:
        Callable[[Any], mpl.figure.Figure]: wrapped `func`
    """

    def wrapped_func(*args, **kwargs) -> mpl.figure.Figure:  # noqa: ANN002, ANN003
        return plotting_func(func(*args, **kwargs))

    return update_wrapper(wrapped_func, func)


def scale(x: np.array, min_val: float, max_val: float) -> np.array:
    """📐 scales an array to a new range.

    Args:
        x (np.array): array
        min_val (float): lower value of the new scale
        max_val (float): upper value of the new scale

    Returns:
        np.array: scaled `x`
    """
    # Normalize the array to the range 0 to 1
    x_norm = (x - x.min()) / (x.max() - x.min())

    # Scale the normalized array to the desired range
    return (x_norm * (max_val - min_val)) + min_val


def affine(x: np.array, offset: float = 0.0, scaling: float = 1.0) -> np.array:
    """➕✖️ performs an affine transformation of `x`."""  # noqa: RUF002
    return scaling * x + offset


def binarize(x: np.array, threshold: float = 0.0) -> np.array:
    """🔢 binarizes an array by applying a threshold.

    Args:
        x (np.array): array
        threshold (float, optional): values greater than this will become 1

    Returns:
        np.array: binarized `x`
    """
    array = np.zeros_like(x, dtype=np.uint8)
    array[x > threshold] = 1
    return array


def flatten_dict(nested_dict: dict, prefix="") -> dict:
    """📖 flattens a nested dictionary into a single-level dictionary.

    Args:
        nested_dict (dict): The dictionary to be flattened.
        prefix (str): The prefix to be added to the keys. Defaults to an empty string.

    Returns:
        dict: The flattened dictionary.
    """
    flattened_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}{key}" if prefix == "" else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key))
        else:
            flattened_dict[new_key] = value
    return flattened_dict


def init_notebook_env() -> None:
    """📓🪐 configures some tools for my preferred notebook environment."""
    rich.traceback.install(show_locals=False, width=200)
    plt.ioff()  # disable interactive mode of matplotlib in order to show the plot just once

    pd.options.display.max_seq_items = 2  # needed to compactly display large nested lists
    itables.init_notebook_mode(all_interactive=False)
    itables.options.maxBytes = 0  # determines if the output should be reduced. 0 means no reduction -> full dataframe
    itables.options.scrollx = True
    itables.options.scrolly = True
    itables.options.classes = "display compact"
    itables.options.style = "table-layout:auto;width:100%;margin:auto;caption-side:bottom"


def clf() -> None:
    """🚮 clear all open matplotlib objects."""
    plt.close()
    plt.cla()
    plt.clf()


# 🏷️ label processing and analysis utilities


def compute_cardinality(targets: Tensor) -> Tensor:
    return targets.sum(dim=1).long()


def compute_long_form_targets_dataframe(
    targets: Tensor,
    class_names: list[str] | None = None,
    no_pattern_name: str = r"$\emptyset$",
) -> pd.DataFrame:
    num_samples, num_classes = targets.shape
    cardinality = compute_cardinality(targets)

    if not class_names:
        class_names = [str(i) for i in range(num_classes)]

    data = []
    for sample_idx in range(num_samples):
        if cardinality[sample_idx] == 0:
            data.append(
                {
                    "sample": sample_idx,
                    "pattern": no_pattern_name,
                    "index": -1,
                    "cardinality": 0,
                },
            )
            continue
        for class_idx in range(num_classes):
            if targets[sample_idx, class_idx] == 1.0:
                data.append(  # noqa: PERF401
                    {
                        "sample": sample_idx,
                        "pattern": class_names[class_idx],
                        "index": class_idx,
                        "cardinality": cardinality[sample_idx].item(),
                    },
                )
    return pd.DataFrame(data)


def compute_cardinality_count_matrix(targets: Tensor) -> np.ndarray:
    num_samples, num_classes = targets.shape
    cardinality = compute_cardinality(targets)
    max_cardinality = cardinality.max().item()

    cardinality_matrix = np.zeros((num_classes + 1, max_cardinality + 1))
    for sample_idx in tqdm(range(num_samples)):
        if cardinality[sample_idx] == 0:
            cardinality_matrix[0, 0] += 1
        for class_idx in range(num_classes):
            if targets[sample_idx, class_idx] == 1.0:
                cardinality_matrix[class_idx + 1, cardinality[sample_idx]] += 1

    return cardinality_matrix


def compute_co_occurrence_matrix(targets: Tensor, normalize: bool = True, remove_diagonal: bool = True) -> np.ndarray:
    num_samples, num_classes = targets.shape
    cardinality = compute_cardinality(targets)
    co_occurrence_matrix = np.zeros((num_classes, num_classes))
    for sample_idx in tqdm(range(num_samples)):
        if cardinality[sample_idx] == 0:
            continue
        for class_idx in range(num_classes):
            if targets[sample_idx, class_idx] == 1.0:
                for co_occurrence_idx in range(num_classes):
                    if targets[sample_idx, co_occurrence_idx] == 1.0:
                        co_occurrence_matrix[class_idx, co_occurrence_idx] += 1

    class_occurrences = np.diag(co_occurrence_matrix.copy())

    if remove_diagonal:
        np.fill_diagonal(co_occurrence_matrix, np.nan)

    return co_occurrence_matrix / np.reshape(class_occurrences, (num_classes, 1)) if normalize else co_occurrence_matrix


def compute_label_combination_stats(targets: Tensor) -> tuple[list[int], list[int]]:
    num_samples, num_classes = targets.shape
    possible_combinations = [1, num_classes]
    encountered_combinations = [1, num_classes]
    unique_targets = targets.unique(dim=0)
    torch = _import_torch()

    for cardinality in range(2, num_classes + 1):
        possible_combinations.append(torch.combinations(torch.arange(num_classes), r=cardinality).shape[0])
        encountered_combinations.append(sum(unique_targets.sum(dim=1) == cardinality).item())

    return possible_combinations, encountered_combinations
