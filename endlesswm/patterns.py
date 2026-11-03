"""# 🌒 Stochastic Pattern Generators

This module contains functions that create our pattern signals.
"""

from copy import deepcopy
from functools import partial
from typing import Callable, Literal, TypeAlias

import numpy as np
from streamgen.nodes import TransformNode
from streamgen.transforms import noop

from .attributes import *  # noqa: F403
from .geometry import *  # noqa: F403
from .noise import pattern_noise
from .transforms import *  # noqa: F403
from .utils import *  # noqa: F403

# Default catalog snapshot. The generation pipeline does not read these; per-preset
# `BranchesConfig.classes` is the single source of truth for active patterns and
# mask-channel order. Kept here for explorer notebooks and as schema defaults.
classes = ["scratch", "center_spot", "edge_spot", "center_donut", "border_donut", "edge_ring", "crescent", "reticle"]
num_classes = len(classes)
pattern2idx = {pattern: idx for idx, pattern in enumerate(classes)}

co_occurrence = np.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ],
)

pattern_generators: dict[str, Callable] = dict.fromkeys(classes, noop)
"➡️ functions that create the pattern signals"

pattern_params: dict = {"num_classes": num_classes}
"⚙️ maximum parameter bounds for the pattern nodes"

pattern_corruptions: dict[str, list[Callable]] = {pattern: [] for pattern in classes}
"🔨 functions that corrupt the pattern signals"

corruption_params: dict = {
    "pattern_noise": {"strength_r": (0.0, 0.5)},
    "linear_radial_gradient": {
        "center_val_r": (0.75, 1.0),
        "outer_val_r": (1.0, 1.25),
    },
    "local_pixel_shuffle": {
        "fraction_r": (0.0, 0.6),
        "radius_r": (0.01, 0.1),
    },
    "swirl": {
        "strength_r": (0.0, 1.2),
    },
    "smooth_normalize": {"sigma_r": (0.001, 2.0)},
    "slant": {
        "scale_r": (0.975, 1.025),
        "rotation_r": (-0.1, 0.1),
        "shear_r": (-0.05, 0.05),
    },
}

# ## Utility Functions


def _resolve_angle(angle: float | None, rng: np.random.Generator | None) -> float:
    if angle is not None:
        return float(angle)
    if rng is None:
        rng = np.random.default_rng(42)
    return float(rng.uniform(0.0, 2 * np.pi))


def _ray_extent_inside_mask(
    mask: np.ndarray,
    origin: tuple[float, float],
    direction: tuple[float, float],
    step_size: float = 0.5,
) -> float:
    extent = 0.0
    probe = 0.0
    while True:
        probe += step_size
        row = int(round(origin[0] + (direction[0] * probe)))
        col = int(round(origin[1] + (direction[1] * probe)))
        if row < 0 or row >= mask.shape[0] or col < 0 or col >= mask.shape[1] or mask[row, col] == 0:
            return extent
        extent = probe


def _sample_segment_inside_wafer(
    shape: tuple[int, int],
    length_pixels: float,
    angle: float,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray, float]:
    wafer_mask = full_ellipse(shape)
    valid_positions = np.argwhere(wafer_mask != 0)
    origin_row, origin_col = valid_positions[rng.integers(len(valid_positions))]
    origin = (float(origin_row), float(origin_col))
    direction = np.array((float(np.cos(angle)), float(np.sin(angle))))
    direction = direction / np.linalg.norm(direction)

    positive_extent = _ray_extent_inside_mask(wafer_mask, origin, (direction[0], direction[1]))
    negative_extent = _ray_extent_inside_mask(wafer_mask, origin, (-direction[0], -direction[1]))
    available_length = positive_extent + negative_extent
    clamped_length = min(float(length_pixels), available_length)

    if clamped_length <= 0:
        point = (int(round(origin[0])), int(round(origin[1])))
        return point, point, wafer_mask, direction, 0.0

    start_offset_low = -negative_extent
    start_offset_high = positive_extent - clamped_length
    if start_offset_low > start_offset_high:
        start_offset = -negative_extent
        clamped_length = available_length
    else:
        start_offset = rng.uniform(start_offset_low, start_offset_high)

    end_offset = start_offset + clamped_length
    start = (
        int(np.clip(round(origin[0] + (direction[0] * start_offset)), 0, shape[0] - 1)),
        int(np.clip(round(origin[1] + (direction[1] * start_offset)), 0, shape[1] - 1)),
    )
    end = (
        int(np.clip(round(origin[0] + (direction[0] * end_offset)), 0, shape[0] - 1)),
        int(np.clip(round(origin[1] + (direction[1] * end_offset)), 0, shape[1] - 1)),
    )
    return start, end, wafer_mask, direction, clamped_length


def _sample_curved_middle_point_inside_wafer(
    start: tuple[int, int],
    end: tuple[int, int],
    wafer_mask: np.ndarray,
    direction: np.ndarray,
    length_pixels: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    midpoint = np.array(((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0), dtype=float)
    perpendicular = np.array((-direction[1], direction[0]), dtype=float)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    positive_extent = _ray_extent_inside_mask(wafer_mask, (midpoint[0], midpoint[1]), (perpendicular[0], perpendicular[1]))
    negative_extent = _ray_extent_inside_mask(wafer_mask, (midpoint[0], midpoint[1]), (-perpendicular[0], -perpendicular[1]))
    max_offset = min(positive_extent, negative_extent, max(1.0, 0.25 * length_pixels))
    offset = rng.uniform(-max_offset, max_offset)
    middle = midpoint + (perpendicular * offset)
    return (
        int(np.clip(round(middle[0]), 0, wafer_mask.shape[0] - 1)),
        int(np.clip(round(middle[1]), 0, wafer_mask.shape[1] - 1)),
    )


# ## ⚡ Scratches
def add_random_scratch(
    signal: np.array,
    rng: np.random.Generator | None = None,
    width: float = 0.05,
    length: float = 0.5,
    curved: bool = False,
    **kwargs,
) -> np.ndarray:
    """⚡ adds random scratches to `signal`.

    Args:
        signal (np.array): input
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.
        width (float, optional): width of the scratch in fractions of the min signal.shape. Defaults to 0.05.
        length (float, optional): length of the scratch in fractions of the min signal.shape. Defaults to 0.5.
        curved (bool, optional): flag that determines if the scratch is curved. Defaults to False.

    Returns:
        np.ndarray: signal with added scratch
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    distance = length * min(*signal.shape)
    width_pixels = max(1.0, float(width) * min(*signal.shape))
    angle = rng.uniform(0, 2 * np.pi)
    start, end, wafer_mask, direction, clamped_distance = _sample_segment_inside_wafer(signal.shape, distance, angle, rng)

    if curved:
        middle = _sample_curved_middle_point_inside_wafer(start, end, wafer_mask, direction, clamped_distance, rng)
        scratch = thick_curved_line(start, middle, end, signal.shape, width_pixels)
    else:
        scratch = thick_line(start, end, signal.shape, width_pixels)

    signal += np.where(wafer_mask != 0, scratch, 0)

    return signal


pattern_generators["scratch"] = repeat(add_random_scratch, (1, 3))
pattern_corruptions["scratch"] = [
    pattern_noise,
    local_pixel_shuffle,
]
pattern_params |= {
    "scratch": {
        "width_r": (0.012, 0.08),
        "curved_c": (True, False),
        "length_r": (0.1, 0.8),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}


# ## ⏺️ Center Spot
def displaced_center_circle(
    signal: np.array,
    distance: float,
    extent: float,
    angle: float | None = None,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    angle = _resolve_angle(angle, rng)
    center = (round((distance * np.sin(angle) + 1) / 2 * signal.shape[0]), round((distance * np.cos(angle) + 1) / 2 * signal.shape[1]))
    diameter = extent * min(*signal.shape)

    signal += circle(center, diameter, signal.shape)

    return signal


pattern_generators["center_spot"] = displaced_center_circle
pattern_corruptions["center_spot"] = [
    pattern_noise,
    smooth_normalize,
    swirl,
    slant,
    local_pixel_shuffle,
]
pattern_params |= {
    "center_spot": {
        "distance_r": (0.0, 0.25),
        "extent_r": (0.1, 0.5),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}

# ## ⏺️ Edge Spot
pattern_generators["edge_spot"] = displaced_center_circle
pattern_corruptions["edge_spot"] = [
    pattern_noise,
    smooth_normalize,
    swirl,
    local_pixel_shuffle,
]
pattern_params |= {
    "edge_spot": {
        "distance_r": (0.8, 0.95),
        "extent_r": (0.1, 0.4),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}

# ## ⭕ Donut


def displaced_center_ring(
    signal: np.array,
    diameter: float,
    width: float,
    distance: float,
    angle: float | None = None,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    angle = _resolve_angle(angle, rng)
    center = (round((distance * np.sin(angle) + 1) / 2 * signal.shape[0]), round((distance * np.cos(angle) + 1) / 2 * signal.shape[1]))
    diameter = diameter * min(*signal.shape)
    width = width * min(*signal.shape)

    signal += ring(center, diameter, width, signal.shape)

    return signal


pattern_generators["center_donut"] = displaced_center_ring
pattern_corruptions["center_donut"] = [
    pattern_noise,
    smooth_normalize,
    slant,
    swirl,
    local_pixel_shuffle,
]
pattern_params |= {
    "center_donut": {
        "diameter_r": (0.3, 0.5),
        "width_r": (0.02, 0.08),
        "distance_r": (0.0, 0.1),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}

# ## ⭕ Border Donut

pattern_generators["border_donut"] = displaced_center_ring
pattern_corruptions["border_donut"] = [
    pattern_noise,
    smooth_normalize,
    swirl,
    local_pixel_shuffle,
]
pattern_params |= {
    "border_donut": {
        "diameter_r": (0.8, 1.0),
        "width_r": (0.05, 0.15),
        "distance_r": (0.0, 0.005),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}

# ## ⭕ Edge Ring


def add_edge_ring(
    signal: np.array,
    diameter: float,
    width: float,
    **kwargs,
) -> np.ndarray:
    outer_ring = full_ellipse(signal.shape)
    inner_diameter = max(0.0, (diameter - (2.0 * width)) * min(*signal.shape))
    inner_hole = circle((signal.shape[0] / 2, signal.shape[1] / 2), inner_diameter, signal.shape)
    signal += np.where(inner_hole == 1, 0, outer_ring)
    return signal


pattern_generators["edge_ring"] = add_edge_ring
pattern_corruptions["edge_ring"] = [
    pattern_noise,
    local_pixel_shuffle,
]
pattern_params |= {
    "edge_ring": {
        "diameter_r": (0.95, 1.0),
        "width_r": (0.01, 0.05),
        "distance": 0.0,
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}

# ## 🌒 Crescent

pattern_generators["crescent"] = generate_and_combine(crescent)
pattern_corruptions["crescent"] = [
    pattern_noise,
    generate_and_combine(linear_radial_gradient, "*"),
    smooth_normalize,
    swirl,
    local_pixel_shuffle,
]
pattern_params |= {
    "crescent": {
        "max_width_r": (0.02, 0.35),
        "angle_r": (0.0, 2 * np.pi),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}


# ## 🟧 Reticle
def add_reticle(
    signal: np.array,
    width: float = 0.5,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """🟧 adds a random reticle pattern to the input.

    Args:
        signal (np.array): signal
        width (float): width of the reticle in fractions of the wafer map width
        aspect_ratio (float): the aspect_ratio of the reticle determines the height
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.

    Returns:
        np.ndarray: signal with added reticle
    """
    start = (rng.uniform(0.0, 1.0 - (width * aspect_ratio)) * signal.shape[0], rng.uniform(0.0, 1.0 - width) * signal.shape[1])

    extent = (round(width * aspect_ratio * signal.shape[0]), round(width * signal.shape[1]))

    signal += rectangle(start, extent, signal.shape)

    return signal


pattern_generators["reticle"] = add_reticle
pattern_corruptions["reticle"] = [
    pattern_noise,
    local_pixel_shuffle,
]
pattern_params |= {
    "reticle": {
        "width_r": (0.15, 0.35),
        "aspect_ratio_r": (0.75, 1.25),
        "polarity_c": ("+", "-", "NaN"),
        "value_r": (0.5, 1.5),
    },
}

# ➡️ node construction


def add_temp(input: dict):
    input["temp"] = np.zeros_like(input["pattern"])
    return input


def del_temp(input: dict):
    del input["temp"]
    return input


def add_to_mask(input: dict, idx):
    input["masks"][idx] = binarize(input["temp"], threshold=0.5)
    return input


Polarity: TypeAlias = Literal["+", "-", "NaN"]


def add_to_pattern(sample: dict, polarity: Polarity = "+", value: float = 1.0, **kwargs) -> dict:
    """➕ scales and adds a tmp signal to the pattern.

    Args:
        sample (dict): sample dictionary
        polarity (Polarity, optional): the pattern polarity determines if the pattern is added ("+"),
            subtracted ("-"), or if the pattern values are set to NaN. Defaults to "+".
        value (float, optional): value of the pattern. Defaults to 1.0.

    Returns:
        dict: updated sample dictionary
    """
    match polarity:
        case "+":
            sample["pattern"] += sample["temp"] * value
        case "-":
            sample["pattern"] -= sample["temp"] * value
        case "NaN":
            sample["pattern"] = np.where(binarize(sample["temp"], threshold=0.5) == 1.0, np.nan, sample["pattern"])

    return sample


def create_pattern_nodes(pattern: str, idx: int | None = None) -> list[TransformNode]:
    """➡️ creates the generation, corruption and data handling nodes for a pattern.

    Args:
        pattern: pattern name; must be a key of `pattern_generators`.
        idx: mask channel index for this pattern. If None, falls back to the
            default catalog `pattern2idx` — only valid when the caller is using
            the full default class list.
    """
    if idx is None:
        idx = pattern2idx[pattern]

    corruption_nodes: list[TransformNode] = []
    for corruption in pattern_corruptions[pattern]:
        storage_name = f"{pattern}.{corruption.__name__}"

        corruption_nodes.append(
            TransformNode(
                sample_and_store_params(operate_on_key(corruption, key="temp"), name=storage_name),
                name=corruption.__name__,
                emoji="🔨",
            ),
        )

    return deepcopy(
        [
            TransformNode(add_temp, name="pattern_setup", emoji="✨"),
            TransformNode(
                sample_and_store_params(operate_on_key(pattern_generators[pattern], key="temp"), name=pattern),
                name=pattern,
                emoji="🌘",
            ),
            *corruption_nodes,
            TransformNode(partial(add_to_mask, idx=idx), name="create_mask", emoji="🏷️"),
            TransformNode(sample_and_store_params(add_to_pattern, name=pattern), name=pattern, emoji="➕"),
            TransformNode(del_temp, name="pattern_cleanup", emoji="🧹"),
        ],
    )
