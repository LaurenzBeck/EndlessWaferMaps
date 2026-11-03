"""# 🌳 Sampling Trees

Background and shared helpers for sampling-tree construction. Multi-label
and single-label trees are built from configs by `endlesswm.pipeline.tree_factory`.
"""

# ruff: noqa: F403, F405

from collections import defaultdict

import numpy as np
from streamgen.nodes import TransformNode

from .geometry import *
from .noise import *
from .patterns import *
from .transforms import *
from .utils import *

# 🔨 Utility Functions


def init_output_dictionary(_: None, size: tuple[int, int], num_classes: int = 8) -> dict:
    """📖 prepares the output dictionary for the transforms that add pattern signals."""
    return {
        "background": np.zeros(size, dtype=np.float32),
        "pattern": np.zeros(size, dtype=np.float32),
        "sample": np.zeros(size, dtype=np.float32),
        "masks": np.zeros((num_classes, *size), dtype=np.uint8),
        "parameters": defaultdict(dict),
    }


def combine_signals(sample: dict) -> dict:
    """➕ combines the individual signals.

    Args:
        sample (dict): sample dictionary

    Returns:
        dict: sample dict
    """
    sample["sample"] = sample["background"] + sample["pattern"]

    return sample


def mask_sample(sample: dict) -> dict:
    """🎭 creates a wafer mask and applies it to every array in sample."""
    shape = sample["background"].shape
    # create mask
    mask = full_ellipse(shape)
    sample["background"] = np.where(mask == 0.0, np.nan, sample["background"])
    sample["pattern"] = np.where(mask == 0.0, np.nan, sample["pattern"])
    sample["masks"] = np.where(mask == 0.0, np.nan, sample["masks"])
    sample["sample"] = np.where(mask == 0.0, np.nan, sample["sample"])
    return sample


# 🖼️ Background

background_params = {
    "beta_weibull": {
        "alpha_r": (0.1, 2.0),
        "beta_r": (0.1, 2.0),
        "weight_beta_r": (0.0, 2.0),
        "shape_weibull_r": (1.5, 3.0),
        "weight_weibull_r": (0.0, 0.5),
    },
    "pink_noise": {
        "roughness_r": (0.5, 2.0),
        "max_value_r": (0.001, 1.0),
    },
    "gaussian_filter": {
        "sigma_r": (0.2, 1.2),
    },
    "impulse_noise": {
        "fraction_log10_r": (-3.0, -1.0),
        "low_r": (-1.0, 0.0),
        "high_r": (0.0, 1.0),
    },
    "linear_gradient": {
        "angle_r": (0.0, 2 * np.pi),
        "slant_fraction_r": (0.0, 0.25),
    },
    "linear_radial_gradient": {
        "center_val_r": (0.5, 1.0),
        "outer_val_r": (1.0, 1.5),
    },
    "affine": {
        "offset_r": (-1.0, 1.0),
        "scaling_r": (0.5, 1.5),
    },
}


def background_nodes():
    return deepcopy(
        [
            TransformNode(init_output_dictionary, emoji="📖"),
            TransformNode(sample_and_store_params(operate_on_key(ignore_first_arg(beta_weibull), key="background")), emoji="🔔"),
            TransformNode(sample_and_store_params(operate_on_key(generate_and_combine(pink_noise), key="background")), emoji="🪷"),
            TransformNode(sample_and_store_params(operate_on_key(gaussian_filter, key="background")), emoji="☁️"),
            TransformNode(sample_and_store_params(operate_on_key(generate_and_combine(impulse_noise), key="background")), emoji="🔨"),
            TransformNode(
                sample_and_store_params(operate_on_key(generate_and_combine(linear_gradient, "*"), key="background")),
                emoji="📏",
            ),
            TransformNode(
                sample_and_store_params(operate_on_key(generate_and_combine(linear_radial_gradient, "*"), key="background")),
                emoji="🛞",
            ),
            TransformNode(sample_and_store_params(operate_on_key(affine, key="background")), emoji="📐"),
        ],
    )
