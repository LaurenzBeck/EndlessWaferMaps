"""🐶 code responsible for generating pattern signals from attribute class descriptions."""

from typing import Literal

import numpy as np

Range = Literal["empty", "tiny", "small", "medium", "large", "full"]
Location = Literal["center", "middle", "border"]


RANGES: dict[Range, tuple[float, float]] = {
    "empty": (0.0, 0.0),
    "tiny": (0.0, 0.05),
    "small": (0.05, 0.25),
    "medium": (0.25, 0.5),
    "large": (0.5, 1.0),
    "full": (1.0, 1.0),
}
"""📏 quantized range steps with their corresponding lower and upper bounds."""

LOCATIONS: dict[Location, tuple[float, float]] = {"center": (0.0, 0.3), "middle": (0.3, 0.7), "border": (0.7, 1.0)}
"""
🗺️ quantized location steps with their corresponding lower and upper bounds
of the distance between the center and the edge of a wafer map.
"""

PATTERN_DEFINITIONS: dict[str, dict] = {
    "Center Spot": {"shape": "circle", "extent": ["small", "medium"], "location": ["center"], "amount": [1]},
    "Donut": {"shape": "ring", "extent": ["small", "medium"], "thickness": ["small", "medium"], "location": ["center"], "amount": [1]},
    "Crescent": {
        "shape": "ring-segment",
        "extent": ["medium", "large"],
        "thickness": ["tiny", "small"],
        "location": ["border"],
        "amount": [1],
    },
    "Border Ring": {"shape": "ring", "extent": ["large"], "thickness": ["tiny", "small"], "location": ["border"], "amount": [1]},
    "Edge Spot": {"shape": "circle", "extent": ["tiny", "small"], "location": ["border"], "amount": [1, 3]},
    "Near Full": {"shape": "random", "extend": ["large"], "thickness": ["small"]},
    "Scratch": {"shape": ["line", "curved-line"], "extend": ["small", "medium", "large"], "thickness": ["tiny"], "amount": [1, 4]},
    "Random": {"shape": "random", "extend": ["large"], "thickness": ["small"]},
}
"""🏷️ pattern class definitions of the MixedWM38 dataset."""


def sample_from_range(ranges: list[Range], rng: np.random.Generator | int = 42) -> float:
    """🎲 sample uniformly from a list of range definitions.

    Args:
        ranges (list[Range]): lists all the applicable ranges.
        rng (np.random.Generator | int, optional): random number generator. Defaults to 42.

    Returns:
        float: sampled value from a valid range
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    range_ = rng.choice(ranges)
    return rng.uniform(*RANGES[range_])


def sample_point(
    distance: tuple[float, float],
    angle: tuple[float, float],
    shape: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """🌐 uniformly samples a random point in a wafer map given a distance and angle range.

    Args:
        distance (tuple[float, float]): range of distances to the center from 0 to 1
        angle (tuple[float, float]): range of angles in radians
        shape (tuple[int, int]): shape of the wafer map
        rng (np.random.Generator | int, optional): random number generator. Defaults to None.

    Returns:
        tuple[int, int]: sampled coordinates
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    distance = rng.uniform(*distance)
    angle = rng.uniform(*angle)
    x = round((distance * np.sin(angle) + 1) / 2 * shape[0])
    y = round((distance * np.cos(angle) + 1) / 2 * shape[1])
    return (x, y)


def sample_point_from_location(locations: list[Location], shape: tuple[int, int], rng: np.random.Generator | int = 42) -> tuple[int, int]:
    """🗺️ samples a random point from a list of locations.

    Args:
        locations (list[Location]): list of valid locations
        shape (tuple[int, int]): shape of the wafer map
        rng (np.random.Generator | int, optional): random number generator. Defaults to 42.

    Returns:
        tuple[int, int]: sampled coordinates
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    location = rng.choice(locations)
    distance_range = LOCATIONS[location]
    return sample_point(distance_range, angle=(0, 2 * np.pi), shape=shape, rng=rng)
